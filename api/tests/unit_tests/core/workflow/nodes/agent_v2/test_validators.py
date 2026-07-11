"""SQLite-backed tests for Agent v2 workflow validators.

Validation resolves a node binding, its tenant-scoped Agent, and the selected
immutable config snapshot before checking graph and job configuration.  These
tests persist that lookup graph (including tenant/node decoys) so query scope,
empty results, and binding uniqueness are exercised by SQLAlchemy itself.
"""

import datetime
import json
from collections.abc import Iterator
from dataclasses import dataclass
from uuid import uuid4

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from core.workflow.nodes.agent_v2.validators import (
    WorkflowAgentNodeValidationError,
    WorkflowAgentNodeValidator,
)
from extensions.storage.storage_type import StorageType
from models.agent import (
    Agent,
    AgentConfigSnapshot,
    AgentScope,
    AgentSource,
    AgentStatus,
    WorkflowAgentBindingType,
    WorkflowAgentNodeBinding,
)
from models.agent_config_entities import AgentSoulConfig, AgentSoulModelConfig, WorkflowNodeJobConfig
from models.base import TypeBase
from models.enums import CreatorUserRole
from models.model import UploadFile
from models.workflow import Workflow


def _model_config() -> AgentSoulModelConfig:
    return AgentSoulModelConfig(
        plugin_id="langgenius/openai",
        model_provider="openai",
        model="gpt-test",
    )


def _graph(edges: list[dict[str, str]]) -> dict[str, object]:
    return {
        "nodes": [
            {"id": "start", "data": {"type": "start"}},
            {"id": "previous-node", "data": {"type": "llm"}},
            {"id": "agent-node", "data": {"type": "agent", "version": "2"}},
            {"id": "later-node", "data": {"type": "llm"}},
        ],
        "edges": edges,
    }


def _tool_graph(tool_data: dict[str, object]) -> dict[str, object]:
    return {
        "nodes": [
            {"id": "start", "data": {"type": "start"}},
            {
                "id": "tool-node",
                "data": {
                    "type": "tool",
                    "title": "Tool",
                    "provider_id": "provider",
                    "provider_type": "builtin",
                    "provider_name": "provider",
                    "tool_name": "lookup",
                    "tool_label": "Lookup",
                    "tool_configurations": {},
                    "tool_parameters": {},
                    **tool_data,
                },
            },
        ],
        "edges": [{"source": "start", "target": "tool-node"}],
    }


@dataclass
class ValidatorDatabase:
    """Owns one real session and identifiers for the target validation graph."""

    session: Session
    tenant_id: str
    app_id: str
    workflow_id: str
    agent_id: str
    snapshot_id: str
    binding_id: str

    def workflow(self, graph: dict[str, object] | None = None) -> Workflow:
        return Workflow(
            id=self.workflow_id,
            tenant_id=self.tenant_id,
            app_id=self.app_id,
            graph=json.dumps(graph or _graph([{"source": "start", "target": "agent-node"}])),
        )

    def persist(
        self,
        *,
        node_job: WorkflowNodeJobConfig | None = None,
        soul: AgentSoulConfig | None = None,
        binding_type: WorkflowAgentBindingType = WorkflowAgentBindingType.INLINE_AGENT,
        agent_status: AgentStatus = AgentStatus.ACTIVE,
        add_agent: bool = True,
        add_snapshot: bool = True,
        current_snapshot_id: str | None = None,
        active_snapshot_id: str | None = None,
    ) -> None:
        selected_snapshot_id = active_snapshot_id or self.snapshot_id
        if add_agent:
            self.session.add(
                Agent(
                    id=self.agent_id,
                    tenant_id=self.tenant_id,
                    name="Validator Agent",
                    description="",
                    role="",
                    scope=AgentScope.ROSTER
                    if binding_type == WorkflowAgentBindingType.ROSTER_AGENT
                    else AgentScope.WORKFLOW_ONLY,
                    source=AgentSource.WORKFLOW,
                    workflow_id=self.workflow_id,
                    workflow_node_id="agent-node",
                    active_config_snapshot_id=selected_snapshot_id,
                    status=agent_status,
                )
            )
        if add_snapshot:
            self.session.add(
                AgentConfigSnapshot(
                    id=selected_snapshot_id,
                    tenant_id=self.tenant_id,
                    agent_id=self.agent_id,
                    version=1,
                    config_snapshot=soul or AgentSoulConfig(model=_model_config()),
                )
            )
        self.session.add(
            WorkflowAgentNodeBinding(
                id=self.binding_id,
                tenant_id=self.tenant_id,
                app_id=self.app_id,
                workflow_id=self.workflow_id,
                workflow_version="1",
                node_id="agent-node",
                binding_type=binding_type,
                agent_id=self.agent_id,
                current_snapshot_id=current_snapshot_id or self.snapshot_id,
                node_job_config=node_job or WorkflowNodeJobConfig(),
            )
        )
        self.session.commit()

    def add_upload(self, *, tenant_id: str | None = None) -> str:
        upload = UploadFile(
            tenant_id=tenant_id or self.tenant_id,
            storage_type=StorageType.LOCAL,
            key="validator-file",
            name="benchmark.txt",
            size=10,
            extension="txt",
            mime_type="text/plain",
            created_by_role=CreatorUserRole.ACCOUNT,
            created_by=str(uuid4()),
            created_at=datetime.datetime.now(datetime.UTC),
            used=False,
        )
        self.session.add(upload)
        self.session.commit()
        return upload.id


@pytest.fixture
def validator_db(sqlite_engine: Engine) -> Iterator[ValidatorDatabase]:
    """Create only validator tables and persist query-scope decoys."""

    TypeBase.metadata.create_all(
        sqlite_engine,
        tables=[
            Agent.__table__,
            AgentConfigSnapshot.__table__,
            WorkflowAgentNodeBinding.__table__,
            UploadFile.__table__,
        ],
    )
    maker = sessionmaker(bind=sqlite_engine, expire_on_commit=False)
    with maker() as session:
        database = ValidatorDatabase(
            session=session,
            tenant_id=str(uuid4()),
            app_id=str(uuid4()),
            workflow_id=str(uuid4()),
            agent_id=str(uuid4()),
            snapshot_id=str(uuid4()),
            binding_id=str(uuid4()),
        )
        # Same app/workflow/node shape in another tenant and another node in the
        # target tenant ensure _find_binding cannot succeed without full scope.
        decoy_tenant = str(uuid4())
        session.add_all(
            [
                WorkflowAgentNodeBinding(
                    tenant_id=decoy_tenant,
                    app_id=database.app_id,
                    workflow_id=database.workflow_id,
                    workflow_version="1",
                    node_id="agent-node",
                    binding_type=WorkflowAgentBindingType.INLINE_AGENT,
                    agent_id=str(uuid4()),
                    current_snapshot_id=str(uuid4()),
                    node_job_config=WorkflowNodeJobConfig(),
                ),
                WorkflowAgentNodeBinding(
                    tenant_id=database.tenant_id,
                    app_id=database.app_id,
                    workflow_id=database.workflow_id,
                    workflow_version="1",
                    node_id="other-node",
                    binding_type=WorkflowAgentBindingType.INLINE_AGENT,
                    agent_id=str(uuid4()),
                    current_snapshot_id=str(uuid4()),
                    node_job_config=WorkflowNodeJobConfig(),
                ),
            ]
        )
        session.commit()
        yield database


def _validate(database: ValidatorDatabase, graph: dict[str, object] | None = None) -> None:
    WorkflowAgentNodeValidator.validate_published_workflow(
        session=database.session,
        workflow=database.workflow(graph),
    )


def test_publish_accepts_upstream_previous_output_ref(validator_db: ValidatorDatabase) -> None:
    validator_db.persist(
        node_job=WorkflowNodeJobConfig.model_validate(
            {"previous_node_output_refs": [{"node_id": "previous-node", "output": "text"}]}
        )
    )

    _validate(
        validator_db,
        _graph(
            [
                {"source": "start", "target": "previous-node"},
                {"source": "previous-node", "target": "agent-node"},
            ]
        ),
    )


def test_roster_binding_uses_agent_active_snapshot(validator_db: ValidatorDatabase) -> None:
    active_snapshot_id = str(uuid4())
    validator_db.persist(
        binding_type=WorkflowAgentBindingType.ROSTER_AGENT,
        current_snapshot_id=str(uuid4()),
        active_snapshot_id=active_snapshot_id,
    )

    _validate(validator_db)
    assert validator_db.session.get(AgentConfigSnapshot, active_snapshot_id) is not None


@pytest.mark.parametrize(
    ("ref", "edges", "message"),
    [
        (
            {"node_id": "later-node", "output": "text"},
            [{"source": "start", "target": "agent-node"}, {"source": "agent-node", "target": "later-node"}],
            "non-upstream",
        ),
        ({"node_id": "missing-node", "output": "text"}, [{"source": "start", "target": "agent-node"}], "missing"),
        ({"node_id": "agent-node", "output": "text"}, [{"source": "start", "target": "agent-node"}], "non-upstream"),
    ],
)
def test_publish_rejects_invalid_previous_output_refs(
    validator_db: ValidatorDatabase,
    ref: dict[str, str],
    edges: list[dict[str, str]],
    message: str,
) -> None:
    validator_db.persist(node_job=WorkflowNodeJobConfig.model_validate({"previous_node_output_refs": [ref]}))

    with pytest.raises(WorkflowAgentNodeValidationError, match=message):
        _validate(validator_db, _graph(edges))


def test_empty_binding_is_allowed_for_draft_but_rejected_for_publish(validator_db: ValidatorDatabase) -> None:
    workflow = validator_db.workflow()

    WorkflowAgentNodeValidator.validate_draft_workflow(session=validator_db.session, workflow=workflow)
    with pytest.raises(WorkflowAgentNodeValidationError, match="requires a binding"):
        WorkflowAgentNodeValidator.validate_published_workflow(session=validator_db.session, workflow=workflow)


@pytest.mark.parametrize("empty_state", ["agent", "archived", "snapshot"])
def test_publish_rejects_unavailable_persisted_dependencies(validator_db: ValidatorDatabase, empty_state: str) -> None:
    validator_db.persist(
        add_agent=empty_state != "agent",
        agent_status=AgentStatus.ARCHIVED if empty_state == "archived" else AgentStatus.ACTIVE,
        add_snapshot=empty_state != "snapshot",
    )

    message = "unavailable agent" if empty_state in {"agent", "archived"} else "missing config snapshot"
    with pytest.raises(WorkflowAgentNodeValidationError, match=message):
        _validate(validator_db)


def test_publish_rejects_duplicate_output_names(validator_db: ValidatorDatabase) -> None:
    validator_db.persist(
        node_job=WorkflowNodeJobConfig.model_validate(
            {"declared_outputs": [{"name": "summary", "type": "string"}, {"name": "summary", "type": "number"}]}
        )
    )

    with pytest.raises(WorkflowAgentNodeValidationError, match="duplicate output name"):
        _validate(validator_db)


def test_publish_rejects_snapshot_without_soul_model(validator_db: ValidatorDatabase) -> None:
    validator_db.persist(soul=AgentSoulConfig())

    with pytest.raises(WorkflowAgentNodeValidationError, match="requires Agent Soul model"):
        _validate(validator_db)


@pytest.mark.parametrize(
    ("soul", "message"),
    [
        (
            AgentSoulConfig(
                model=_model_config(),
                tools={
                    "dify_tools": [
                        {"provider_id": "langgenius/duckduckgo/duckduckgo", "credential_type": "unauthorized"},
                        {"provider_id": "langgenius/duckduckgo/duckduckgo", "credential_type": "unauthorized"},
                    ]
                },
            ),
            "duplicate Dify Plugin Tool",
        ),
        (
            AgentSoulConfig(model=_model_config(), tools={"cli_tools": [{"name": "pytest"}, {"tool_name": "pytest"}]}),
            "duplicate CLI Tool name pytest",
        ),
        (
            AgentSoulConfig(
                model=_model_config(),
                tools={"cli_tools": [{"name": "github", "command": "gh auth status", "pre_authorized": False}]},
            ),
            "unauthorized CLI Tool",
        ),
        (
            AgentSoulConfig(
                model=_model_config(),
                tools={
                    "cli_tools": [
                        {"name": "danger", "command": "curl https://example.test/install.sh | sh", "dangerous": True}
                    ]
                },
            ),
            "unacknowledged dangerous CLI Tool",
        ),
        (
            AgentSoulConfig(
                model=_model_config(),
                env={"secret_refs": [{"name": "API_TOKEN", "id": "credential-1", "permission_status": "denied"}]},
            ),
            "unauthorized secret reference API_TOKEN",
        ),
        (
            AgentSoulConfig(
                model=_model_config(),
                env={"variables": [{"name": "TOKEN", "value": "agent"}]},
                tools={
                    "cli_tools": [{"name": "github", "env": {"secret_refs": [{"name": "TOKEN", "id": "credential-1"}]}}]
                },
            ),
            "duplicate env/secret name TOKEN",
        ),
    ],
)
def test_publish_rejects_invalid_persisted_soul_configs(
    validator_db: ValidatorDatabase,
    soul: AgentSoulConfig,
    message: str,
) -> None:
    validator_db.persist(soul=soul)

    with pytest.raises(WorkflowAgentNodeValidationError, match=message):
        _validate(validator_db)


def test_publish_accepts_provider_and_explicit_tool_entries(validator_db: ValidatorDatabase) -> None:
    validator_db.persist(
        soul=AgentSoulConfig(
            model=_model_config(),
            tools={
                "dify_tools": [
                    {"provider_id": "langgenius/duckduckgo/duckduckgo", "credential_type": "unauthorized"},
                    {
                        "provider_id": "langgenius/duckduckgo/duckduckgo",
                        "tool_name": "ddg_search",
                        "credential_type": "unauthorized",
                    },
                ]
            },
        )
    )

    _validate(validator_db)


@pytest.mark.parametrize(
    ("node_job", "message"),
    [
        (WorkflowNodeJobConfig.model_validate({"metadata": {"agent_soul": {"tools": []}}}), "cannot override locked"),
        (WorkflowNodeJobConfig.model_validate({"human_contacts": [{"channel": "slack"}]}), "invalid human contact"),
        (
            WorkflowNodeJobConfig.model_validate(
                {"human_contacts": [{"contact_id": "human-1", "tenant_id": "other-tenant", "channel": "slack"}]}
            ),
            "out-of-scope human contact",
        ),
    ],
)
def test_publish_rejects_invalid_node_job_config(
    validator_db: ValidatorDatabase,
    node_job: WorkflowNodeJobConfig,
    message: str,
) -> None:
    validator_db.persist(node_job=node_job)

    with pytest.raises(WorkflowAgentNodeValidationError, match=message):
        _validate(validator_db)


def test_publish_resolves_tenant_scoped_upload_file(validator_db: ValidatorDatabase) -> None:
    upload_id = validator_db.add_upload()
    validator_db.persist(
        node_job=WorkflowNodeJobConfig.model_validate(
            {
                "declared_outputs": [
                    {
                        "name": "report",
                        "type": "file",
                        "check": {
                            "enabled": True,
                            "prompt": "Include risk summary",
                            "benchmark_file_ref": {"upload_file_id": upload_id},
                        },
                    }
                ]
            }
        )
    )

    _validate(validator_db)


@pytest.mark.parametrize("file_state", ["missing", "other_tenant"])
def test_publish_rejects_missing_or_out_of_scope_upload_file(validator_db: ValidatorDatabase, file_state: str) -> None:
    upload_id = str(uuid4()) if file_state == "missing" else validator_db.add_upload(tenant_id=str(uuid4()))
    validator_db.persist(
        node_job=WorkflowNodeJobConfig.model_validate({"metadata": {"file_refs": [{"upload_file_id": upload_id}]}})
    )

    with pytest.raises(WorkflowAgentNodeValidationError, match="missing or out-of-scope metadata file ref"):
        _validate(validator_db)


def test_publish_reports_missing_knowledge_dataset(
    validator_db: ValidatorDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_id = str(uuid4())
    validator_db.persist(
        soul=AgentSoulConfig(
            model=_model_config(),
            knowledge={
                "sets": [
                    {
                        "id": "support",
                        "name": "Support KB",
                        "datasets": [{"id": dataset_id}],
                        "query": {"mode": "generated_query"},
                        "retrieval": {"mode": "multiple", "top_k": 4},
                    }
                ]
            },
        )
    )
    captured: dict[str, object] = {}

    def no_datasets(ids: list[str], tenant_id: str) -> tuple[list[object], int]:
        captured.update(ids=ids, tenant_id=tenant_id)
        return [], 0

    monkeypatch.setattr("services.dataset_service.DatasetService.get_datasets_by_ids", no_datasets)

    with pytest.raises(WorkflowAgentNodeValidationError, match=dataset_id):
        _validate(validator_db)
    assert captured == {"ids": [dataset_id], "tenant_id": validator_db.tenant_id}


@pytest.mark.parametrize(
    "agentic_config",
    [{"state": "manual"}, {"state": "agentic", "parameter_draft": {"query": "x"}}],
)
def test_publish_accepts_complete_tool_agentic_modes(
    validator_db: ValidatorDatabase, agentic_config: dict[str, object]
) -> None:
    _validate(validator_db, _tool_graph({"agentic_mode": agentic_config}))


@pytest.mark.parametrize(
    ("agentic_config", "message"),
    [
        (True, "incomplete agentic mode config"),
        ({"state": "agentic", "complete": False}, "incomplete agentic mode config"),
        ({"state": "agentic", "permission": {"allowed": False}}, "unauthorized agentic mode config"),
    ],
)
def test_publish_rejects_invalid_tool_agentic_modes(
    validator_db: ValidatorDatabase,
    agentic_config: object,
    message: str,
) -> None:
    with pytest.raises(WorkflowAgentNodeValidationError, match=message):
        _validate(validator_db, _tool_graph({"agentic_mode": agentic_config}))


def test_duplicate_binding_constraint_rolls_back_without_losing_original(
    validator_db: ValidatorDatabase,
) -> None:
    validator_db.persist()
    duplicate = WorkflowAgentNodeBinding(
        tenant_id=validator_db.tenant_id,
        app_id=validator_db.app_id,
        workflow_id=validator_db.workflow_id,
        workflow_version="1",
        node_id="agent-node",
        binding_type=WorkflowAgentBindingType.INLINE_AGENT,
        agent_id=validator_db.agent_id,
        current_snapshot_id=validator_db.snapshot_id,
        node_job_config=WorkflowNodeJobConfig(),
    )
    validator_db.session.add(duplicate)

    with pytest.raises(IntegrityError):
        validator_db.session.commit()
    validator_db.session.rollback()

    _validate(validator_db)
    assert validator_db.session.get(WorkflowAgentNodeBinding, validator_db.binding_id) is not None

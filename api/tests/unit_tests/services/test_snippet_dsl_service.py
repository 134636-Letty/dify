"""SQLite-backed tests for :mod:`services.snippet_dsl_service`."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from sqlalchemy import Engine, event, select
from sqlalchemy.orm import Session

from graphon.nodes import BuiltinNodeTypes
from models.base import TypeBase
from models.snippet import CustomizedSnippet, SnippetType
from models.workflow import Workflow, WorkflowKind, WorkflowType
from services.snippet_dsl_service import (
    ImportMode,
    ImportStatus,
    SnippetDslService,
    SnippetPendingData,
    _check_version_compatibility,
)


@pytest.fixture
def orm_session(sqlite_engine: Engine) -> Iterator[Session]:
    TypeBase.metadata.create_all(sqlite_engine, tables=[CustomizedSnippet.__table__, Workflow.__table__])
    with Session(sqlite_engine, expire_on_commit=False) as session:
        yield session


@pytest.fixture
def service(orm_session: Session) -> SnippetDslService:
    return SnippetDslService(session=orm_session)


def _account(*, tenant_id: str = "tenant-1") -> SimpleNamespace:
    return SimpleNamespace(id="account-1", current_tenant_id=tenant_id)


def _snippet(session: Session, *, snippet_id: str = "snippet-1", tenant_id: str = "tenant-1") -> CustomizedSnippet:
    snippet = CustomizedSnippet(
        id=snippet_id,
        tenant_id=tenant_id,
        name="Snippet",
        description="description",
        type=SnippetType.NODE.value,
        created_by="account-1",
    )
    session.add(snippet)
    session.commit()
    return snippet


def _workflow(session: Session, snippet: CustomizedSnippet, *, graph: dict | None = None) -> Workflow:
    workflow = Workflow(
        id=f"workflow-{snippet.id}",
        tenant_id=snippet.tenant_id,
        app_id=snippet.id,
        type=WorkflowType.WORKFLOW,
        kind=WorkflowKind.SNIPPET,
        version=Workflow.VERSION_DRAFT,
        graph=json.dumps(graph or {"nodes": [], "edges": []}),
        features="{}",
        created_by="account-1",
        environment_variables=[],
        conversation_variables=[],
        rag_pipeline_variables=[],
    )
    session.add(workflow)
    session.commit()
    return workflow


@contextmanager
def _raise_on_workflow_insert(engine: Engine) -> Iterator[None]:
    def raise_error(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith("INSERT") and "workflows" in statement:
            raise RuntimeError("forced workflow INSERT")

    event.listen(engine, "before_cursor_execute", raise_error)
    try:
        yield
    finally:
        event.remove(engine, "before_cursor_execute", raise_error)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("not-a-version", ImportStatus.FAILED),
        ("999.0.0", ImportStatus.PENDING),
        ("0.1.0", ImportStatus.COMPLETED),
        ("0.0.9", ImportStatus.COMPLETED_WITH_WARNINGS),
    ],
)
def test_check_version_compatibility_special_cases(version: str, expected: ImportStatus) -> None:
    assert _check_version_compatibility(version) == expected


@pytest.mark.parametrize(
    ("kwargs", "expected_error"),
    [
        ({"import_mode": ImportMode.YAML_CONTENT.value}, "yaml_content is required"),
        ({"import_mode": ImportMode.YAML_URL.value}, "yaml_url is required"),
        (
            {"import_mode": ImportMode.YAML_URL.value, "yaml_url": "file:///tmp/snippet.yaml"},
            "Invalid URL scheme",
        ),
        (
            {"import_mode": ImportMode.YAML_CONTENT.value, "yaml_content": "- item"},
            "Invalid YAML format: expected a dictionary",
        ),
        (
            {
                "import_mode": ImportMode.YAML_CONTENT.value,
                "yaml_content": "version: 0.1.0\nsnippet:\n  name: Missing Kind\n",
            },
            "Missing 'kind' field",
        ),
        (
            {
                "import_mode": ImportMode.YAML_CONTENT.value,
                "yaml_content": "version: 0.1.0\nkind: app\nsnippet:\n  name: Wrong Kind\n",
            },
            "Invalid DSL kind",
        ),
        (
            {"import_mode": ImportMode.YAML_CONTENT.value, "yaml_content": "version: 0.1.0\nkind: snippet\n"},
            "Missing snippet data",
        ),
        (
            {
                "import_mode": ImportMode.YAML_CONTENT.value,
                "yaml_content": "version: 1\nkind: snippet\nsnippet:\n  name: Bad Version\n",
            },
            "Invalid version type",
        ),
    ],
)
def test_import_validation_errors(service: SnippetDslService, kwargs: dict[str, str], expected_error: str) -> None:
    result = service.import_snippet(account=_account(), **kwargs)
    assert result.status == ImportStatus.FAILED
    assert expected_error in result.error


def test_import_rejects_invalid_mode(service: SnippetDslService) -> None:
    with pytest.raises(ValueError, match="Invalid import_mode"):
        service.import_snippet(account=_account(), import_mode="bad-mode")


def test_import_url_boundary_failures(monkeypatch: pytest.MonkeyPatch, service: SnippetDslService) -> None:
    monkeypatch.setattr(
        "services.snippet_dsl_service.ssrf_proxy.get",
        Mock(return_value=SimpleNamespace(status_code=404, content=b"not found")),
    )
    failed = service.import_snippet(
        account=_account(), import_mode=ImportMode.YAML_URL.value, yaml_url="https://example.com/snippet.yaml"
    )
    assert failed.error == "Failed to fetch YAML from URL: 404"

    monkeypatch.setattr("services.snippet_dsl_service.DSL_MAX_SIZE", 1)
    monkeypatch.setattr(
        "services.snippet_dsl_service.ssrf_proxy.get",
        Mock(return_value=SimpleNamespace(status_code=200, content=b"large")),
    )
    oversized = service.import_snippet(
        account=_account(), import_mode=ImportMode.YAML_URL.value, yaml_url="https://example.com/snippet.yaml"
    )
    assert "size exceeds" in oversized.error


def test_import_rejects_forbidden_nodes(service: SnippetDslService) -> None:
    result = service.import_snippet(
        account=_account(),
        import_mode=ImportMode.YAML_CONTENT.value,
        yaml_content="""
version: 0.1.0
kind: snippet
snippet:
  name: Bad
workflow:
  graph:
    nodes:
      - id: start-1
        data: {type: start}
    edges: []
""",
    )
    assert result.status == ImportStatus.FAILED
    assert result.error == "Snippet cannot contain the following node types: start"


def test_import_newer_dsl_stores_pending_data(monkeypatch: pytest.MonkeyPatch, service: SnippetDslService) -> None:
    setex = Mock()
    monkeypatch.setattr("services.snippet_dsl_service.redis_client.setex", setex)
    result = service.import_snippet(
        account=_account(),
        import_mode=ImportMode.YAML_CONTENT.value,
        yaml_content="""
version: 999.0.0
kind: snippet
snippet: {name: Future}
workflow: {graph: {nodes: [], edges: []}}
""",
        name="Override",
        description="Override description",
    )
    assert result.status == ImportStatus.PENDING
    pending = SnippetPendingData.model_validate_json(setex.call_args.args[2])
    assert pending.name == "Override"
    assert pending.description == "Override description"


def test_import_update_target_is_tenant_scoped(service: SnippetDslService, orm_session: Session) -> None:
    foreign = _snippet(orm_session, tenant_id="tenant-2")
    yaml_content = """
version: 0.1.0
kind: snippet
snippet: {name: Existing}
workflow: {graph: {nodes: [], edges: []}}
"""
    missing = service.import_snippet(
        account=_account(tenant_id="tenant-1"),
        import_mode=ImportMode.YAML_CONTENT.value,
        yaml_content=yaml_content,
        snippet_id=foreign.id,
    )
    assert missing.status == ImportStatus.FAILED
    assert missing.error == "Snippet not found"


def test_import_creates_persisted_snippet_workflow_and_passes_dependencies(
    monkeypatch: pytest.MonkeyPatch, service: SnippetDslService, orm_session: Session
) -> None:
    dependency_check = Mock(return_value=[])
    monkeypatch.setattr(
        "services.snippet_dsl_service.DependenciesAnalysisService.generate_dependencies", dependency_check
    )
    result = service.import_snippet(
        account=_account(),
        import_mode=ImportMode.YAML_CONTENT.value,
        yaml_content="""
version: 0.1.0
kind: snippet
snippet:
  name: Imported
  type: group
  input_fields: [{variable: query}]
dependencies:
  - type: marketplace
    value:
      marketplace_plugin_unique_identifier: langgenius/openai:0.0.1
workflow:
  graph: {nodes: [], edges: []}
""",
    )
    assert result.status == ImportStatus.COMPLETED
    persisted = orm_session.get(CustomizedSnippet, result.snippet_id)
    assert persisted is not None
    assert persisted.name == "Imported"
    assert persisted.type == SnippetType.GROUP
    workflow = orm_session.scalar(
        select(Workflow).where(Workflow.app_id == persisted.id, Workflow.version == Workflow.VERSION_DRAFT)
    )
    assert workflow is not None


def test_confirm_import_handles_missing_invalid_and_creates_from_pending(
    monkeypatch: pytest.MonkeyPatch, service: SnippetDslService, orm_session: Session
) -> None:
    get = Mock(return_value=None)
    monkeypatch.setattr("services.snippet_dsl_service.redis_client.get", get)
    assert service.confirm_import(import_id="missing", account=_account()).status == ImportStatus.FAILED
    get.return_value = object()
    assert service.confirm_import(import_id="invalid", account=_account()).error == "Invalid import information"

    pending = SnippetPendingData(
        import_mode=ImportMode.YAML_CONTENT.value,
        yaml_content="""
version: 9.0.0
kind: snippet
snippet: {name: From DSL, type: node}
workflow: {graph: {nodes: [], edges: []}}
""",
        name="Override",
        snippet_id=None,
    )
    get.return_value = pending.model_dump_json()
    delete = Mock()
    monkeypatch.setattr("services.snippet_dsl_service.redis_client.delete", delete)
    result = service.confirm_import(import_id="import-1", account=_account())
    assert result.status == ImportStatus.COMPLETED
    assert orm_session.get(CustomizedSnippet, result.snippet_id).name == "Override"
    delete.assert_called_once_with("snippet_import_info:import-1")


def test_check_dependencies_reads_real_draft_workflow(
    monkeypatch: pytest.MonkeyPatch, service: SnippetDslService, orm_session: Session
) -> None:
    snippet = _snippet(orm_session)
    assert service.check_dependencies(snippet).leaked_dependencies == []
    _workflow(
        orm_session,
        snippet,
        graph={
            "nodes": [
                {
                    "data": {
                        "type": BuiltinNodeTypes.TOOL,
                        "tool_configurations": {"provider_type": "builtin", "provider": "langgenius/openai"},
                    }
                }
            ],
            "edges": [],
        },
    )
    leaked = [
        {
            "type": "marketplace",
            "value": {"marketplace_plugin_unique_identifier": "langgenius/openai:0.0.1"},
        }
    ]
    monkeypatch.setattr(
        "services.snippet_dsl_service.DependenciesAnalysisService.generate_dependencies", Mock(return_value=leaked)
    )
    result = service.check_dependencies(snippet)
    assert result.leaked_dependencies[0].value.plugin_unique_identifier == "langgenius/openai:0.0.1"


def test_create_or_update_persists_existing_and_new_rows(service: SnippetDslService, orm_session: Session) -> None:
    existing = _snippet(orm_session)
    _workflow(orm_session, existing)
    result = service._create_or_update_snippet(
        snippet=existing,
        data={
            "snippet": {
                "name": "Updated",
                "description": "new",
                "type": "unknown",
                "icon_info": {"icon": "x"},
                "input_fields": [{"variable": "query"}],
            },
            "workflow": {"graph": {"nodes": [], "edges": []}},
        },
        account=_account(),
    )
    assert result.id == existing.id
    assert result.name == "Updated"
    assert result.type == SnippetType.NODE
    assert result.input_fields_list == [{"variable": "query"}]

    created = service._create_or_update_snippet(
        snippet=None,
        data={"snippet": {"name": "New", "type": "group"}, "workflow": {"graph": {"nodes": [], "edges": []}}},
        account=_account(),
    )
    assert orm_session.get(CustomizedSnippet, created.id) is not None
    assert created.type == SnippetType.GROUP
    assert orm_session.scalar(select(Workflow).where(Workflow.app_id == created.id)) is not None


def test_create_rolls_back_when_workflow_insert_fails(
    service: SnippetDslService, orm_session: Session, sqlite_engine: Engine
) -> None:
    with _raise_on_workflow_insert(sqlite_engine), pytest.raises(RuntimeError, match="forced workflow INSERT"):
        service._create_or_update_snippet(
            snippet=None,
            data={"snippet": {"name": "Broken"}, "workflow": {"graph": {"nodes": [], "edges": []}}},
            account=_account(),
        )
    orm_session.rollback()
    assert orm_session.scalar(select(CustomizedSnippet)) is None
    assert orm_session.scalar(select(Workflow)) is None


def test_export_snippet_dsl_reads_real_draft_and_filters_credentials(
    monkeypatch: pytest.MonkeyPatch, service: SnippetDslService, orm_session: Session
) -> None:
    snippet = _snippet(orm_session)
    with pytest.raises(ValueError, match="Missing draft workflow"):
        service.export_snippet_dsl(snippet)
    workflow = _workflow(
        orm_session,
        snippet,
        graph={
            "nodes": [
                {
                    "data": {
                        "type": BuiltinNodeTypes.TOOL,
                        "credential_id": "secret",
                        "tool_configurations": {"provider_type": "builtin", "provider": "langgenius/google"},
                    }
                },
                {
                    "data": {
                        "type": BuiltinNodeTypes.KNOWLEDGE_RETRIEVAL,
                        "dataset_ids": ["dataset-1"],
                    }
                },
            ],
            "edges": [],
        },
    )
    monkeypatch.setattr(
        "services.snippet_dsl_service.DependenciesAnalysisService.generate_dependencies", Mock(return_value=[])
    )
    exported = yaml.safe_load(service.export_snippet_dsl(snippet))
    assert exported["kind"] == "snippet"
    assert exported["snippet"]["name"] == "Snippet"
    nodes = exported["workflow"]["graph"]["nodes"]
    assert "credential_id" not in nodes[0]["data"]
    assert nodes[1]["data"]["dataset_ids"] == ["dataset-1"]
    assert workflow.id.startswith("workflow-")

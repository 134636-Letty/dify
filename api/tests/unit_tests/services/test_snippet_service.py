"""State-based tests for :mod:`services.snippet_service`."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import Engine, event, select
from sqlalchemy.orm import Session, sessionmaker

from models import TagBinding
from models.agent import Agent, WorkflowAgentNodeBinding
from models.base import TypeBase
from models.model import UploadFile
from models.snippet import CustomizedSnippet, SnippetType
from models.tools import WorkflowToolProvider
from models.workflow import (
    Workflow,
    WorkflowAppLog,
    WorkflowArchiveLog,
    WorkflowDraftVariable,
    WorkflowDraftVariableFile,
    WorkflowKind,
    WorkflowNodeExecutionModel,
    WorkflowRun,
    WorkflowType,
)
from services.errors.app import IsDraftWorkflowError, WorkflowHashNotEqualError
from services.snippet_service import SnippetService


@dataclass(frozen=True)
class Database:
    session: Session
    maker: sessionmaker[Session]


@pytest.fixture
def database(sqlite_engine: Engine) -> Iterator[Database]:
    """Create the real tables used by snippet and workflow lifecycle operations."""

    models = (
        CustomizedSnippet,
        Workflow,
        WorkflowDraftVariable,
        WorkflowDraftVariableFile,
        UploadFile,
        WorkflowToolProvider,
        WorkflowAppLog,
        WorkflowArchiveLog,
        WorkflowNodeExecutionModel,
        WorkflowRun,
        TagBinding,
        Agent,
        WorkflowAgentNodeBinding,
    )
    TypeBase.metadata.create_all(sqlite_engine, tables=[model.__table__ for model in models])
    maker = sessionmaker(bind=sqlite_engine, expire_on_commit=False)
    with maker() as session:
        yield Database(session=session, maker=maker)


def _service(database: Database, *, caller_owned: bool = False) -> SnippetService:
    service = SnippetService.__new__(SnippetService)
    service._session = database.session if caller_owned else None
    service._session_maker = database.maker
    service._workflow_run_repo = Mock()
    service._node_execution_service_repo = Mock()
    return service


def _snippet(
    session: Session,
    *,
    snippet_id: str = "snippet-1",
    tenant_id: str = "tenant-1",
    name: str = "Snippet",
    published: bool = False,
) -> CustomizedSnippet:
    snippet = CustomizedSnippet(
        id=snippet_id,
        tenant_id=tenant_id,
        name=name,
        description="description",
        type=SnippetType.NODE.value,
        created_by="account-1",
        is_published=published,
    )
    session.add(snippet)
    session.commit()
    return snippet


def _workflow(
    session: Session,
    snippet: CustomizedSnippet,
    *,
    workflow_id: str,
    version: str,
    graph: dict | None = None,
) -> Workflow:
    workflow = Workflow(
        id=workflow_id,
        tenant_id=snippet.tenant_id,
        app_id=snippet.id,
        type=WorkflowType.WORKFLOW,
        kind=WorkflowKind.SNIPPET,
        version=version,
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
def _raise_on_insert(engine: Engine, table_name: str) -> Iterator[None]:
    def raise_error(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith("INSERT") and table_name in statement:
            raise RuntimeError("forced INSERT")

    event.listen(engine, "before_cursor_execute", raise_error)
    try:
        yield
    finally:
        event.remove(engine, "before_cursor_execute", raise_error)


def test_create_snippet_allows_duplicate_names_and_commits_owned_session(database: Database) -> None:
    _snippet(database.session, snippet_id="existing", name="shared name")
    service = _service(database)
    created = service.create_snippet(
        tenant_id="tenant-1",
        name="shared name",
        description=None,
        snippet_type=SnippetType.NODE,
        icon_info=None,
        input_fields=[{"variable": "query"}],
        account=SimpleNamespace(id="account-1"),
    )
    database.session.expire_all()
    persisted = database.session.get(CustomizedSnippet, created.id)
    assert persisted is not None
    assert persisted.name == "shared name"
    assert persisted.input_fields_list == [{"variable": "query"}]
    assert database.session.scalar(select(CustomizedSnippet).where(CustomizedSnippet.name == "shared name"))


def test_create_snippet_caller_owned_session_does_not_commit(database: Database) -> None:
    service = _service(database, caller_owned=True)
    created = service.create_snippet(
        tenant_id="tenant-1",
        name="Pending",
        description=None,
        snippet_type=SnippetType.NODE,
        icon_info=None,
        input_fields=None,
        account=SimpleNamespace(id="account-1"),
    )
    assert created in database.session.new
    assert database.session.in_transaction()


def test_create_snippet_rolls_back_owned_session_on_constraint_hook(database: Database, sqlite_engine: Engine) -> None:
    service = _service(database)
    with _raise_on_insert(sqlite_engine, "customized_snippets"), pytest.raises(RuntimeError, match="forced INSERT"):
        service.create_snippet(
            tenant_id="tenant-1",
            name="Broken",
            description=None,
            snippet_type=SnippetType.NODE,
            icon_info=None,
            input_fields=None,
            account=SimpleNamespace(id="account-1"),
        )
    assert database.session.scalar(select(CustomizedSnippet)) is None


def test_validate_snippet_graph_forbidden_nodes_handles_malformed_and_rejects_start() -> None:
    SnippetService.validate_snippet_graph_forbidden_nodes(
        {"nodes": ["bad", {"id": "empty", "data": {}}, {"id": "llm", "data": {"type": "llm"}}]}
    )
    with pytest.raises(ValueError, match="start-1:start"):
        SnippetService.validate_snippet_graph_forbidden_nodes({"nodes": [{"id": "start-1", "data": {"type": "start"}}]})


def test_get_snippets_filters_paginates_and_is_tenant_scoped(
    monkeypatch: pytest.MonkeyPatch, database: Database
) -> None:
    _snippet(database.session, snippet_id="one", name="Search One", published=True)
    _snippet(database.session, snippet_id="two", name="Search Two", published=True)
    _snippet(database.session, snippet_id="three", name="Search Three", published=True)
    _snippet(database.session, snippet_id="foreign", tenant_id="tenant-2", name="Search Foreign", published=True)
    monkeypatch.setattr(
        "services.snippet_service.TagService.get_target_ids_by_tag_ids",
        Mock(return_value=["one", "two", "three", "foreign"]),
    )
    result, total, has_more = _service(database).get_snippets(
        tenant_id="tenant-1",
        session=database.session,
        page=1,
        limit=2,
        keyword="Search",
        is_published=True,
        creators=["account-1"],
        tag_ids=["tag-1"],
    )
    assert len(result) == 2
    assert {snippet.id for snippet in result}.issubset({"one", "two", "three"})
    assert all(snippet.tenant_id == "tenant-1" for snippet in result)
    assert total == 3
    assert has_more is True


def test_get_snippets_returns_empty_when_tag_filter_has_no_targets(
    monkeypatch: pytest.MonkeyPatch, database: Database
) -> None:
    monkeypatch.setattr("services.snippet_service.TagService.get_target_ids_by_tag_ids", Mock(return_value=[]))
    assert _service(database).get_snippets(tenant_id="tenant-1", session=database.session, tag_ids=["missing"]) == (
        [],
        0,
        False,
    )


def test_get_snippet_by_id_enforces_tenant(database: Database) -> None:
    snippet = _snippet(database.session)
    service = _service(database, caller_owned=True)
    assert service.get_snippet_by_id(snippet_id=snippet.id, tenant_id="tenant-1").id == snippet.id
    assert service.get_snippet_by_id(snippet_id=snippet.id, tenant_id="tenant-2") is None


def test_update_snippet_persists_optional_fields_and_duplicate_name(database: Database) -> None:
    first = _snippet(database.session, snippet_id="first", name="shared")
    second = _snippet(database.session, snippet_id="second", name="other")
    SnippetService.update_snippet(
        session=database.session,
        snippet=second,
        account_id="account-2",
        data={"name": first.name, "description": "new", "icon_info": {"icon": "star"}},
    )
    database.session.commit()
    database.session.expire_all()
    persisted = database.session.get(CustomizedSnippet, second.id)
    assert persisted is not None
    assert persisted.name == "shared"
    assert persisted.description == "new"
    assert persisted.icon_info == {"icon": "star"}
    assert persisted.updated_by == "account-2"


def test_sync_draft_workflow_creates_and_updates_real_draft(database: Database) -> None:
    snippet = _snippet(database.session)
    service = _service(database, caller_owned=True)
    account = SimpleNamespace(id="account-1")
    created = service.sync_draft_workflow(
        snippet=snippet,
        graph={"nodes": [{"id": "llm-1", "data": {"type": "llm"}}], "edges": []},
        unique_hash=None,
        account=account,
        input_fields=[{"variable": "query"}],
    )
    database.session.commit()
    assert created.version == Workflow.VERSION_DRAFT
    assert created.kind == WorkflowKind.SNIPPET
    database.session.expire_all()
    assert database.session.get(CustomizedSnippet, snippet.id).input_fields_list == [{"variable": "query"}]

    original_hash = created.unique_hash
    updated = service.sync_draft_workflow(
        snippet=snippet,
        graph={"nodes": [{"id": "llm-2", "data": {"type": "llm"}}], "edges": []},
        unique_hash=original_hash,
        account=account,
    )
    assert updated.id == created.id
    assert updated.graph_dict["nodes"][0]["id"] == "llm-2"
    assert updated.environment_variables == []


def test_sync_draft_workflow_rejects_stale_hash(database: Database) -> None:
    snippet = _snippet(database.session)
    _workflow(database.session, snippet, workflow_id="draft", version=Workflow.VERSION_DRAFT)
    with pytest.raises(WorkflowHashNotEqualError):
        _service(database).sync_draft_workflow(
            snippet=snippet,
            graph={"nodes": [], "edges": []},
            unique_hash="stale",
            account=SimpleNamespace(id="account-1"),
        )


def test_publish_update_and_paginate_workflows(database: Database) -> None:
    snippet = _snippet(database.session)
    _workflow(database.session, snippet, workflow_id="draft", version=Workflow.VERSION_DRAFT)
    service = _service(database)
    published = service.publish_workflow(
        session=database.session, snippet=snippet, account=SimpleNamespace(id="account-1")
    )
    database.session.commit()
    assert snippet.is_published is True
    assert snippet.workflow_id == published.id
    assert snippet.version == 2

    updated = service.update_workflow(
        session=database.session,
        snippet=snippet,
        workflow_id=published.id,
        account=SimpleNamespace(id="account-2"),
        data={"marked_name": "v2", "marked_comment": "published", "ignored": "x"},
    )
    assert updated is not None
    assert updated.marked_name == "v2"
    assert updated.marked_comment == "published"
    assert updated.updated_by == "account-2"
    workflows, has_more = service.get_all_published_workflows(
        session=database.session, snippet=snippet, page=1, limit=1
    )
    assert [workflow.id for workflow in workflows] == [published.id]
    assert has_more is False


def test_published_workflow_lookup_rejects_draft_and_is_tenant_scoped(database: Database) -> None:
    snippet = _snippet(database.session)
    draft = _workflow(database.session, snippet, workflow_id="draft", version=Workflow.VERSION_DRAFT)
    service = _service(database)
    with pytest.raises(IsDraftWorkflowError):
        service.get_published_workflow_by_id(snippet=snippet, workflow_id=draft.id)
    foreign = CustomizedSnippet(
        id=snippet.id,
        tenant_id="tenant-2",
        name="foreign",
        description="",
        type=SnippetType.NODE.value,
    )
    assert service.get_published_workflow_by_id(snippet=foreign, workflow_id=draft.id) is None


def test_restore_published_workflow_copies_snapshot_to_draft(database: Database) -> None:
    snippet = _snippet(database.session)
    draft = _workflow(database.session, snippet, workflow_id="draft", version=Workflow.VERSION_DRAFT)
    published = _workflow(
        database.session,
        snippet,
        workflow_id="published",
        version="2",
        graph={"nodes": [{"id": "llm-1", "data": {"type": "llm"}}], "edges": []},
    )
    restored = _service(database).restore_published_workflow_to_draft(
        snippet=snippet, workflow_id=published.id, account=SimpleNamespace(id="account-2")
    )
    database.session.expire_all()
    persisted = database.session.get(Workflow, draft.id)
    assert restored.id == draft.id
    assert persisted.graph_dict == published.graph_dict
    assert persisted.updated_by == "account-2"


def test_default_block_configs_keep_runtime_boundary_mocked(
    monkeypatch: pytest.MonkeyPatch, database: Database
) -> None:
    with_default = SimpleNamespace(get_default_config=Mock(return_value={"type": "llm"}))
    without_default = SimpleNamespace(get_default_config=Mock(return_value=None))
    monkeypatch.setattr(
        "services.snippet_service.NODE_TYPE_CLASSES_MAPPING",
        {"llm": {"1": with_default}, "empty": {"1": without_default}},
    )
    monkeypatch.setattr("services.snippet_service.LATEST_VERSION", "1")
    service = _service(database)
    assert service.get_default_block_configs() == [{"type": "llm"}]
    assert service.get_default_block_config("llm", filters={"k": "v"}) == {"type": "llm"}
    assert service.get_default_block_config("missing") is None


def test_delete_snippet_removes_workflow_and_tag_rows(database: Database) -> None:
    snippet = _snippet(database.session)
    workflow = _workflow(database.session, snippet, workflow_id="draft", version=Workflow.VERSION_DRAFT)
    binding = TagBinding(tenant_id=snippet.tenant_id, tag_id="tag-1", target_id=snippet.id, created_by="account-1")
    database.session.add(binding)
    database.session.commit()
    snippet_id = snippet.id
    workflow_id = workflow.id
    binding_id = binding.id
    assert SnippetService.delete_snippet(session=database.session, snippet=snippet) is True
    database.session.commit()
    database.session.expire_all()
    assert database.session.get(CustomizedSnippet, snippet_id) is None
    assert database.session.get(Workflow, workflow_id) is None
    assert database.session.get(TagBinding, binding_id) is None


def test_delete_snippet_archives_owned_agents_and_schedules_backing_app_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snippet = SimpleNamespace(id="snippet-1", tenant_id="tenant-1")
    agent = SimpleNamespace(
        backing_app_id="backing-app-1",
        status="active",
        archived_by=None,
        archived_at=None,
        updated_by="creator-1",
        updated_at=None,
    )
    scalar_results = [
        SimpleNamespace(all=Mock(return_value=[])),
        SimpleNamespace(all=Mock(return_value=[agent])),
    ]
    session = SimpleNamespace(
        execute=Mock(),
        scalars=Mock(side_effect=scalar_results),
        delete=Mock(),
    )
    listen = Mock()
    monkeypatch.setattr("services.snippet_service.event.listen", listen)

    result = SnippetService.delete_snippet(
        session=session,
        snippet=snippet,
        account_id="account-1",
    )

    assert result is True
    assert agent.status == "archived"
    assert agent.archived_by == "account-1"
    assert agent.archived_at is not None
    assert agent.updated_by == "account-1"
    executed_sql = "\n".join(str(call.args[0]) for call in session.execute.call_args_list)
    assert "DELETE FROM apps" in executed_sql
    listen.assert_called_once_with(session, "after_commit", listen.call_args.args[2], once=True)


def test_delete_archived_workflow_run_files_uses_storage_boundary(
    monkeypatch: pytest.MonkeyPatch, database: Database
) -> None:
    from configs import dify_config

    snippet = _snippet(database.session)
    archive_storage = SimpleNamespace(
        list_objects=Mock(return_value=["tenant-1/app_id=snippet-1/run.json"]), delete_object=Mock()
    )
    monkeypatch.setattr(dify_config, "BILLING_ENABLED", True)
    monkeypatch.setattr(dify_config, "ARCHIVE_STORAGE_ENABLED", True)
    monkeypatch.setattr("libs.archive_storage.get_archive_storage", Mock(return_value=archive_storage))
    SnippetService._delete_archived_workflow_run_files(snippet=snippet)
    archive_storage.list_objects.assert_called_once_with("tenant-1/app_id=snippet-1/")
    archive_storage.delete_object.assert_called_once()


def test_workflow_run_queries_delegate_to_repositories(database: Database) -> None:
    service = _service(database)
    service._workflow_run_repo.get_paginated_workflow_runs.return_value = SimpleNamespace(data=[])
    service._workflow_run_repo.get_workflow_run_by_id.return_value = SimpleNamespace(id="run-1")
    service._node_execution_service_repo.get_executions_by_workflow_run.return_value = [
        SimpleNamespace(id="execution-1")
    ]
    service._node_execution_service_repo.get_node_last_execution.return_value = SimpleNamespace(id="last-1")
    snippet = _snippet(database.session)
    workflow = _workflow(database.session, snippet, workflow_id="draft", version=Workflow.VERSION_DRAFT)
    assert service.get_snippet_workflow_runs(snippet=snippet, args={"limit": "5"}).data == []
    assert service.get_snippet_workflow_run_node_executions(snippet=snippet, run_id="run-1")[0].id == "execution-1"
    assert service.get_snippet_node_last_run(snippet=snippet, workflow=workflow, node_id="llm").id == "last-1"


def test_increment_use_count_persists_real_snippet(database: Database) -> None:
    snippet = _snippet(database.session)
    SnippetService.increment_use_count(session=database.session, snippet=snippet)
    database.session.commit()
    database.session.expire_all()
    assert database.session.get(CustomizedSnippet, snippet.id).use_count == 1

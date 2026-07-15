"""Unit tests for PipelineRunner behavior.

This module validates core control-flow outcomes for
``core.app.apps.pipeline.pipeline_runner``: app/workflow lookup, graph
initialization guards, invoke-source to user-source resolution, and failed-run
event handling. Pipeline, Workflow, EndUser, and Document state is persisted in
SQLite; graph execution, queues, and workflow repositories remain mocked.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from sqlalchemy.orm import Session, sessionmaker

import core.app.apps.pipeline.pipeline_runner as module
from core.app.apps.pipeline.pipeline_runner import PipelineRunner
from core.app.entities.app_invoke_entities import InvokeFrom, UserFrom
from core.rag.index_processor.constant.index_type import IndexStructureType
from graphon.graph_events import GraphRunFailedEvent
from models.dataset import Document, Pipeline
from models.enums import DataSourceType, DocumentCreatedFrom, EndUserType, IndexingStatus
from models.model import EndUser
from models.workflow import Workflow, WorkflowType

TENANT_ID = "00000000-0000-0000-0000-000000000001"
PIPELINE_ID = "00000000-0000-0000-0000-000000000002"
WORKFLOW_ID = "00000000-0000-0000-0000-000000000003"
USER_ID = "00000000-0000-0000-0000-000000000004"
DOCUMENT_ID = "00000000-0000-0000-0000-000000000005"
DATASET_ID = "00000000-0000-0000-0000-000000000006"
CREATOR_ID = "00000000-0000-0000-0000-000000000007"

pytestmark = pytest.mark.parametrize(
    "sqlite_session",
    [(Pipeline, Workflow, EndUser, Document)],
    indirect=True,
)


def _build_app_generate_entity() -> SimpleNamespace:
    app_config = SimpleNamespace(app_id=PIPELINE_ID, workflow_id=WORKFLOW_ID, tenant_id=TENANT_ID)
    return SimpleNamespace(
        app_config=app_config,
        invoke_from=InvokeFrom.WEB_APP,
        user_id=USER_ID,
        trace_manager=MagicMock(),
        inputs={"input1": "v1"},
        files=[],
        workflow_execution_id="run",
        document_id=DOCUMENT_ID,
        original_document_id=None,
        batch="batch",
        dataset_id=DATASET_ID,
        datasource_type="local_file",
        datasource_info={"name": "file"},
        start_node_id="start",
        call_depth=0,
        single_iteration_run=None,
        single_loop_run=None,
    )


@pytest.fixture(autouse=True)
def _bind_sqlite_sessions(monkeypatch: pytest.MonkeyPatch, sqlite_session: Session) -> None:
    """Bind runner-owned short-lived sessions to the isolated SQLite engine."""
    monkeypatch.setattr(
        module,
        "create_session",
        sessionmaker(bind=sqlite_session.get_bind(), expire_on_commit=False),
    )


def _persist_pipeline(sqlite_session: Session) -> Pipeline:
    pipeline = Pipeline(tenant_id=TENANT_ID, name="Pipeline", description="Test pipeline")
    pipeline.id = PIPELINE_ID
    sqlite_session.add(pipeline)
    sqlite_session.commit()
    return pipeline


def _persist_workflow(
    sqlite_session: Session,
    *,
    graph: dict[str, object] | None = None,
    rag_pipeline_variables: list[dict[str, str]] | None = None,
) -> Workflow:
    workflow = Workflow.new(
        tenant_id=TENANT_ID,
        app_id=PIPELINE_ID,
        type=WorkflowType.RAG_PIPELINE.value,
        version="v1",
        graph=json.dumps(graph if graph is not None else {"nodes": [], "edges": []}),
        features="{}",
        created_by=CREATOR_ID,
        environment_variables=[],
        conversation_variables=[],
        rag_pipeline_variables=rag_pipeline_variables or [],
    )
    workflow.id = WORKFLOW_ID
    sqlite_session.add(workflow)
    sqlite_session.commit()
    return workflow


def _persist_end_user(sqlite_session: Session) -> EndUser:
    end_user = EndUser(
        id=USER_ID,
        tenant_id=TENANT_ID,
        app_id=PIPELINE_ID,
        type=EndUserType.BROWSER,
        name="User",
        session_id="sess",
    )
    sqlite_session.add(end_user)
    sqlite_session.commit()
    return end_user


def _persist_document(sqlite_session: Session) -> Document:
    document = Document(
        id=DOCUMENT_ID,
        tenant_id=TENANT_ID,
        dataset_id=DATASET_ID,
        position=1,
        data_source_type=DataSourceType.LOCAL_FILE,
        batch="batch",
        name="Document",
        created_from=DocumentCreatedFrom.API,
        created_by=CREATOR_ID,
        indexing_status=IndexingStatus.COMPLETED,
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
    )
    sqlite_session.add(document)
    sqlite_session.commit()
    return document


@pytest.fixture
def runner():
    app_generate_entity = _build_app_generate_entity()
    queue_manager = MagicMock()
    variable_loader = MagicMock()
    workflow = MagicMock()
    workflow_execution_repository = MagicMock()
    workflow_node_execution_repository = MagicMock()

    return PipelineRunner(
        application_generate_entity=app_generate_entity,
        queue_manager=queue_manager,
        variable_loader=variable_loader,
        workflow=workflow,
        system_user_id="sys",
        workflow_execution_repository=workflow_execution_repository,
        workflow_node_execution_repository=workflow_node_execution_repository,
    )


def test_get_app_id(runner):
    assert runner._get_app_id() == PIPELINE_ID


def test_get_workflow_returns_workflow(runner, sqlite_session: Session):
    pipeline = _persist_pipeline(sqlite_session)
    workflow = _persist_workflow(sqlite_session)

    result = runner.get_workflow(session=sqlite_session, pipeline=pipeline, workflow_id=WORKFLOW_ID)

    assert result is workflow


def test_init_rag_pipeline_graph_invalid_config(mocker, runner):
    workflow = MagicMock(id="wf", tenant_id="tenant", graph_dict={})

    with pytest.raises(ValueError):
        runner._init_rag_pipeline_graph(workflow=workflow, graph_runtime_state=MagicMock())

    workflow.graph_dict = {"nodes": "bad", "edges": []}
    with pytest.raises(ValueError):
        runner._init_rag_pipeline_graph(workflow=workflow, graph_runtime_state=MagicMock())

    workflow.graph_dict = {"nodes": [], "edges": "bad"}
    with pytest.raises(ValueError):
        runner._init_rag_pipeline_graph(workflow=workflow, graph_runtime_state=MagicMock())


def test_init_rag_pipeline_graph_not_found(mocker, runner):
    workflow = MagicMock(id="wf", tenant_id="tenant", graph_dict={"nodes": [], "edges": []})
    mocker.patch.object(module.Graph, "init", return_value=None)

    with pytest.raises(ValueError):
        runner._init_rag_pipeline_graph(workflow=workflow, graph_runtime_state=MagicMock())


def test_update_document_status_on_failure(runner, sqlite_session: Session):
    document = _persist_document(sqlite_session)
    event = GraphRunFailedEvent(error="boom")

    runner._update_document_status(event, document_id=DOCUMENT_ID, dataset_id=DATASET_ID)

    sqlite_session.expire_all()
    stored_document = sqlite_session.get(Document, document.id)
    assert stored_document is not None
    assert stored_document.indexing_status == IndexingStatus.ERROR
    assert stored_document.error == "boom"


def test_run_pipeline_not_found():
    app_generate_entity = _build_app_generate_entity()
    app_generate_entity.invoke_from = InvokeFrom.WEB_APP
    app_generate_entity.single_iteration_run = None
    app_generate_entity.single_loop_run = None

    runner = PipelineRunner(
        application_generate_entity=app_generate_entity,
        queue_manager=MagicMock(),
        variable_loader=MagicMock(),
        workflow=MagicMock(),
        system_user_id="sys",
        workflow_execution_repository=MagicMock(),
        workflow_node_execution_repository=MagicMock(),
    )

    with pytest.raises(ValueError):
        runner.run()


def test_run_workflow_not_initialized(sqlite_session: Session):
    app_generate_entity = _build_app_generate_entity()
    _persist_pipeline(sqlite_session)

    runner = PipelineRunner(
        application_generate_entity=app_generate_entity,
        queue_manager=MagicMock(),
        variable_loader=MagicMock(),
        workflow=MagicMock(),
        system_user_id="sys",
        workflow_execution_repository=MagicMock(),
        workflow_node_execution_repository=MagicMock(),
    )
    with pytest.raises(ValueError):
        runner.run()


def test_run_single_iteration_path(mocker: MockerFixture, sqlite_session: Session):
    app_generate_entity = _build_app_generate_entity()
    app_generate_entity.single_iteration_run = MagicMock()
    _persist_pipeline(sqlite_session)
    _persist_workflow(sqlite_session)

    runner = PipelineRunner(
        application_generate_entity=app_generate_entity,
        queue_manager=MagicMock(),
        variable_loader=MagicMock(),
        workflow=MagicMock(),
        system_user_id="sys",
        workflow_execution_repository=MagicMock(),
        workflow_node_execution_repository=MagicMock(),
    )

    runner._resolve_user_from = MagicMock(return_value=UserFrom.ACCOUNT)
    runner._prepare_single_node_execution = MagicMock(return_value=("graph", "pool", "state"))
    runner._update_document_status = MagicMock()
    runner._handle_event = MagicMock()

    workflow_entry = MagicMock()
    workflow_entry.graph_engine = MagicMock()
    workflow_entry.run.return_value = [MagicMock()]
    mocker.patch.object(module, "WorkflowEntry", return_value=workflow_entry)

    mocker.patch.object(module, "WorkflowPersistenceLayer", return_value=MagicMock())

    runner.run()

    runner._prepare_single_node_execution.assert_called_once()
    runner._handle_event.assert_called()


def test_run_normal_path_builds_graph(mocker: MockerFixture, sqlite_session: Session):
    app_generate_entity = _build_app_generate_entity()
    events: list[str] = []
    _persist_end_user(sqlite_session)
    _persist_pipeline(sqlite_session)
    workflow = _persist_workflow(
        sqlite_session,
        rag_pipeline_variables=[
            {
                "variable": "input1",
                "belong_to_node_id": "start",
                "type": "text-input",
                "label": "Input 1",
            }
        ],
    )

    runner = PipelineRunner(
        application_generate_entity=app_generate_entity,
        queue_manager=MagicMock(),
        variable_loader=MagicMock(),
        workflow=workflow,
        system_user_id="sys",
        workflow_execution_repository=MagicMock(),
        workflow_node_execution_repository=MagicMock(),
    )

    runner._resolve_user_from = MagicMock(return_value=UserFrom.ACCOUNT)
    runner._init_rag_pipeline_graph = MagicMock(return_value="graph")
    runner._update_document_status = MagicMock()
    runner._handle_event = MagicMock()

    class FakeVariablePool:
        def add(self, selector, value):
            return None

    mocker.patch.object(module, "VariablePool", return_value=FakeVariablePool())

    workflow_entry = MagicMock()
    workflow_entry.graph_engine = MagicMock()
    workflow_entry.run.side_effect = lambda: events.append("workflow_run") or []
    mocker.patch.object(module, "WorkflowEntry", return_value=workflow_entry)
    mocker.patch.object(module, "WorkflowPersistenceLayer", return_value=MagicMock())

    runner.run()

    assert events == ["workflow_run"]
    runner._init_rag_pipeline_graph.assert_called_once()

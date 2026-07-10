"""SQLite-backed input-moderation tests for ``AdvancedChatAppRunner``."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session, sessionmaker

import core.app.apps.advanced_chat.app_runner as module
from core.app.apps.advanced_chat.app_runner import AdvancedChatAppRunner
from core.app.entities.app_invoke_entities import AdvancedChatAppGenerateEntity, InvokeFrom
from core.app.entities.queue_entities import QueueStopEvent
from core.moderation.base import ModerationError
from models import ConversationVariable
from models.model import App, AppMode, Conversation, Message
from models.workflow import Workflow, WorkflowType

MINIMAL_GRAPH = {
    "nodes": [
        {
            "id": "start",
            "data": {
                "type": "start",
                "title": "Start",
            },
        }
    ],
    "edges": [],
}

pytestmark = pytest.mark.parametrize("sqlite_session", [(App, ConversationVariable)], indirect=True)


@pytest.fixture
def build_runner(sqlite_session: Session) -> AdvancedChatAppRunner:
    """Construct a runner with real ORM entities and SQLite-backed session work."""
    app_id = str(uuid4())
    workflow_id = str(uuid4())
    tenant_id = str(uuid4())

    app = App(
        id=app_id,
        tenant_id=tenant_id,
        name="Test advanced chat app",
        description="",
        mode=AppMode.ADVANCED_CHAT,
        workflow_id=workflow_id,
        enable_site=True,
        enable_api=True,
        max_active_requests=0,
    )
    sqlite_session.add(app)
    sqlite_session.commit()

    # Queue delivery remains an external boundary for these runner tests.
    mock_queue_manager = MagicMock()

    conversation = Conversation(id=str(uuid4()), app_id=app_id)
    message = Message(id=str(uuid4()))
    workflow = Workflow(
        id=workflow_id,
        tenant_id=tenant_id,
        app_id=app_id,
        type=WorkflowType.CHAT,
        version=Workflow.VERSION_DRAFT,
        graph=json.dumps(MINIMAL_GRAPH),
        features="{}",
        created_by=str(uuid4()),
        environment_variables=[],
        conversation_variables=[],
        rag_pipeline_variables=[],
    )

    mock_app_config = MagicMock()
    mock_app_config.app_id = app_id
    mock_app_config.workflow_id = workflow_id
    mock_app_config.tenant_id = str(uuid4())

    gen = MagicMock(spec=AdvancedChatAppGenerateEntity)
    gen.app_config = mock_app_config
    gen.inputs = {"q": "raw"}
    gen.query = "raw-query"
    gen.files = []
    gen.user_id = str(uuid4())
    gen.invoke_from = InvokeFrom.SERVICE_API
    gen.workflow_run_id = str(uuid4())
    gen.task_id = str(uuid4())
    gen.call_depth = 0
    gen.single_iteration_run = None
    gen.single_loop_run = None
    gen.extras = {}
    gen.trace_manager = None

    runner = AdvancedChatAppRunner(
        application_generate_entity=gen,
        queue_manager=mock_queue_manager,
        conversation=conversation,
        message=message,
        dialogue_count=1,
        variable_loader=MagicMock(),
        workflow=workflow,
        system_user_id=str(uuid4()),
        app=app,
        workflow_execution_repository=MagicMock(),
        workflow_node_execution_repository=MagicMock(),
    )

    return runner


@contextmanager
def _patch_common_run_deps(sqlite_session: Session, *, workflow_entry: MagicMock | None = None) -> Iterator[None]:
    """Bind runner-owned sessions to SQLite while isolating graph and Redis boundaries."""
    session_maker = sessionmaker(bind=sqlite_session.get_bind(), expire_on_commit=False)
    workflow_entry = workflow_entry or MagicMock(**{"run.return_value": iter([])})

    with (
        patch.multiple(
            module,
            create_session=session_maker,
            RedisChannel=MagicMock(),
            redis_client=MagicMock(),
            WorkflowEntry=MagicMock(return_value=workflow_entry),
            GraphRuntimeState=MagicMock(),
        ),
        patch.object(module.session_factory, "get_session_maker", new=lambda: session_maker),
    ):
        yield


def test_handle_input_moderation_stops_on_moderation_error(build_runner):
    runner = build_runner

    # moderation_for_inputs raises ModerationError -> should stop and emit stop event
    with (
        patch.object(runner, "moderation_for_inputs", side_effect=ModerationError("blocked")),
        patch.object(runner, "_complete_with_stream_output") as mock_complete,
    ):
        stop, new_inputs, new_query = runner.handle_input_moderation(
            app_record=runner._app,
            app_generate_entity=runner.application_generate_entity,
            inputs={"k": "v"},
            query="hello",
            message_id="mid",
        )

        assert stop is True
        # inputs/query should be unchanged on error path
        assert new_inputs == {"k": "v"}
        assert new_query == "hello"
        # ensure stopped_by reason is INPUT_MODERATION
        assert mock_complete.called
        args, kwargs = mock_complete.call_args
        assert kwargs.get("stopped_by") == QueueStopEvent.StopBy.INPUT_MODERATION


def test_run_applies_overridden_inputs_and_query_from_moderation(
    build_runner: AdvancedChatAppRunner, sqlite_session: Session
) -> None:
    runner = build_runner

    overridden_inputs = {"q": "sanitized"}
    overridden_query = "sanitized-query"

    with (
        _patch_common_run_deps(sqlite_session),
        patch.object(
            runner,
            "moderation_for_inputs",
            return_value=(True, overridden_inputs, overridden_query),
        ) as mock_moderate,
        patch.object(runner, "handle_annotation_reply", return_value=False) as mock_anno,
        patch.object(runner, "_init_graph", return_value=MagicMock()) as mock_init_graph,
    ):
        runner.run()

        # moderation called with original values
        mock_moderate.assert_called_once()

        # application_generate_entity should be updated to overridden values
        assert runner.application_generate_entity.inputs == overridden_inputs
        assert runner.application_generate_entity.query == overridden_query

        # annotation reply should use the new query
        mock_anno.assert_called()
        assert mock_anno.call_args.kwargs.get("query") == overridden_query

        # since not stopped, graph initialization should proceed
        assert mock_init_graph.called


def test_run_returns_early_when_direct_output_via_handle_input_moderation(
    build_runner: AdvancedChatAppRunner, sqlite_session: Session
) -> None:
    runner = build_runner

    with (
        _patch_common_run_deps(sqlite_session),
        # Simulate handle_input_moderation signalling to stop
        patch.object(
            runner,
            "handle_input_moderation",
            return_value=(True, runner.application_generate_entity.inputs, runner.application_generate_entity.query),
        ) as mock_handle,
        patch.object(runner, "_init_graph") as mock_init_graph,
        patch.object(runner, "handle_annotation_reply") as mock_anno,
    ):
        runner.run()

        mock_handle.assert_called_once()
        # Ensure no further steps executed
        mock_anno.assert_not_called()
        mock_init_graph.assert_not_called()


def test_run_closes_scoped_session_before_workflow_run(
    build_runner: AdvancedChatAppRunner, sqlite_session: Session
) -> None:
    runner = build_runner
    events = []

    workflow_entry = MagicMock()

    def run_workflow():
        events.append("run")
        return iter([])

    workflow_entry.run.side_effect = run_workflow

    with (
        _patch_common_run_deps(sqlite_session, workflow_entry=workflow_entry),
        patch.object(module.db.session, "close", new=lambda: events.append("close")),
        patch.object(
            runner,
            "handle_input_moderation",
            return_value=(False, runner.application_generate_entity.inputs, runner.application_generate_entity.query),
        ),
        patch.object(runner, "handle_annotation_reply", return_value=False),
        patch.object(runner, "_init_graph", return_value=MagicMock()),
    ):
        runner.run()

    assert events == ["close", "run"]

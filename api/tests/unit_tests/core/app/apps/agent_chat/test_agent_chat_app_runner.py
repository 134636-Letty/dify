"""Tests for selecting and invoking the agent-chat runner.

The runner's application, conversation, and message lookups use real, short-lived
SQLite sessions. External model providers, moderation, queues, and concrete agent
runners remain mocked at their I/O boundaries.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pytest_mock import MockerFixture
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from core.agent.entities import AgentEntity
from core.app.apps.agent_chat.app_runner import AgentChatAppRunner
from core.moderation.base import ModerationError
from graphon.model_runtime.entities.llm_entities import LLMMode
from graphon.model_runtime.entities.model_entities import ModelFeature, ModelPropertyKey
from models.base import TypeBase
from models.enums import ConversationFromSource
from models.model import App, AppMode, Conversation, Message


@dataclass(frozen=True)
class _Database:
    """Hold the caller session and factory for runner-owned read sessions."""

    session: Session
    session_factory: sessionmaker[Session]


@dataclass(frozen=True)
class _Records:
    app: App
    conversation: Conversation
    message: Message


@pytest.fixture
def database(sqlite_engine: Engine) -> Iterator[_Database]:
    """Create the runner's three queried tables on an isolated SQLite engine."""

    models = (App, Conversation, Message)
    tables = [TypeBase.metadata.tables[model.__tablename__] for model in models]
    TypeBase.metadata.create_all(sqlite_engine, tables=tables)
    factory = sessionmaker(bind=sqlite_engine, expire_on_commit=False)
    with factory() as session:
        yield _Database(session=session, session_factory=factory)


@pytest.fixture
def runner() -> AgentChatAppRunner:
    return AgentChatAppRunner()


def _persist_records(session: Session) -> _Records:
    app = App(
        id=str(uuid4()),
        tenant_id=str(uuid4()),
        name="Agent chat",
        description="",
        mode=AppMode.AGENT_CHAT,
        icon_type=None,
        icon="",
        icon_background=None,
        enable_site=True,
        enable_api=True,
    )
    conversation = Conversation(
        id=str(uuid4()),
        app_id=app.id,
        mode=AppMode.AGENT_CHAT,
        name="Conversation",
        _inputs={},
        from_source=ConversationFromSource.API,
    )
    message = Message(
        id=str(uuid4()),
        app_id=app.id,
        conversation_id=conversation.id,
        _inputs={},
        query="q",
        message={},
        message_unit_price=Decimal(0),
        answer="answer",
        answer_unit_price=Decimal(0),
        currency="USD",
        from_source=ConversationFromSource.API,
    )
    session.add_all([app, conversation, message])
    session.commit()
    return _Records(app=app, conversation=conversation, message=message)


def _patch_owned_sessions(mocker: MockerFixture, database: _Database) -> None:
    """Bind each production ``create_session`` scope to the SQLite engine."""

    mocker.patch("core.app.apps.agent_chat.app_runner.create_session", side_effect=database.session_factory)


def _generate_entity(
    mocker: MockerFixture,
    records: _Records,
    *,
    strategy: AgentEntity.Strategy | str = AgentEntity.Strategy.CHAIN_OF_THOUGHT,
) -> MagicMock:
    agent = (
        AgentEntity(provider="p", model="m", strategy=strategy)
        if isinstance(strategy, AgentEntity.Strategy)
        else mocker.MagicMock(strategy=strategy, provider="p", model="m")
    )
    app_config = mocker.MagicMock(
        app_id=records.app.id,
        tenant_id=records.app.tenant_id,
        prompt_template=mocker.MagicMock(),
        agent=agent,
        external_data_variables=[],
    )
    return mocker.MagicMock(
        app_config=app_config,
        inputs={},
        query="q",
        files=[],
        stream=True,
        model_conf=mocker.MagicMock(
            provider_model_bundle=mocker.MagicMock(),
            model="m",
            provider="p",
            credentials={"k": "v"},
        ),
        conversation_id=None,
        invoke_from=mocker.MagicMock(),
        user_id="user",
    )


def _patch_pre_agent_flow(mocker: MockerFixture, runner: AgentChatAppRunner) -> None:
    mocker.patch.object(runner, "organize_prompt_messages", return_value=([], None))
    mocker.patch.object(runner, "moderation_for_inputs", return_value=(None, {}, "q"))
    mocker.patch.object(runner, "query_app_annotations_to_reply", return_value=None)
    mocker.patch.object(runner, "check_hosting_moderation", return_value=False)


def _patch_model_schema(
    mocker: MockerFixture,
    *,
    mode: LLMMode | str = LLMMode.CHAT,
    features: list[ModelFeature] | None = None,
) -> None:
    model_schema = mocker.MagicMock(
        features=features or [],
        model_properties={ModelPropertyKey.MODE: mode},
    )
    llm_instance = mocker.MagicMock()
    llm_instance.model_type_instance.get_model_schema.return_value = model_schema
    mocker.patch("core.app.apps.agent_chat.app_runner.ModelInstance", return_value=llm_instance)


class TestAgentChatAppRunnerRun:
    def test_run_app_not_found(self, runner: AgentChatAppRunner, database: _Database, mocker: MockerFixture) -> None:
        app_config = mocker.MagicMock(app_id=str(uuid4()), tenant_id=str(uuid4()), agent=mocker.MagicMock())
        generate_entity = mocker.MagicMock(app_config=app_config, inputs={}, query="q", files=[], stream=True)
        _patch_owned_sessions(mocker, database)

        with pytest.raises(ValueError, match="App not found"):
            runner.run(database.session, generate_entity, mocker.MagicMock(), mocker.MagicMock(), mocker.MagicMock())

    def test_run_moderation_error_direct_output(
        self, runner: AgentChatAppRunner, database: _Database, mocker: MockerFixture
    ) -> None:
        records = _persist_records(database.session)
        generate_entity = _generate_entity(mocker, records)
        _patch_owned_sessions(mocker, database)
        mocker.patch.object(runner, "organize_prompt_messages", return_value=([], None))
        mocker.patch.object(runner, "moderation_for_inputs", side_effect=ModerationError("bad"))
        mocker.patch.object(runner, "direct_output")

        runner.run(database.session, generate_entity, mocker.MagicMock(), records.conversation, records.message)

        runner.direct_output.assert_called_once()

    def test_run_annotation_reply_short_circuits(
        self, runner: AgentChatAppRunner, database: _Database, mocker: MockerFixture
    ) -> None:
        records = _persist_records(database.session)
        generate_entity = _generate_entity(mocker, records)
        _patch_owned_sessions(mocker, database)
        mocker.patch.object(runner, "organize_prompt_messages", return_value=([], None))
        mocker.patch.object(runner, "moderation_for_inputs", return_value=(None, {}, "q"))
        annotation = mocker.MagicMock(id=str(uuid4()), content="answer")
        mocker.patch.object(runner, "query_app_annotations_to_reply", return_value=annotation)
        mocker.patch.object(runner, "direct_output")
        queue_manager = mocker.MagicMock()

        runner.run(database.session, generate_entity, queue_manager, records.conversation, records.message)

        queue_manager.publish.assert_called_once()
        runner.direct_output.assert_called_once()

    def test_run_hosting_moderation_short_circuits(
        self, runner: AgentChatAppRunner, database: _Database, mocker: MockerFixture
    ) -> None:
        records = _persist_records(database.session)
        generate_entity = _generate_entity(mocker, records)
        _patch_owned_sessions(mocker, database)
        _patch_pre_agent_flow(mocker, runner)
        runner.check_hosting_moderation.return_value = True

        runner.run(database.session, generate_entity, mocker.MagicMock(), records.conversation, records.message)

    def test_run_model_schema_missing(
        self, runner: AgentChatAppRunner, database: _Database, mocker: MockerFixture
    ) -> None:
        records = _persist_records(database.session)
        generate_entity = _generate_entity(mocker, records)
        _patch_owned_sessions(mocker, database)
        _patch_pre_agent_flow(mocker, runner)
        llm_instance = mocker.MagicMock()
        llm_instance.model_type_instance.get_model_schema.return_value = None
        mocker.patch("core.app.apps.agent_chat.app_runner.ModelInstance", return_value=llm_instance)

        with pytest.raises(ValueError, match="Model schema not found"):
            runner.run(database.session, generate_entity, mocker.MagicMock(), records.conversation, records.message)

    @pytest.mark.parametrize(
        ("mode", "expected_runner"),
        [(LLMMode.CHAT, "CotChatAgentRunner"), (LLMMode.COMPLETION, "CotCompletionAgentRunner")],
    )
    def test_run_chain_of_thought_modes(
        self,
        runner: AgentChatAppRunner,
        database: _Database,
        mocker: MockerFixture,
        mode: LLMMode,
        expected_runner: str,
    ) -> None:
        records = _persist_records(database.session)
        generate_entity = _generate_entity(mocker, records)
        _patch_owned_sessions(mocker, database)
        _patch_pre_agent_flow(mocker, runner)
        _patch_model_schema(mocker, mode=mode)
        runner_cls = mocker.patch(f"core.app.apps.agent_chat.app_runner.{expected_runner}")
        runner_cls.return_value.run.return_value = []
        mocker.patch.object(runner, "_handle_invoke_result")

        runner.run(database.session, generate_entity, mocker.MagicMock(), records.conversation, records.message)

        runner_cls.return_value.run.assert_called_once()
        runner._handle_invoke_result.assert_called_once()

    def test_run_invalid_llm_mode_raises(
        self, runner: AgentChatAppRunner, database: _Database, mocker: MockerFixture
    ) -> None:
        records = _persist_records(database.session)
        generate_entity = _generate_entity(mocker, records)
        _patch_owned_sessions(mocker, database)
        _patch_pre_agent_flow(mocker, runner)
        _patch_model_schema(mocker, mode="invalid")

        with pytest.raises(ValueError, match="Invalid LLM mode"):
            runner.run(database.session, generate_entity, mocker.MagicMock(), records.conversation, records.message)

    def test_run_function_calling_strategy_selected_by_features(
        self, runner: AgentChatAppRunner, database: _Database, mocker: MockerFixture
    ) -> None:
        records = _persist_records(database.session)
        generate_entity = _generate_entity(mocker, records)
        _patch_owned_sessions(mocker, database)
        _patch_pre_agent_flow(mocker, runner)
        _patch_model_schema(mocker, features=[ModelFeature.TOOL_CALL])
        runner_cls = mocker.patch("core.app.apps.agent_chat.app_runner.FunctionCallAgentRunner")
        runner_cls.return_value.run.return_value = []
        mocker.patch.object(runner, "_handle_invoke_result")

        runner.run(database.session, generate_entity, mocker.MagicMock(), records.conversation, records.message)

        assert generate_entity.app_config.agent.strategy == AgentEntity.Strategy.FUNCTION_CALLING
        runner_cls.return_value.run.assert_called_once()

    def test_run_conversation_not_found(
        self, runner: AgentChatAppRunner, database: _Database, mocker: MockerFixture
    ) -> None:
        records = _persist_records(database.session)
        generate_entity = _generate_entity(mocker, records, strategy=AgentEntity.Strategy.FUNCTION_CALLING)
        database.session.delete(records.conversation)
        database.session.commit()
        _patch_owned_sessions(mocker, database)
        _patch_pre_agent_flow(mocker, runner)
        _patch_model_schema(mocker)

        with pytest.raises(ValueError, match="Conversation not found"):
            runner.run(database.session, generate_entity, mocker.MagicMock(), records.conversation, records.message)

    def test_run_message_not_found(
        self, runner: AgentChatAppRunner, database: _Database, mocker: MockerFixture
    ) -> None:
        records = _persist_records(database.session)
        generate_entity = _generate_entity(mocker, records, strategy=AgentEntity.Strategy.FUNCTION_CALLING)
        database.session.delete(records.message)
        database.session.commit()
        _patch_owned_sessions(mocker, database)
        _patch_pre_agent_flow(mocker, runner)
        _patch_model_schema(mocker)

        with pytest.raises(ValueError, match="Message not found"):
            runner.run(database.session, generate_entity, mocker.MagicMock(), records.conversation, records.message)

    def test_run_invalid_agent_strategy_raises(
        self, runner: AgentChatAppRunner, database: _Database, mocker: MockerFixture
    ) -> None:
        records = _persist_records(database.session)
        generate_entity = _generate_entity(mocker, records, strategy="invalid")
        _patch_owned_sessions(mocker, database)
        _patch_pre_agent_flow(mocker, runner)
        _patch_model_schema(mocker)

        with pytest.raises(ValueError, match="Invalid agent strategy"):
            runner.run(database.session, generate_entity, mocker.MagicMock(), records.conversation, records.message)

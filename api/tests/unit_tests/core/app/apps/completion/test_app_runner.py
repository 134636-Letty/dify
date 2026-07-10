from types import SimpleNamespace
from unittest.mock import ANY, MagicMock
from uuid import uuid4

import pytest
from pytest_mock import MockerFixture
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

import core.app.apps.completion.app_runner as module
from core.app.apps.completion.app_runner import CompletionAppRunner
from core.moderation.base import ModerationError
from graphon.model_runtime.entities.message_entities import ImagePromptMessageContent
from models.model import App, AppMode, IconType, Message


@pytest.fixture
def runner():
    return CompletionAppRunner()


def _build_app_config(*, app_id: str, tenant_id: str, dataset=None, external_tools=None, additional_features=None):
    app_config = MagicMock()
    app_config.app_id = app_id
    app_config.tenant_id = tenant_id
    app_config.prompt_template = MagicMock()
    app_config.dataset = dataset
    app_config.external_data_variables = external_tools or []
    app_config.additional_features = additional_features
    app_config.app_model_config_dict = {"file_upload": {"enabled": True}}
    return app_config


def _build_generate_entity(app_config, file_upload_config=None):
    model_conf = MagicMock(
        provider_model_bundle="bundle",
        model="model",
        parameters={"max_tokens": 10},
        stop=["stop"],
    )
    return SimpleNamespace(
        app_config=app_config,
        model_conf=model_conf,
        inputs={"qvar": "query_from_input"},
        query="original_query",
        files=[],
        file_upload_config=file_upload_config,
        stream=True,
        user_id="user",
        invoke_from=MagicMock(),
    )


class _DatabaseBinding:
    session: Session

    def __init__(self, session: Session) -> None:
        self.session = session


class _TrackingSession(Session):
    events: list[str]

    def __init__(self, *, bind: Engine, events: list[str]) -> None:
        super().__init__(bind=bind, expire_on_commit=False)
        self.events = events

    def close(self) -> None:
        self.events.append("close")
        super().close()


def _bind_sessions(monkeypatch: pytest.MonkeyPatch, sqlite_engine: Engine, sqlite_session: Session) -> None:
    monkeypatch.setattr(module, "create_session", sessionmaker(bind=sqlite_engine, expire_on_commit=False))
    monkeypatch.setattr(module, "db", _DatabaseBinding(sqlite_session))


def _app(sqlite_session: Session) -> App:
    app = App(
        tenant_id=str(uuid4()),
        name="Completion App",
        description="",
        mode=AppMode.COMPLETION,
        icon_type=IconType.EMOJI,
        icon="🤖",
        icon_background="#fff",
        enable_site=True,
        enable_api=True,
        max_active_requests=0,
    )
    sqlite_session.add(app)
    sqlite_session.commit()
    return app


def _message(app: App) -> Message:
    return Message(
        app_id=app.id,
        model_provider="provider",
        model_id="model",
        override_model_configs=None,
        conversation_id=str(uuid4()),
        inputs={},
        query="query",
        message="",
        message_tokens=0,
        message_unit_price=0,
        message_price_unit=0,
        answer="",
        answer_tokens=0,
        answer_unit_price=0,
        answer_price_unit=0,
        parent_message_id=None,
        provider_response_latency=0,
        total_price=0,
        currency="USD",
        invoke_from="web-app",
        from_source="api",
        from_end_user_id=str(uuid4()),
        from_account_id=None,
        app_mode=AppMode.COMPLETION,
    )


@pytest.mark.parametrize("sqlite_session", [(App,)], indirect=True)
class TestCompletionAppRunner:
    def test_run_app_not_found(
        self,
        runner,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
        sqlite_engine: Engine,
        sqlite_session: Session,
    ):
        _bind_sessions(monkeypatch, sqlite_engine, sqlite_session)
        app_config = _build_app_config(app_id=str(uuid4()), tenant_id=str(uuid4()))
        app_generate_entity = _build_generate_entity(app_config)

        with pytest.raises(ValueError):
            runner.run(sqlite_session, app_generate_entity, MagicMock(), MagicMock())

    def test_run_moderation_error_outputs_direct(
        self,
        runner,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
        sqlite_engine: Engine,
        sqlite_session: Session,
    ):
        _bind_sessions(monkeypatch, sqlite_engine, sqlite_session)
        app_record = _app(sqlite_session)

        app_config = _build_app_config(app_id=app_record.id, tenant_id=app_record.tenant_id)
        app_generate_entity = _build_generate_entity(app_config)

        runner.organize_prompt_messages = MagicMock(return_value=([], None))
        runner.moderation_for_inputs = MagicMock(side_effect=ModerationError("blocked"))
        runner.direct_output = MagicMock()
        runner._handle_invoke_result = MagicMock()

        runner.run(sqlite_session, app_generate_entity, MagicMock(), _message(app_record))

        runner.direct_output.assert_called_once()
        runner._handle_invoke_result.assert_not_called()

    def test_run_hosting_moderation_stops(
        self,
        runner,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
        sqlite_engine: Engine,
        sqlite_session: Session,
    ):
        _bind_sessions(monkeypatch, sqlite_engine, sqlite_session)
        app_record = _app(sqlite_session)

        app_config = _build_app_config(app_id=app_record.id, tenant_id=app_record.tenant_id)
        app_generate_entity = _build_generate_entity(app_config)

        runner.organize_prompt_messages = MagicMock(return_value=([], None))
        runner.moderation_for_inputs = MagicMock(return_value=(None, app_generate_entity.inputs, "query"))
        runner.check_hosting_moderation = MagicMock(return_value=True)
        runner._handle_invoke_result = MagicMock()

        runner.run(sqlite_session, app_generate_entity, MagicMock(), _message(app_record))

        runner._handle_invoke_result.assert_not_called()

    def test_run_dataset_and_external_tools_flow(
        self,
        runner,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
        sqlite_engine: Engine,
        sqlite_session: Session,
    ):
        _bind_sessions(monkeypatch, sqlite_engine, sqlite_session)
        app_record = _app(sqlite_session)

        retrieve_config = MagicMock(query_variable="qvar")
        dataset_config = MagicMock(dataset_ids=["ds"], retrieve_config=retrieve_config)
        additional_features = MagicMock(show_retrieve_source=True)
        app_config = _build_app_config(
            app_id=app_record.id,
            tenant_id=app_record.tenant_id,
            dataset=dataset_config,
            external_tools=["tool"],
            additional_features=additional_features,
        )

        file_upload_config = MagicMock()
        file_upload_config.image_config.detail = ImagePromptMessageContent.DETAIL.HIGH

        app_generate_entity = _build_generate_entity(app_config, file_upload_config=file_upload_config)

        runner.organize_prompt_messages = MagicMock(side_effect=[(["pm1"], ["stop"]), (["pm2"], ["stop"])])
        runner.moderation_for_inputs = MagicMock(return_value=(None, app_generate_entity.inputs, "query"))
        runner.fill_in_inputs_from_external_data_tools = MagicMock(return_value=app_generate_entity.inputs)
        runner.check_hosting_moderation = MagicMock(return_value=False)
        runner.recalc_llm_max_tokens = MagicMock()
        runner._handle_invoke_result = MagicMock()

        dataset_retrieval = MagicMock()
        dataset_retrieval.retrieve.return_value = ("ctx", ["file1"])
        mocker.patch.object(module, "DatasetRetrieval", return_value=dataset_retrieval)

        model_instance = MagicMock()
        model_instance.invoke_llm.return_value = "invoke_result"
        mocker.patch.object(module, "ModelInstance", return_value=model_instance)

        runner.run(sqlite_session, app_generate_entity, MagicMock(), _message(app_record))

        dataset_retrieval.retrieve.assert_called_once()
        assert dataset_retrieval.retrieve.call_args.kwargs["query"] == "query_from_input"
        runner._handle_invoke_result.assert_called_once()

    def test_run_closes_scoped_session_before_stream_consumption(
        self,
        runner,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
        sqlite_engine: Engine,
        sqlite_session: Session,
    ):
        app_record = _app(sqlite_session)
        app_config = _build_app_config(app_id=app_record.id, tenant_id=app_record.tenant_id)
        app_generate_entity = _build_generate_entity(app_config)
        queue_manager = MagicMock()

        events = []
        runner.organize_prompt_messages = MagicMock(return_value=([], None))
        runner.moderation_for_inputs = MagicMock(return_value=(None, app_generate_entity.inputs, "query"))
        runner.check_hosting_moderation = MagicMock(return_value=False)
        runner.recalc_llm_max_tokens = MagicMock()
        runner._handle_invoke_result = MagicMock(side_effect=lambda invoke_result, **kwargs: list(invoke_result))

        model_instance = MagicMock()

        def invoke_stream():
            events.append("first-chunk")
            yield "chunk"

        def invoke_llm(**kwargs):
            events.append("invoke")
            return invoke_stream()

        model_instance.invoke_llm.side_effect = invoke_llm
        mocker.patch.object(module, "ModelInstance", return_value=model_instance)
        monkeypatch.setattr(module, "create_session", sessionmaker(bind=sqlite_engine, expire_on_commit=False))
        tracking_session = _TrackingSession(bind=sqlite_engine, events=events)
        monkeypatch.setattr(module, "db", _DatabaseBinding(tracking_session))

        message = _message(app_record)
        runner.run(sqlite_session, app_generate_entity, queue_manager, message)

        assert events == ["close", "invoke", "first-chunk"]
        runner._handle_invoke_result.assert_called_once_with(
            invoke_result=ANY,
            queue_manager=queue_manager,
            stream=True,
            message_id=message.id,
            user_id="user",
            tenant_id=app_record.tenant_id,
        )

    def test_run_uses_low_image_detail_default(
        self,
        runner,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
        sqlite_engine: Engine,
        sqlite_session: Session,
    ):
        _bind_sessions(monkeypatch, sqlite_engine, sqlite_session)
        app_record = _app(sqlite_session)

        app_config = _build_app_config(app_id=app_record.id, tenant_id=app_record.tenant_id)
        app_generate_entity = _build_generate_entity(app_config, file_upload_config=None)

        runner.organize_prompt_messages = MagicMock(return_value=([], None))
        runner.moderation_for_inputs = MagicMock(return_value=(None, app_generate_entity.inputs, "query"))
        runner.check_hosting_moderation = MagicMock(return_value=True)

        runner.run(sqlite_session, app_generate_entity, MagicMock(), _message(app_record))

        assert (
            runner.organize_prompt_messages.call_args.kwargs["image_detail_config"]
            == ImagePromptMessageContent.DETAIL.LOW
        )

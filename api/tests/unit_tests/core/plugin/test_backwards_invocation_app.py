import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from pydantic import BaseModel
from pytest_mock import MockerFixture
from sqlalchemy import Engine, event
from sqlalchemy.orm import Session, sessionmaker

from core.app.layers.pause_state_persist_layer import PauseStateLayerConfig
from core.plugin.backwards_invocation.app import PluginAppBackwardsInvocation
from core.plugin.backwards_invocation.base import BaseBackwardsInvocation
from extensions.ext_database import db
from models import Account, TenantAccountJoin
from models.base import TypeBase
from models.enums import EndUserType
from models.model import App, AppMode, AppModelConfig, EndUser, IconType
from models.workflow import Workflow, WorkflowType


class _Chunk(BaseModel):
    value: int


@pytest.fixture
def orm_session(sqlite_engine: Engine) -> Iterator[Session]:
    models = (App, EndUser, Account, TenantAccountJoin, Workflow, AppModelConfig)
    tables = [model.metadata.tables[model.__tablename__] for model in models]
    TypeBase.metadata.create_all(sqlite_engine, tables=tables)
    session_maker = sessionmaker(bind=sqlite_engine, expire_on_commit=False)

    with (
        patch("core.plugin.backwards_invocation.app.create_session", new=session_maker),
        patch.object(type(db), "engine", new_callable=PropertyMock, return_value=sqlite_engine),
    ):
        with session_maker() as session:
            yield session


def _persist_app(
    session: Session,
    *,
    mode: AppMode,
    tenant_id: str | None = None,
) -> App:
    app = App(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id or str(uuid.uuid4()),
        name="Plugin App",
        mode=mode,
        icon_type=IconType.EMOJI,
        icon="plugin",
        icon_background="#FFFFFF",
        enable_site=True,
        enable_api=False,
    )
    session.add(app)
    session.commit()
    return app


def _persist_end_user(
    session: Session,
    app: App,
    *,
    session_id: str | None = None,
) -> EndUser:
    end_user = EndUser(
        id=str(uuid.uuid4()),
        tenant_id=app.tenant_id,
        app_id=app.id,
        type=EndUserType.BROWSER,
        name="Plugin User",
        session_id=session_id or str(uuid.uuid4()),
    )
    session.add(end_user)
    session.commit()
    return end_user


def _persist_account(session: Session, app: App, *, joined: bool = True) -> Account:
    account = Account(name="Plugin Account", email=f"{uuid.uuid4()}@example.com")
    session.add(account)
    if joined:
        session.add(TenantAccountJoin(tenant_id=app.tenant_id, account_id=account.id))
    session.commit()
    return account


def _persist_workflow(session: Session, app: App) -> Workflow:
    graph = {
        "nodes": [
            {
                "data": {
                    "type": "start",
                    "variables": [{"type": "text-input", "variable": "foo", "label": "Foo"}],
                }
            }
        ]
    }
    workflow = Workflow.new(
        tenant_id=app.tenant_id,
        app_id=app.id,
        type=WorkflowType.WORKFLOW.value,
        version="1",
        graph=json.dumps(graph),
        features=json.dumps({"feature": "v"}),
        created_by="owner-id",
        environment_variables=[],
        conversation_variables=[],
        rag_pipeline_variables=[],
    )
    app.workflow_id = workflow.id
    session.add_all([workflow, app])
    session.commit()
    return workflow


def _persist_app_model_config(session: Session, app: App) -> AppModelConfig:
    app_model_config = AppModelConfig(
        app_id=app.id,
        user_input_form=json.dumps([{"name": "bar"}]),
    )
    app.app_model_config_id = app_model_config.id
    session.add_all([app_model_config, app])
    session.commit()
    return app_model_config


@contextmanager
def _raise_on_apps(engine: Engine) -> Iterator[None]:
    """Force only the App lookup SQL to fail while retaining a real Session."""

    def fail_app_query(_conn, _cursor, statement, _parameters, _context, _executemany):
        if "FROM apps" in statement:
            raise RuntimeError("forced app lookup failure")

    event.listen(engine, "before_cursor_execute", fail_app_query)
    try:
        yield
    finally:
        event.remove(engine, "before_cursor_execute", fail_app_query)


class TestBaseBackwardsInvocation:
    def test_convert_to_event_stream_with_generator_and_error(self):
        def _stream():
            yield _Chunk(value=1)
            yield {"x": 2}
            yield "ignored"
            raise RuntimeError("boom")

        chunks = list(BaseBackwardsInvocation.convert_to_event_stream(_stream()))

        assert len(chunks) == 3
        first = json.loads(chunks[0].decode())
        second = json.loads(chunks[1].decode())
        error = json.loads(chunks[2].decode())
        assert first["data"]["value"] == 1
        assert second["data"]["x"] == 2
        assert error["error"] == "boom"

    def test_convert_to_event_stream_with_non_generator(self):
        chunks = list(BaseBackwardsInvocation.convert_to_event_stream({"ok": True}))
        payload = json.loads(chunks[0].decode())
        assert payload["data"] == {"ok": True}
        assert payload["error"] == ""


class TestPluginAppBackwardsInvocation:
    def test_fetch_app_info_workflow_path(self, mocker: MockerFixture, orm_session: Session):
        app = _persist_app(orm_session, mode=AppMode.WORKFLOW)
        _persist_workflow(orm_session, app)
        mapper = mocker.patch(
            "core.plugin.backwards_invocation.app.get_parameters_from_feature_dict",
            return_value={"mapped": True},
        )

        result = PluginAppBackwardsInvocation.fetch_app_info(app.id, app.tenant_id)

        assert result == {"data": {"mapped": True}}
        mapper.assert_called_once_with(
            features_dict={"feature": "v"},
            user_input_form=[{"text-input": {"type": "text-input", "variable": "foo", "label": "Foo"}}],
        )

    def test_fetch_app_info_model_config_path(self, mocker: MockerFixture, orm_session: Session):
        app = _persist_app(orm_session, mode=AppMode.COMPLETION)
        _persist_app_model_config(orm_session, app)
        mocker.patch(
            "core.plugin.backwards_invocation.app.load_annotation_reply_config",
            return_value={"enabled": False},
        )
        mocker.patch(
            "core.plugin.backwards_invocation.app.get_parameters_from_feature_dict",
            return_value={"mapped": True},
        )

        result = PluginAppBackwardsInvocation.fetch_app_info(app.id, app.tenant_id)

        assert result["data"] == {"mapped": True}

    @pytest.mark.parametrize(
        ("mode", "route_method"),
        [
            (AppMode.CHAT, "invoke_chat_app"),
            (AppMode.ADVANCED_CHAT, "invoke_chat_app"),
            (AppMode.AGENT_CHAT, "invoke_chat_app"),
            (AppMode.WORKFLOW, "invoke_workflow_app"),
            (AppMode.COMPLETION, "invoke_completion_app"),
        ],
    )
    def test_invoke_app_routes_by_mode(
        self,
        mocker: MockerFixture,
        mode: AppMode,
        route_method: str,
        orm_session: Session,
    ):
        app = _persist_app(orm_session, mode=mode)
        user = _persist_end_user(orm_session, app)
        if mode == AppMode.WORKFLOW:
            _persist_workflow(orm_session, app)
        route = mocker.patch.object(PluginAppBackwardsInvocation, route_method, return_value={"routed": True})

        result = PluginAppBackwardsInvocation.invoke_app(
            orm_session,
            app_id=app.id,
            user_id=user.id,
            tenant_id=app.tenant_id,
            conversation_id=None,
            query="hello",
            stream=False,
            inputs={"x": 1},
            files=[],
        )

        assert result == {"routed": True}
        assert route.call_count == 1

    def test_invoke_app_uses_end_user_when_user_id_missing(self, mocker: MockerFixture, orm_session: Session):
        app = _persist_app(orm_session, mode=AppMode.WORKFLOW)
        workflow = _persist_workflow(orm_session, app)
        end_user = MagicMock()
        get_or_create = mocker.patch(
            "core.plugin.backwards_invocation.app.EndUserService.get_or_create_end_user",
            return_value=end_user,
        )
        route = mocker.patch.object(PluginAppBackwardsInvocation, "invoke_workflow_app", return_value={"ok": True})

        result = PluginAppBackwardsInvocation.invoke_app(
            orm_session,
            app_id=app.id,
            user_id="",
            tenant_id=app.tenant_id,
            conversation_id="",
            query=None,
            stream=True,
            inputs={},
            files=[],
        )

        assert result == {"ok": True}
        assert get_or_create.call_count == 1
        assert get_or_create.call_args.args[0].id == app.id
        assert route.call_args.args[1].id == workflow.id
        assert route.call_args.args[2] is end_user

    def test_invoke_app_missing_query_for_chat_raises(self, orm_session: Session):
        app = _persist_app(orm_session, mode=AppMode.CHAT)
        user = _persist_end_user(orm_session, app)

        with pytest.raises(ValueError, match="missing query"):
            PluginAppBackwardsInvocation.invoke_app(
                orm_session,
                app_id=app.id,
                user_id=user.id,
                tenant_id=app.tenant_id,
                conversation_id=None,
                query="",
                stream=False,
                inputs={},
                files=[],
            )

    def test_invoke_app_unexpected_mode_raises(self, orm_session: Session):
        app = _persist_app(orm_session, mode=AppMode.CHANNEL)
        user = _persist_end_user(orm_session, app)

        with pytest.raises(ValueError, match="unexpected app type"):
            PluginAppBackwardsInvocation.invoke_app(
                orm_session,
                app_id=app.id,
                user_id=user.id,
                tenant_id=app.tenant_id,
                conversation_id=None,
                query="q",
                stream=False,
                inputs={},
                files=[],
            )

    @pytest.mark.parametrize(
        ("mode", "generator_path"),
        [
            (AppMode.AGENT_CHAT, "core.plugin.backwards_invocation.app.AgentChatAppGenerator.generate"),
            (AppMode.CHAT, "core.plugin.backwards_invocation.app.ChatAppGenerator.generate"),
        ],
    )
    def test_invoke_chat_app_agent_and_chat(
        self,
        mocker: MockerFixture,
        mode: AppMode,
        generator_path: str,
        orm_session: Session,
    ):
        app = _persist_app(orm_session, mode=mode)
        user = _persist_end_user(orm_session, app)
        spy = mocker.patch(generator_path, return_value={"result": "ok"})

        result = PluginAppBackwardsInvocation.invoke_chat_app(
            orm_session,
            app=app,
            user=user,
            conversation_id="conv-1",
            query="hello",
            stream=False,
            inputs={"k": "v"},
            files=[],
        )

        assert result == {"result": "ok"}
        assert spy.call_count == 1

    def test_invoke_chat_app_advanced_chat_injects_pause_state_config(
        self, mocker: MockerFixture, orm_session: Session
    ):
        app = _persist_app(orm_session, mode=AppMode.ADVANCED_CHAT)
        _persist_workflow(orm_session, app)
        user = _persist_end_user(orm_session, app)
        generator_spy = mocker.patch(
            "core.plugin.backwards_invocation.app.AdvancedChatAppGenerator.generate",
            return_value={"result": "ok"},
        )

        result = PluginAppBackwardsInvocation.invoke_chat_app(
            orm_session,
            app=app,
            user=user,
            conversation_id="conv-1",
            query="hello",
            stream=False,
            inputs={"k": "v"},
            files=[],
        )

        assert result == {"result": "ok"}
        call_kwargs = generator_spy.call_args.kwargs
        pause_state_config = call_kwargs.get("pause_state_config")
        assert isinstance(pause_state_config, PauseStateLayerConfig)
        assert pause_state_config.state_owner_user_id == "owner-id"

    def test_invoke_chat_app_advanced_chat_without_workflow_raises(self, orm_session: Session):
        app = _persist_app(orm_session, mode=AppMode.ADVANCED_CHAT)
        user = _persist_end_user(orm_session, app)
        with pytest.raises(ValueError, match="unexpected app type"):
            PluginAppBackwardsInvocation.invoke_chat_app(
                orm_session,
                app=app,
                user=user,
                conversation_id="conv-1",
                query="hello",
                stream=False,
                inputs={},
                files=[],
            )

    def test_invoke_chat_app_unexpected_mode_raises(self, orm_session: Session):
        app = _persist_app(orm_session, mode=AppMode.CHANNEL)
        user = _persist_end_user(orm_session, app)
        with pytest.raises(ValueError, match="unexpected app type"):
            PluginAppBackwardsInvocation.invoke_chat_app(
                orm_session,
                app=app,
                user=user,
                conversation_id="conv-1",
                query="hello",
                stream=False,
                inputs={},
                files=[],
            )

    def test_invoke_workflow_app_injects_pause_state_config(self, mocker: MockerFixture, orm_session: Session):
        app = _persist_app(orm_session, mode=AppMode.WORKFLOW)
        workflow = _persist_workflow(orm_session, app)
        user = _persist_end_user(orm_session, app)
        generator_spy = mocker.patch(
            "core.plugin.backwards_invocation.app.WorkflowAppGenerator.generate",
            return_value={"result": "ok"},
        )

        result = PluginAppBackwardsInvocation.invoke_workflow_app(
            app=app,
            workflow=workflow,
            user=user,
            stream=False,
            inputs={"k": "v"},
            files=[],
        )

        assert result == {"result": "ok"}
        call_kwargs = generator_spy.call_args.kwargs
        pause_state_config = call_kwargs.get("pause_state_config")
        assert isinstance(pause_state_config, PauseStateLayerConfig)
        assert pause_state_config.state_owner_user_id == "owner-id"

    def test_invoke_app_workflow_without_workflow_raises(self, orm_session: Session):
        app = _persist_app(orm_session, mode=AppMode.WORKFLOW)
        user = _persist_end_user(orm_session, app)
        with pytest.raises(ValueError, match="unexpected app type"):
            PluginAppBackwardsInvocation.invoke_app(
                orm_session,
                app_id=app.id,
                user_id=user.id,
                tenant_id=app.tenant_id,
                conversation_id=None,
                query=None,
                stream=False,
                inputs={},
                files=[],
            )

    def test_invoke_completion_app(self, mocker: MockerFixture, orm_session: Session):
        spy = mocker.patch(
            "core.plugin.backwards_invocation.app.CompletionAppGenerator.generate", return_value={"ok": 1}
        )
        app = _persist_app(orm_session, mode=AppMode.COMPLETION)
        user = _persist_end_user(orm_session, app)

        result = PluginAppBackwardsInvocation.invoke_completion_app(orm_session, app, user, False, {"x": 1}, [])

        assert result == {"ok": 1}
        assert spy.call_count == 1

    def test_get_user_returns_end_user(self, orm_session: Session):
        app = _persist_app(orm_session, mode=AppMode.CHAT)
        end_user = _persist_end_user(orm_session, app)

        user = PluginAppBackwardsInvocation._get_user(end_user.id, app)

        assert user.id == end_user.id
        assert user.tenant_id == app.tenant_id
        assert user.app_id == app.id

    def test_get_user_returns_end_user_by_session_id(self, orm_session: Session):
        app = _persist_app(orm_session, mode=AppMode.CHAT)
        other_app = _persist_app(orm_session, mode=AppMode.CHAT, tenant_id=app.tenant_id)
        session_id = "wecom-sender-1"
        _persist_end_user(orm_session, other_app, session_id=session_id)
        target_user = _persist_end_user(orm_session, app, session_id=session_id)

        user = PluginAppBackwardsInvocation._get_user(session_id, app)

        assert user.id == target_user.id
        assert user.app_id == app.id

    def test_get_user_falls_back_to_account_user(self, orm_session: Session):
        app = _persist_app(orm_session, mode=AppMode.CHAT)
        account = _persist_account(orm_session, app)

        user = PluginAppBackwardsInvocation._get_user(account.id, app)

        assert user.id == account.id
        assert user.email == account.email

    def test_get_user_raises_when_user_not_found(self, orm_session: Session):
        app = _persist_app(orm_session, mode=AppMode.CHAT)
        other_app = _persist_app(orm_session, mode=AppMode.CHAT)
        account = _persist_account(orm_session, app, joined=False)
        _persist_end_user(orm_session, other_app, session_id=account.id)

        with pytest.raises(ValueError, match="user not found"):
            PluginAppBackwardsInvocation._get_user(account.id, app)

    def test_invoke_app_creates_end_user_for_unknown_external_user_id(
        self, mocker: MockerFixture, orm_session: Session
    ):
        app = _persist_app(orm_session, mode=AppMode.WORKFLOW)
        workflow = _persist_workflow(orm_session, app)
        end_user = MagicMock()
        get_or_create = mocker.patch(
            "core.plugin.backwards_invocation.app.EndUserService.get_or_create_end_user",
            return_value=end_user,
        )
        route = mocker.patch.object(PluginAppBackwardsInvocation, "invoke_workflow_app", return_value={"ok": True})

        result = PluginAppBackwardsInvocation.invoke_app(
            orm_session,
            app_id=app.id,
            user_id="wecom-sender-1",
            tenant_id=app.tenant_id,
            conversation_id="",
            query=None,
            stream=True,
            inputs={},
            files=[],
        )

        assert result == {"ok": True}
        assert get_or_create.call_count == 1
        assert get_or_create.call_args.args[0].id == app.id
        assert get_or_create.call_args.kwargs == {"user_id": "wecom-sender-1"}
        assert route.call_args.args[2] is end_user

    def test_get_app_returns_app(self, orm_session: Session):
        app = _persist_app(orm_session, mode=AppMode.CHAT)

        result = PluginAppBackwardsInvocation._get_app(app.id, app.tenant_id)
        assert result.id == app.id
        assert result.tenant_id == app.tenant_id

    def test_get_app_raises_when_missing(self, orm_session: Session):
        app = _persist_app(orm_session, mode=AppMode.CHAT)

        with pytest.raises(ValueError, match="app not found"):
            PluginAppBackwardsInvocation._get_app(app.id, str(uuid.uuid4()))

    def test_get_app_raises_when_query_fails(self, orm_session: Session, sqlite_engine: Engine):
        app = _persist_app(orm_session, mode=AppMode.CHAT)

        with _raise_on_apps(sqlite_engine), pytest.raises(ValueError, match="app not found"):
            PluginAppBackwardsInvocation._get_app(app.id, app.tenant_id)

    def test_get_workflow_stays_inside_app_boundary(self, orm_session: Session):
        app = _persist_app(orm_session, mode=AppMode.WORKFLOW)
        workflow = _persist_workflow(orm_session, app)

        result = PluginAppBackwardsInvocation._get_workflow(app)

        assert result is not None
        assert result.id == workflow.id
        assert result.tenant_id == app.tenant_id
        assert result.app_id == app.id

    def test_get_app_model_config_dict_uses_explicit_session_for_annotation_reply(
        self, mocker: MockerFixture, orm_session: Session
    ):
        annotation_reply = {"enabled": False}
        app = _persist_app(orm_session, mode=AppMode.COMPLETION)
        app_model_config = _persist_app_model_config(orm_session, app)
        load_annotation_reply_config = mocker.patch(
            "core.plugin.backwards_invocation.app.load_annotation_reply_config",
            return_value=annotation_reply,
        )
        result = PluginAppBackwardsInvocation._get_app_model_config_dict(app)

        assert result is not None
        assert result["user_input_form"] == [{"name": "bar"}]
        assert result["annotation_reply"] == annotation_reply
        load_args = load_annotation_reply_config.call_args.args
        assert isinstance(load_args[0], Session)
        assert load_args[1] == app.id
        assert app_model_config.app_id == app.id

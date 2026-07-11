from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from models.agent import Agent, AgentIconType, AgentScope, AgentSource, AgentStatus
from models.base import Base
from models.model import App, AppMode, AppStatus, IconType
from services.agent.errors import AgentNameConflictError
from services.app_service import AppService


@pytest.fixture
def orm_session(sqlite_engine: Engine) -> Iterator[Session]:
    """Provide persisted apps and backing agents through a real SQLite session."""

    Base.metadata.create_all(sqlite_engine, tables=[App.__table__, Agent.__table__])
    with sessionmaker(sqlite_engine, expire_on_commit=False)() as session:
        yield session


def _app(
    *,
    app_id: str,
    tenant_id: str = "tenant-1",
    name: str = "App",
    mode: AppMode = AppMode.CHAT,
    status: AppStatus = AppStatus.NORMAL,
) -> App:
    app = App()
    app.id = app_id
    app.tenant_id = tenant_id
    app.name = name
    app.description = "old"
    app.mode = mode
    app.icon_type = IconType.EMOJI
    app.icon = "robot"
    app.icon_background = "#fff"
    app.status = status
    app.enable_site = True
    app.enable_api = True
    app.max_active_requests = None
    app.created_by = "account-1"
    app.use_icon_as_answer_icon = False
    return app


def _agent(*, app_id: str, name: str = "Old") -> Agent:
    return Agent(
        tenant_id="tenant-1",
        name=name,
        description="old",
        role="research assistant",
        icon_type=AgentIconType.EMOJI,
        icon="robot",
        icon_background="#fff",
        scope=AgentScope.ROSTER,
        source=AgentSource.AGENT_APP,
        app_id=app_id,
        created_by="account-1",
    )


class TestOpenapiVisibilityHelpers:
    """Coverage for the session-injected, openapi-visibility-scoped
    ``AppService`` getters used by ``/openapi/v1/apps*``. These helpers
    centralise the "row exists + status normal + openapi-visibility
    gate passes" check so the controller can stay free of SQL.
    """

    def test_get_app_by_id_is_plain_session_get(self, orm_session: Session):
        """``get_app_by_id`` must NOT apply status / visibility filters
        — callers (e.g. the openapi auth pipeline) need to differentiate
        404 (missing) from 403 (``enable_api`` off) and would lose that
        signal if the helper coalesced both into ``None``.
        """
        sentinel_app = _app(app_id="app-uuid")
        orm_session.add(sentinel_app)
        orm_session.commit()
        sentinel_app.status = "archived"  # type: ignore[assignment]

        assert AppService.get_app_by_id("app-uuid", session=orm_session) is sentinel_app

    def test_get_app_by_id_returns_none_when_missing(self, orm_session: Session):
        assert AppService.get_app_by_id("missing", session=orm_session) is None

    def test_get_visible_app_by_id_returns_app_when_visible(self, orm_session: Session):
        app = _app(app_id="app-uuid")
        orm_session.add(app)
        orm_session.commit()

        with patch("services.app_service.is_openapi_visible", return_value=True):
            assert AppService.get_visible_app_by_id("app-uuid", session=orm_session) is app

    def test_get_visible_app_by_id_returns_none_when_row_missing(self, orm_session: Session):
        assert AppService.get_visible_app_by_id("missing", session=orm_session) is None

    def test_get_visible_app_by_id_returns_none_when_status_not_normal(self, orm_session: Session):
        """Soft-deleted/archived rows must not surface on the openapi
        surface — the helper hides them by returning ``None``.
        """
        app = _app(app_id="app-uuid")
        orm_session.add(app)
        orm_session.commit()
        app.status = "archived"  # type: ignore[assignment]

        with patch("services.app_service.is_openapi_visible", return_value=True):
            assert AppService.get_visible_app_by_id("app-uuid", session=orm_session) is None

    def test_get_visible_app_by_id_returns_none_when_visibility_gate_rejects(self, orm_session: Session):
        """``is_openapi_visible`` is the per-row counterpart to
        ``apply_openapi_gate`` — when it returns False the helper must
        treat the row as invisible (not "found but unauthorized").
        """
        orm_session.add(_app(app_id="app-uuid"))
        orm_session.commit()

        with patch("services.app_service.is_openapi_visible", return_value=False):
            assert AppService.get_visible_app_by_id("app-uuid", session=orm_session) is None

    def test_find_visible_apps_by_name_returns_scalars_through_visibility_gate(self, orm_session: Session):
        """Tenant-scoped name lookup. The helper passes the SELECT through
        ``apply_openapi_gate`` and materialises ``.scalars()`` into a list
        so the controller can branch on length (404 / single / 409).
        """
        rows = [_app(app_id="app-1", name="my-app"), _app(app_id="app-2", name="my-app")]
        orm_session.add_all([*rows, _app(app_id="other-tenant", tenant_id="tenant-2", name="my-app")])
        orm_session.commit()

        with patch("services.app_service.apply_openapi_gate", side_effect=lambda q: q) as gate:
            out = AppService.find_visible_apps_by_name(name="my-app", tenant_id="tenant-1", session=orm_session)

        assert {app.id for app in out} == {"app-1", "app-2"}
        # Visibility gate must wrap the SELECT exactly once.
        gate.assert_called_once()

    def test_find_visible_apps_by_name_returns_empty_list_on_no_match(self, orm_session: Session):
        with patch("services.app_service.apply_openapi_gate", side_effect=lambda q: q):
            out = AppService.find_visible_apps_by_name(name="nope", tenant_id="tenant-1", session=orm_session)

        assert out == []

    def test_find_visible_apps_by_ids_short_circuits_on_empty_input(self, orm_session: Session):
        """Empty id list must not emit ``WHERE id IN ()`` — Postgres
        rejects empty IN lists and the call is a guaranteed no-op
        anyway. The helper returns ``[]`` without touching the session.
        """
        assert AppService.find_visible_apps_by_ids([], session=orm_session) == []

    def test_find_visible_apps_by_ids_passes_through_visibility_gate(self, orm_session: Session):
        """Bulk fetch routes through ``apply_openapi_gate`` exactly once
        and materialises the scalar rows. **No** status filter is
        applied here — the EE permitted-external pipeline filters
        non-normal hits in Python so its page count stays anchored.
        """
        rows = [_app(app_id="a"), _app(app_id="b")]
        orm_session.add_all(rows)
        orm_session.commit()

        with patch("services.app_service.apply_openapi_gate", side_effect=lambda q: q) as gate:
            out = AppService.find_visible_apps_by_ids(["a", "b"], session=orm_session)

        assert {app.id for app in out} == {"a", "b"}
        gate.assert_called_once()


class TestAgentAppType:
    """S1: new ``AppMode.AGENT`` app type wiring."""

    @staticmethod
    def _persist_agent_app(session: Session) -> tuple[App, Agent]:
        app = _app(app_id="app-1", mode=AppMode.AGENT, name="Old")
        agent = _agent(app_id=app.id)
        session.add_all([app, agent])
        session.commit()
        return app, agent

    def test_agent_mode_enum_and_template_exist(self):
        from constants.model_template import default_app_templates
        from models.model import AppMode

        assert AppMode.AGENT.value == "agent"
        assert AppMode.AGENT in default_app_templates
        # Runtime config comes from the Agent Soul, so no model_config is seeded.
        assert "model_config" not in default_app_templates[AppMode.AGENT]
        assert default_app_templates[AppMode.AGENT]["app"]["mode"] == AppMode.AGENT

    def test_create_app_params_accepts_agent_mode(self):
        from services.app_service import CreateAppParams

        params = CreateAppParams(name="Iris", mode="agent")
        assert params.mode == "agent"

    def test_bound_agent_id_is_none_for_non_agent_app(self):
        """Non-agent apps short-circuit without touching the DB."""
        from models.model import App, AppMode

        app = App()
        app.mode = AppMode.CHAT
        assert app.bound_agent_id is None

    def test_update_agent_app_syncs_backing_agent_identity(self, orm_session: Session):
        app, backing_agent = self._persist_agent_app(orm_session)

        with patch("services.app_service.current_user", SimpleNamespace(id="account-2")):
            updated_app = AppService().update_app(
                app,
                {
                    "name": "Iris",
                    "description": "agent app",
                    "role": "research assistant",
                    "icon_type": "image",
                    "icon": "file-id",
                    "icon_background": "#123456",
                    "use_icon_as_answer_icon": False,
                    "max_active_requests": 0,
                },
                session=orm_session,
            )

        assert updated_app.name == "Iris"
        assert backing_agent.name == "Iris"
        assert backing_agent.description == "agent app"
        assert backing_agent.role == "research assistant"
        assert backing_agent.icon_type == AgentIconType.IMAGE
        assert backing_agent.icon == "file-id"
        assert backing_agent.icon_background == "#123456"
        assert backing_agent.updated_by == "account-2"
        assert backing_agent.updated_at == updated_app.updated_at

    def test_update_agent_app_preserves_role_when_args_omit_it(self, orm_session: Session):
        app, backing_agent = self._persist_agent_app(orm_session)

        with patch("services.app_service.current_user", SimpleNamespace(id="account-2")):
            AppService().update_app(
                app,
                {
                    "name": "Iris",
                    "description": "agent app",
                    "icon_type": "image",
                    "icon": "file-id",
                    "icon_background": "#123456",
                    "use_icon_as_answer_icon": False,
                    "max_active_requests": 0,
                },
                session=orm_session,
            )

        assert backing_agent.role == "research assistant"

    def test_update_agent_app_clears_role_when_args_set_empty_string(self, orm_session: Session):
        app, backing_agent = self._persist_agent_app(orm_session)

        with patch("services.app_service.current_user", SimpleNamespace(id="account-2")):
            AppService().update_app(
                app,
                {
                    "name": "Iris",
                    "description": "agent app",
                    "role": "",
                    "icon_type": "image",
                    "icon": "file-id",
                    "icon_background": "#123456",
                    "use_icon_as_answer_icon": False,
                    "max_active_requests": 0,
                },
                session=orm_session,
            )

        assert backing_agent.role == ""

    def test_update_agent_app_duplicate_name_rolls_back_and_raises_conflict(self, orm_session: Session):
        app, backing_agent = self._persist_agent_app(orm_session)
        orm_session.add(_agent(app_id="app-2", name="Existing Agent"))
        orm_session.commit()

        with patch("services.app_service.current_user", SimpleNamespace(id="account-2")):
            with pytest.raises(AgentNameConflictError):
                AppService().update_app(
                    app,
                    {
                        "name": "Existing Agent",
                        "description": "agent app",
                        "role": "research assistant",
                        "icon_type": "emoji",
                        "icon": "robot",
                        "icon_background": "#fff",
                        "use_icon_as_answer_icon": False,
                        "max_active_requests": 0,
                    },
                    session=orm_session,
                )

        assert orm_session.scalar(select(Agent.name).where(Agent.id == backing_agent.id)) == "Old"
        assert orm_session.get(App, app.id).name == "Old"

    def test_delete_agent_app_archives_backing_agent(self, orm_session: Session):
        app, backing_agent = self._persist_agent_app(orm_session)

        with (
            patch("services.app_service.current_user", SimpleNamespace(id="account-2")),
            patch("services.app_service.BillingService"),
            patch("services.app_service.EnterpriseService"),
            patch("services.app_service.FeatureService"),
            patch("services.app_service.dify_config"),
            patch("services.app_service.remove_app_and_related_data_task"),
        ):
            AppService().delete_app(app, session=orm_session)

        assert orm_session.get(App, app.id) is None
        persisted_agent = orm_session.get(Agent, backing_agent.id)
        assert persisted_agent is not None
        assert persisted_agent.status == AgentStatus.ARCHIVED
        assert persisted_agent.archived_by == "account-2"
        assert persisted_agent.archived_at is not None

"""Unit tests for database-backed Agent App agent/snapshot resolution."""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

import core.app.apps.agent_app.app_generator as gen_mod
from core.app.apps.agent_app.app_generator import AgentAppGenerator, AgentAppGeneratorError, AgentAppNotPublishedError
from core.app.entities.app_invoke_entities import InvokeFrom
from models.account import Account
from models.agent import Agent, AgentConfigDraft, AgentConfigSnapshot, AgentScope, AgentSource, AgentStatus
from models.agent_config_entities import AgentSoulConfig
from models.model import App, AppMode, IconType

_SOUL_DICT = {
    "model": {
        "plugin_id": "langgenius/openai",
        "model_provider": "langgenius/openai/openai",
        "model": "gpt-4o-mini",
    },
    "prompt": {"system_prompt": "You are Iris."},
}
AGENT_MODELS = (Agent, AgentConfigSnapshot, AgentConfigDraft)


class _DatabaseBinding:
    """Expose the real SQLite session to generator code using ``db.session``."""

    session: Session

    def __init__(self, session: Session) -> None:
        self.session = session


def _agent(*, tenant_id: str, app_id: str | None = None, snapshot_id: str | None = None) -> Agent:
    return Agent(
        tenant_id=tenant_id,
        name=f"Agent {uuid4()}",
        description="",
        role="",
        icon_type=None,
        icon=None,
        icon_background=None,
        scope=AgentScope.ROSTER,
        source=AgentSource.AGENT_APP,
        app_id=app_id,
        backing_app_id=None,
        workflow_id=None,
        workflow_node_id=None,
        active_config_snapshot_id=snapshot_id,
        active_config_has_model=snapshot_id is not None,
        active_config_is_published=True,
        status=AgentStatus.ACTIVE,
        created_by=None,
        updated_by=None,
        archived_by=None,
        archived_at=None,
    )


def _snapshot(*, tenant_id: str, agent_id: str, version: int = 1) -> AgentConfigSnapshot:
    return AgentConfigSnapshot(
        tenant_id=tenant_id,
        agent_id=agent_id,
        version=version,
        config_snapshot=AgentSoulConfig.model_validate(_SOUL_DICT),
        summary=None,
        version_note=None,
        created_by=None,
    )


def _app(*, tenant_id: str, app_id: str) -> App:
    app = App(
        tenant_id=tenant_id,
        name="Agent App",
        description="",
        mode=AppMode.AGENT,
        icon_type=IconType.EMOJI,
        icon="🤖",
        icon_background="#fff",
        enable_site=True,
        enable_api=True,
        max_active_requests=0,
    )
    app.id = app_id
    return app


def _user() -> Account:
    user = Account(name="Tester", email=f"{uuid4()}@example.com")
    user.id = str(uuid4())
    return user


@pytest.mark.parametrize("sqlite_session", [AGENT_MODELS], indirect=True)
class TestResolveAgentById:
    def test_success_returns_agent_snapshot_soul(
        self, monkeypatch: pytest.MonkeyPatch, sqlite_session: Session
    ) -> None:
        tenant_id = str(uuid4())
        agent = _agent(tenant_id=tenant_id)
        sqlite_session.add(agent)
        sqlite_session.flush()
        snapshot = _snapshot(tenant_id=tenant_id, agent_id=agent.id)
        sqlite_session.add(snapshot)
        sqlite_session.commit()
        monkeypatch.setattr(gen_mod, "db", _DatabaseBinding(sqlite_session))

        resolved_agent, resolved_snapshot, soul = AgentAppGenerator._resolve_agent_by_id(
            tenant_id=tenant_id, agent_id=agent.id, snapshot_id=snapshot.id
        )

        assert resolved_agent is agent
        assert resolved_snapshot is snapshot
        assert soul.prompt.system_prompt == "You are Iris."
        assert soul.model is not None
        assert soul.model.model == "gpt-4o-mini"

    def test_agent_missing_raises(self, monkeypatch: pytest.MonkeyPatch, sqlite_session: Session) -> None:
        monkeypatch.setattr(gen_mod, "db", _DatabaseBinding(sqlite_session))

        with pytest.raises(AgentAppGeneratorError, match="Agent not found"):
            AgentAppGenerator._resolve_agent_by_id(
                tenant_id=str(uuid4()), agent_id=str(uuid4()), snapshot_id=str(uuid4())
            )

    def test_no_published_version_raises(self, monkeypatch: pytest.MonkeyPatch, sqlite_session: Session) -> None:
        tenant_id = str(uuid4())
        agent = _agent(tenant_id=tenant_id)
        sqlite_session.add(agent)
        sqlite_session.commit()
        monkeypatch.setattr(gen_mod, "db", _DatabaseBinding(sqlite_session))

        with pytest.raises(AgentAppGeneratorError, match="no published version"):
            AgentAppGenerator._resolve_agent_by_id(tenant_id=tenant_id, agent_id=agent.id, snapshot_id=None)

    def test_snapshot_missing_raises(self, monkeypatch: pytest.MonkeyPatch, sqlite_session: Session) -> None:
        tenant_id = str(uuid4())
        agent = _agent(tenant_id=tenant_id)
        sqlite_session.add(agent)
        sqlite_session.commit()
        monkeypatch.setattr(gen_mod, "db", _DatabaseBinding(sqlite_session))

        with pytest.raises(AgentAppGeneratorError, match="published version not found"):
            AgentAppGenerator._resolve_agent_by_id(tenant_id=tenant_id, agent_id=agent.id, snapshot_id=str(uuid4()))


@pytest.mark.parametrize("sqlite_session", [AGENT_MODELS], indirect=True)
class TestResolveAgent:
    @staticmethod
    def _persist_bound_agent(sqlite_session: Session, *, published: bool) -> tuple[App, Agent, AgentConfigSnapshot]:
        tenant_id = str(uuid4())
        app = _app(tenant_id=tenant_id, app_id=str(uuid4()))
        agent = _agent(tenant_id=tenant_id, app_id=app.id)
        sqlite_session.add(agent)
        sqlite_session.flush()
        snapshot = _snapshot(tenant_id=tenant_id, agent_id=agent.id)
        sqlite_session.add(snapshot)
        sqlite_session.flush()
        agent.active_config_snapshot_id = snapshot.id
        agent.active_config_has_model = True
        agent.active_config_is_published = published
        sqlite_session.commit()
        return app, agent, snapshot

    def test_success_chains_to_resolve_by_id(self, monkeypatch: pytest.MonkeyPatch, sqlite_session: Session) -> None:
        app, bound_agent, snapshot = self._persist_bound_agent(sqlite_session, published=True)
        monkeypatch.setattr(gen_mod, "db", _DatabaseBinding(sqlite_session))

        agent, config_id, config_version_kind, soul = AgentAppGenerator()._resolve_agent(
            app,
            invoke_from=InvokeFrom.WEB_APP,
            draft_type=None,
            user=_user(),
        )

        assert agent is bound_agent
        assert config_id == snapshot.id
        assert config_version_kind == "snapshot"
        assert soul.model is not None

    def test_unpublished_draft_still_resolves_active_snapshot(
        self, monkeypatch: pytest.MonkeyPatch, sqlite_session: Session
    ) -> None:
        app, bound_agent, snapshot = self._persist_bound_agent(sqlite_session, published=False)
        monkeypatch.setattr(gen_mod, "db", _DatabaseBinding(sqlite_session))

        agent, config_id, config_version_kind, soul = AgentAppGenerator()._resolve_agent(
            app,
            invoke_from=InvokeFrom.WEB_APP,
            draft_type=None,
            user=_user(),
        )

        assert agent is bound_agent
        assert config_id == snapshot.id
        assert config_version_kind == "snapshot"
        assert soul.prompt.system_prompt == "You are Iris."

    def test_agent_without_active_snapshot_raises_before_model_resolution(
        self, monkeypatch: pytest.MonkeyPatch, sqlite_session: Session
    ) -> None:
        tenant_id = str(uuid4())
        app = _app(tenant_id=tenant_id, app_id=str(uuid4()))
        sqlite_session.add(_agent(tenant_id=tenant_id, app_id=app.id))
        sqlite_session.commit()
        monkeypatch.setattr(gen_mod, "db", _DatabaseBinding(sqlite_session))

        with pytest.raises(AgentAppNotPublishedError, match="not been published"):
            AgentAppGenerator()._resolve_agent(
                app,
                invoke_from=InvokeFrom.WEB_APP,
                draft_type=None,
                user=_user(),
            )

    def test_unbound_app_raises(self, monkeypatch: pytest.MonkeyPatch, sqlite_session: Session) -> None:
        tenant_id = str(uuid4())
        app = _app(tenant_id=tenant_id, app_id=str(uuid4()))
        sqlite_session.add(_agent(tenant_id=tenant_id, app_id=str(uuid4())))
        sqlite_session.commit()
        monkeypatch.setattr(gen_mod, "db", _DatabaseBinding(sqlite_session))

        with pytest.raises(AgentAppGeneratorError, match="has no bound Agent"):
            AgentAppGenerator()._resolve_agent(
                app,
                invoke_from=InvokeFrom.WEB_APP,
                draft_type=None,
                user=_user(),
            )

from types import SimpleNamespace
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid5

import pytest
from sqlalchemy.orm import Session

from controllers.common.agent_app_parameters import get_published_agent_app_feature_dict_and_user_input_form
from core.app.app_config.common.parameters_mapping import get_parameters_from_feature_dict
from core.app.apps.agent_app.errors import AgentAppGeneratorError, AgentAppNotPublishedError
from models.agent import Agent, AgentConfigSnapshot, AgentScope, AgentSource, AgentStatus


def _stable_uuid(value: str) -> str:
    return str(uuid5(NAMESPACE_URL, value))


def _persist_agent(
    session: Session,
    *,
    tenant_id: str,
    agent_id: str,
    active_config_snapshot_id: str | None,
    active_config_is_published: bool,
) -> Agent:
    agent = Agent(
        id=agent_id,
        tenant_id=tenant_id,
        name="Agent",
        scope=AgentScope.ROSTER,
        source=AgentSource.AGENT_APP,
        status=AgentStatus.ACTIVE,
        active_config_snapshot_id=active_config_snapshot_id,
        active_config_is_published=active_config_is_published,
    )
    session.add(agent)
    session.commit()
    return agent


def _persist_snapshot(
    session: Session,
    *,
    snapshot_id: str,
    tenant_id: str,
    agent_id: str,
    config_snapshot: dict,
) -> AgentConfigSnapshot:
    snapshot = AgentConfigSnapshot(
        id=snapshot_id,
        tenant_id=tenant_id,
        agent_id=agent_id,
        version=1,
        config_snapshot=config_snapshot,
    )
    session.add(snapshot)
    session.commit()
    return snapshot


@pytest.mark.parametrize("sqlite_session", [(Agent, AgentConfigSnapshot)], indirect=True)
def test_published_agent_app_parameters_use_soul_file_upload(sqlite_session: Session):
    tenant_id = _stable_uuid("tenant:one")
    agent_id = _stable_uuid("agent:one")
    snapshot_id = _stable_uuid("snapshot:one")
    app_model_config = SimpleNamespace(
        to_dict=lambda: {
            "opening_statement": "Hi from legacy presentation config",
            "file_upload": {
                "enabled": False,
                "image": {"enabled": False},
            },
        }
    )
    app_model = SimpleNamespace(
        tenant_id=tenant_id,
        bound_agent_id=agent_id,
        app_model_config=app_model_config,
    )
    _persist_agent(
        sqlite_session,
        tenant_id=tenant_id,
        agent_id=agent_id,
        active_config_snapshot_id=snapshot_id,
        active_config_is_published=True,
    )
    _persist_snapshot(
        sqlite_session,
        snapshot_id=snapshot_id,
        tenant_id=tenant_id,
        agent_id=agent_id,
        config_snapshot={
            "app_features": {
                "file_upload": {
                    "enabled": True,
                    "allowed_file_extensions": ["PNG"],
                    "allowed_file_types": ["image"],
                    "allowed_file_upload_methods": ["local_file"],
                    "image": {"enabled": True},
                    "number_limits": 2,
                }
            },
            "app_variables": [{"name": "topic", "type": "string", "required": True}],
        },
    )

    with patch("controllers.common.agent_app_parameters.db.session", sqlite_session):
        features_dict, user_input_form = get_published_agent_app_feature_dict_and_user_input_form(app_model)
    parameters = get_parameters_from_feature_dict(features_dict=features_dict, user_input_form=user_input_form)

    assert parameters["opening_statement"] == "Hi from legacy presentation config"
    assert parameters["file_upload"] == {
        "enabled": True,
        "allowed_file_extensions": ["PNG"],
        "allowed_file_types": ["image"],
        "allowed_file_upload_methods": ["local_file"],
        "image": {"enabled": True},
        "number_limits": 2,
    }
    assert parameters["user_input_form"] == [{"text-input": {"label": "topic", "variable": "topic", "required": True}}]


def test_published_agent_app_parameters_requires_bound_agent():
    app_model = SimpleNamespace(
        tenant_id="tenant-1",
        bound_agent_id=None,
        app_model_config=None,
    )

    with pytest.raises(AgentAppGeneratorError, match="no bound Agent"):
        get_published_agent_app_feature_dict_and_user_input_form(app_model)


@pytest.mark.parametrize("sqlite_session", [(Agent, AgentConfigSnapshot)], indirect=True)
def test_published_agent_app_parameters_requires_existing_active_agent(sqlite_session: Session):
    requested_tenant_id = _stable_uuid("tenant:requested")
    agent_id = _stable_uuid("agent:cross-tenant")
    app_model = SimpleNamespace(
        tenant_id=requested_tenant_id,
        bound_agent_id=agent_id,
        app_model_config=None,
    )
    _persist_agent(
        sqlite_session,
        tenant_id=_stable_uuid("tenant:other"),
        agent_id=agent_id,
        active_config_snapshot_id=None,
        active_config_is_published=False,
    )

    with (
        patch("controllers.common.agent_app_parameters.db.session", sqlite_session),
        pytest.raises(AgentAppGeneratorError, match="no bound Agent"),
    ):
        get_published_agent_app_feature_dict_and_user_input_form(app_model)


@pytest.mark.parametrize(
    "active_config_is_published",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize("sqlite_session", [(Agent, AgentConfigSnapshot)], indirect=True)
def test_published_agent_app_parameters_requires_published_agent(
    active_config_is_published: bool, sqlite_session: Session
):
    tenant_id = _stable_uuid(f"tenant:published:{active_config_is_published}")
    agent_id = _stable_uuid(f"agent:published:{active_config_is_published}")
    app_model = SimpleNamespace(
        tenant_id=tenant_id,
        bound_agent_id=agent_id,
        app_model_config=None,
    )
    _persist_agent(
        sqlite_session,
        tenant_id=tenant_id,
        agent_id=agent_id,
        active_config_snapshot_id=None,
        active_config_is_published=active_config_is_published,
    )

    with (
        patch("controllers.common.agent_app_parameters.db.session", sqlite_session),
        pytest.raises(AgentAppNotPublishedError, match="not been published"),
    ):
        get_published_agent_app_feature_dict_and_user_input_form(app_model)


@pytest.mark.parametrize("sqlite_session", [(Agent, AgentConfigSnapshot)], indirect=True)
def test_published_agent_app_parameters_allows_unpublished_draft_with_active_snapshot(sqlite_session: Session):
    tenant_id = _stable_uuid("tenant:unpublished-draft")
    agent_id = _stable_uuid("agent:unpublished-draft")
    snapshot_id = _stable_uuid("snapshot:unpublished-draft")
    app_model = SimpleNamespace(
        tenant_id=tenant_id,
        bound_agent_id=agent_id,
        app_model_config=None,
    )
    _persist_agent(
        sqlite_session,
        tenant_id=tenant_id,
        agent_id=agent_id,
        active_config_snapshot_id=snapshot_id,
        active_config_is_published=False,
    )
    _persist_snapshot(
        sqlite_session,
        snapshot_id=snapshot_id,
        tenant_id=tenant_id,
        agent_id=agent_id,
        config_snapshot={},
    )

    with patch("controllers.common.agent_app_parameters.db.session", sqlite_session):
        features_dict, user_input_form = get_published_agent_app_feature_dict_and_user_input_form(app_model)

    assert features_dict["file_upload"]["enabled"] is True
    assert user_input_form == []


@pytest.mark.parametrize("sqlite_session", [(Agent, AgentConfigSnapshot)], indirect=True)
def test_published_agent_app_parameters_requires_published_snapshot(sqlite_session: Session):
    tenant_id = _stable_uuid("tenant:missing-snapshot")
    agent_id = _stable_uuid("agent:missing-snapshot")
    app_model = SimpleNamespace(
        tenant_id=tenant_id,
        bound_agent_id=agent_id,
        app_model_config=None,
    )
    _persist_agent(
        sqlite_session,
        tenant_id=tenant_id,
        agent_id=agent_id,
        active_config_snapshot_id=_stable_uuid("snapshot:missing"),
        active_config_is_published=True,
    )

    with (
        patch("controllers.common.agent_app_parameters.db.session", sqlite_session),
        pytest.raises(AgentAppGeneratorError, match="published version not found"),
    ):
        get_published_agent_app_feature_dict_and_user_input_form(app_model)


@pytest.mark.parametrize("sqlite_session", [(Agent, AgentConfigSnapshot)], indirect=True)
def test_published_agent_app_parameters_allows_missing_legacy_app_model_config(sqlite_session: Session):
    tenant_id = _stable_uuid("tenant:no-legacy-config")
    agent_id = _stable_uuid("agent:no-legacy-config")
    snapshot_id = _stable_uuid("snapshot:no-legacy-config")
    app_model = SimpleNamespace(
        tenant_id=tenant_id,
        bound_agent_id=agent_id,
        app_model_config=None,
    )
    _persist_agent(
        sqlite_session,
        tenant_id=tenant_id,
        agent_id=agent_id,
        active_config_snapshot_id=snapshot_id,
        active_config_is_published=True,
    )
    _persist_snapshot(
        sqlite_session,
        snapshot_id=snapshot_id,
        tenant_id=tenant_id,
        agent_id=agent_id,
        config_snapshot={},
    )

    with patch("controllers.common.agent_app_parameters.db.session", sqlite_session):
        features_dict, user_input_form = get_published_agent_app_feature_dict_and_user_input_form(app_model)

    assert features_dict["file_upload"] == {
        "allowed_file_extensions": ["JPG", "JPEG", "PNG", "GIF", "WEBP", "SVG"],
        "allowed_file_types": ["document", "image", "audio", "video"],
        "allowed_file_upload_methods": ["local_file", "remote_url"],
        "enabled": True,
        "image": {"enabled": True},
        "number_limits": 3,
    }
    assert user_input_form == []

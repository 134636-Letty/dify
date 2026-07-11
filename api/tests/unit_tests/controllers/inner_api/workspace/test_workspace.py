"""SQLite-backed tests for the inner API workspace endpoints.

Authentication decorators are covered separately; handler tests unwrap them
and keep tenant events and service orchestration as explicit boundaries.
"""

import inspect
from collections.abc import Iterator
from dataclasses import dataclass
from unittest.mock import patch
from uuid import uuid4

import pytest
from flask import Flask
from pydantic import ValidationError
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, scoped_session, sessionmaker

from controllers.inner_api.workspace import workspace as workspace_module
from controllers.inner_api.workspace.workspace import (
    EnterpriseWorkspace,
    EnterpriseWorkspaceNoOwnerEmail,
    WorkspaceCreatePayload,
    WorkspaceOwnerlessPayload,
)
from models.account import Account, Tenant, TenantAccountJoin, TenantAccountRole
from models.base import TypeBase


@dataclass(frozen=True)
class Database:
    """Typed binding matching Flask-SQLAlchemy's callable session interface."""

    engine: Engine
    session: scoped_session[Session]


@pytest.fixture
def database(sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[Database]:
    TypeBase.metadata.create_all(
        sqlite_engine,
        tables=[Account.__table__, Tenant.__table__, TenantAccountJoin.__table__],
    )
    session_registry = scoped_session(sessionmaker(bind=sqlite_engine, expire_on_commit=False))
    database = Database(engine=sqlite_engine, session=session_registry)
    monkeypatch.setattr(workspace_module, "db", database)
    try:
        yield database
    finally:
        session_registry.remove()


def _persist_account(session: Session, *, email: str = "owner@example.com") -> Account:
    account = Account(name="Workspace owner", email=email)
    account.id = str(uuid4())
    session.add(account)
    session.commit()
    return account


def _persist_tenant(session: Session, *, name: str, public_key: str | None = None) -> Tenant:
    tenant = Tenant(name=name, encrypt_public_key=public_key)
    session.add(tenant)
    session.commit()
    return tenant


def _create_tenant_service_side_effect(name: str, *, is_from_dashboard: bool, session: Session) -> Tenant:
    assert is_from_dashboard is True
    return _persist_tenant(session, name=name)


def _create_ownerless_tenant_side_effect(name: str, *, is_from_dashboard: bool, session: Session) -> Tenant:
    assert is_from_dashboard is True
    return _persist_tenant(session, name=name, public_key="pub-key")


def _create_member_service_side_effect(
    tenant: Tenant,
    account: Account,
    session: Session,
    *,
    role: str,
) -> TenantAccountJoin:
    membership = TenantAccountJoin(
        tenant_id=tenant.id,
        account_id=account.id,
        role=TenantAccountRole(role),
    )
    session.add(membership)
    session.commit()
    return membership


class TestWorkspaceCreatePayload:
    def test_valid_payload(self) -> None:
        payload = WorkspaceCreatePayload.model_validate({"name": "My Workspace", "owner_email": "owner@example.com"})
        assert payload.name == "My Workspace"
        assert payload.owner_email == "owner@example.com"

    def test_missing_name_fails_validation(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            WorkspaceCreatePayload.model_validate({"owner_email": "owner@example.com"})
        assert "name" in str(exc_info.value)

    def test_missing_owner_email_fails_validation(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            WorkspaceCreatePayload.model_validate({"name": "My Workspace"})
        assert "owner_email" in str(exc_info.value)


class TestWorkspaceOwnerlessPayload:
    def test_valid_payload(self) -> None:
        assert WorkspaceOwnerlessPayload.model_validate({"name": "My Workspace"}).name == "My Workspace"

    def test_missing_name_fails_validation(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            WorkspaceOwnerlessPayload.model_validate({})
        assert "name" in str(exc_info.value)


class TestEnterpriseWorkspace:
    @pytest.fixture
    def api_instance(self) -> EnterpriseWorkspace:
        return EnterpriseWorkspace()

    def test_has_post_method(self, api_instance: EnterpriseWorkspace) -> None:
        assert callable(api_instance.post)

    def test_post_creates_workspace_with_persisted_owner_membership(
        self,
        database: Database,
        api_instance: EnterpriseWorkspace,
        app: Flask,
    ) -> None:
        session = database.session()
        account = _persist_account(session)
        old_tenant = _persist_tenant(session, name="Existing Workspace")
        session.add(
            TenantAccountJoin(
                tenant_id=old_tenant.id,
                account_id=account.id,
                role=TenantAccountRole.NORMAL,
            )
        )
        session.commit()

        with (
            patch.object(workspace_module, "TenantService") as tenant_service,
            patch.object(workspace_module, "tenant_was_created") as tenant_created,
            patch.object(workspace_module, "inner_api_ns") as namespace,
            app.test_request_context(),
        ):
            tenant_service.create_tenant.side_effect = _create_tenant_service_side_effect
            tenant_service.create_tenant_member.side_effect = _create_member_service_side_effect
            namespace.payload = {"name": "My Workspace", "owner_email": account.email}

            result = inspect.unwrap(api_instance.post)(api_instance)

        new_tenant = session.scalar(select(Tenant).where(Tenant.name == "My Workspace"))
        assert new_tenant is not None
        memberships = session.scalars(select(TenantAccountJoin).where(TenantAccountJoin.account_id == account.id)).all()
        assert {(membership.tenant_id, membership.role) for membership in memberships} == {
            (old_tenant.id, TenantAccountRole.NORMAL),
            (new_tenant.id, TenantAccountRole.OWNER),
        }
        assert result["message"] == "enterprise workspace created."
        assert result["tenant"]["id"] == new_tenant.id
        tenant_created.send.assert_called_once_with(new_tenant)

    def test_post_returns_404_when_owner_is_absent(
        self,
        database: Database,
        api_instance: EnterpriseWorkspace,
        app: Flask,
    ) -> None:
        with (
            patch.object(workspace_module, "TenantService") as tenant_service,
            patch.object(workspace_module, "inner_api_ns") as namespace,
            app.test_request_context(),
        ):
            namespace.payload = {"name": "My Workspace", "owner_email": "missing@example.com"}

            result = inspect.unwrap(api_instance.post)(api_instance)

        assert result == ({"message": "owner account not found."}, 404)
        tenant_service.create_tenant.assert_not_called()
        assert database.session().scalar(select(func.count(Tenant.id))) == 0


class TestEnterpriseWorkspaceNoOwnerEmail:
    @pytest.fixture
    def api_instance(self) -> EnterpriseWorkspaceNoOwnerEmail:
        return EnterpriseWorkspaceNoOwnerEmail()

    def test_has_post_method(self, api_instance: EnterpriseWorkspaceNoOwnerEmail) -> None:
        assert callable(api_instance.post)

    def test_post_creates_persisted_ownerless_workspace(
        self,
        database: Database,
        api_instance: EnterpriseWorkspaceNoOwnerEmail,
        app: Flask,
    ) -> None:
        with (
            patch.object(workspace_module, "TenantService") as tenant_service,
            patch.object(workspace_module, "tenant_was_created") as tenant_created,
            patch.object(workspace_module, "inner_api_ns") as namespace,
            app.test_request_context(),
        ):
            tenant_service.create_tenant.side_effect = _create_ownerless_tenant_side_effect
            namespace.payload = {"name": "My Workspace"}

            result = inspect.unwrap(api_instance.post)(api_instance)

        session = database.session()
        tenant = session.scalar(select(Tenant).where(Tenant.name == "My Workspace"))
        assert tenant is not None
        assert session.scalar(select(func.count(TenantAccountJoin.id))) == 0
        assert result["message"] == "enterprise workspace created."
        assert result["tenant"]["id"] == tenant.id
        assert result["tenant"]["encrypt_public_key"] == "pub-key"
        assert result["tenant"]["custom_config"] == {}
        tenant_created.send.assert_called_once_with(tenant)

"""SQLite-backed tests for dataset controller resource wrappers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from controllers.console.datasets import wraps as wraps_module
from controllers.console.datasets.error import PipelineNotFoundError
from controllers.console.datasets.wraps import get_rag_pipeline
from models.account import Account, Tenant, TenantAccountJoin, TenantAccountRole
from models.base import TypeBase
from models.dataset import Pipeline


@dataclass(frozen=True)
class Database:
    """Explicit database binding used by the legacy wrapper session path."""

    engine: Engine
    session: Session


@dataclass(frozen=True)
class WorkspaceIdentity:
    """Persisted account and its current workspace membership."""

    account: Account
    tenant: Tenant


@pytest.fixture
def database(sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[Database]:
    TypeBase.metadata.create_all(
        sqlite_engine,
        tables=[Account.__table__, Tenant.__table__, TenantAccountJoin.__table__, Pipeline.__table__],
    )
    with Session(sqlite_engine, expire_on_commit=False) as session:
        database = Database(engine=sqlite_engine, session=session)
        monkeypatch.setattr(wraps_module, "db", database)
        yield database


@pytest.fixture
def workspace(database: Database, monkeypatch: pytest.MonkeyPatch) -> WorkspaceIdentity:
    account = Account(name="Dataset owner", email="owner@example.com")
    account.id = "account-1"
    tenant = Tenant(name="Workspace")
    tenant.id = "tenant-1"
    membership = TenantAccountJoin(
        tenant_id=tenant.id,
        account_id=account.id,
        role=TenantAccountRole.OWNER,
        current=True,
    )
    account._current_tenant = tenant
    database.session.add_all([account, tenant, membership])
    database.session.commit()
    monkeypatch.setattr(wraps_module, "current_account_with_tenant", lambda: (account, tenant.id))
    return WorkspaceIdentity(account=account, tenant=tenant)


def _persist_pipeline(
    session: Session,
    *,
    pipeline_id: str = "pipeline-1",
    tenant_id: str = "tenant-1",
) -> Pipeline:
    pipeline = Pipeline(
        tenant_id=tenant_id,
        name="Knowledge pipeline",
        description="Pipeline fixture",
        created_by="account-1",
    )
    pipeline.id = pipeline_id
    session.add(pipeline)
    session.commit()
    return pipeline


class TestGetRagPipeline:
    def test_missing_pipeline_id(self) -> None:
        @get_rag_pipeline
        def dummy_view(**kwargs):
            return "ok"

        with pytest.raises(ValueError, match="missing pipeline_id"):
            dummy_view()

    def test_pipeline_not_found(self, database: Database, workspace: WorkspaceIdentity) -> None:
        @get_rag_pipeline
        def dummy_view(**kwargs):
            return "ok"

        with pytest.raises(PipelineNotFoundError):
            dummy_view(pipeline_id="pipeline-1")

    def test_pipeline_from_another_tenant_is_not_visible(
        self,
        database: Database,
        workspace: WorkspaceIdentity,
    ) -> None:
        _persist_pipeline(database.session, tenant_id="tenant-2")

        @get_rag_pipeline
        def dummy_view(**kwargs):
            return "ok"

        with pytest.raises(PipelineNotFoundError):
            dummy_view(pipeline_id="pipeline-1")

    def test_pipeline_found_and_injected(self, database: Database, workspace: WorkspaceIdentity) -> None:
        pipeline = _persist_pipeline(database.session)

        @get_rag_pipeline
        def dummy_view(**kwargs):
            return kwargs["pipeline"]

        result = dummy_view(pipeline_id="pipeline-1")

        assert result is pipeline

    def test_pipeline_id_removed_from_kwargs(self, database: Database, workspace: WorkspaceIdentity) -> None:
        _persist_pipeline(database.session)

        @get_rag_pipeline
        def dummy_view(**kwargs):
            assert "pipeline_id" not in kwargs
            return "ok"

        assert dummy_view(pipeline_id="pipeline-1") == "ok"

    def test_pipeline_id_cast_to_string(self, database: Database, workspace: WorkspaceIdentity) -> None:
        pipeline = _persist_pipeline(database.session, pipeline_id="123")

        @get_rag_pipeline
        def dummy_view(**kwargs):
            return kwargs["pipeline"]

        assert dummy_view(pipeline_id=123) is pipeline

    def test_uses_explicit_request_session(self, database: Database, workspace: WorkspaceIdentity) -> None:
        pipeline = _persist_pipeline(database.session)

        @get_rag_pipeline
        def dummy_view(controller, session, **kwargs):
            assert session is database.session
            return kwargs["pipeline"]

        assert dummy_view(object(), database.session, pipeline_id="pipeline-1") is pipeline

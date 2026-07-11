"""Tests for workflow-run archive command database boundaries.

Planning deliberately creates a fresh session for every tenant prefix and for
every database retry.  SQLite-backed tests keep the query, filtering, counting,
and session lifecycle real while clocks, billing lookup, and command failures
remain narrow external-boundary substitutions.
"""

import datetime
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import click
import pytest
from sqlalchemy import Engine, event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, scoped_session, sessionmaker

from commands import retention
from graphon.enums import WorkflowExecutionStatus, WorkflowNodeExecutionStatus
from models.base import TypeBase
from models.enums import CreatorUserRole, WorkflowRunTriggeredFrom
from models.workflow import (
    WorkflowNodeExecutionModel,
    WorkflowNodeExecutionTriggeredFrom,
    WorkflowRun,
    WorkflowType,
)


def _db_disconnect_error() -> OperationalError:
    return OperationalError(
        "select 1",
        {},
        RuntimeError("server closed the connection unexpectedly"),
        connection_invalidated=True,
    )


@dataclass(frozen=True)
class ArchiveDatabase:
    """Creates archive candidates and their node executions in SQLite."""

    session_maker: sessionmaker[Session]
    end_before: datetime.datetime

    def add_run(
        self,
        tenant_id: str,
        *,
        created_at: datetime.datetime | None = None,
        status: WorkflowExecutionStatus = WorkflowExecutionStatus.SUCCEEDED,
        run_type: WorkflowType = WorkflowType.WORKFLOW,
    ) -> str:
        run_id = str(uuid4())
        with self.session_maker.begin() as session:
            session.add(
                WorkflowRun(
                    id=run_id,
                    tenant_id=tenant_id,
                    app_id=str(uuid4()),
                    workflow_id=str(uuid4()),
                    type=run_type,
                    triggered_from=WorkflowRunTriggeredFrom.APP_RUN,
                    version="1",
                    graph="{}",
                    inputs="{}",
                    status=status,
                    outputs="{}",
                    error=None,
                    elapsed_time=0,
                    total_tokens=0,
                    total_steps=1,
                    created_by_role=CreatorUserRole.ACCOUNT,
                    created_by=str(uuid4()),
                    created_at=created_at or self.end_before - datetime.timedelta(days=1),
                )
            )
        return run_id

    def add_node(self, run_id: str, tenant_id: str, *, index: int) -> None:
        with self.session_maker.begin() as session:
            session.add(
                WorkflowNodeExecutionModel(
                    id=str(uuid4()),
                    tenant_id=tenant_id,
                    app_id=str(uuid4()),
                    workflow_id=str(uuid4()),
                    triggered_from=WorkflowNodeExecutionTriggeredFrom.WORKFLOW_RUN,
                    workflow_run_id=run_id,
                    index=index,
                    predecessor_node_id=None,
                    node_execution_id=None,
                    node_id=f"node-{index}",
                    node_type="start",
                    title="Start",
                    inputs="{}",
                    process_data="{}",
                    outputs="{}",
                    status=WorkflowNodeExecutionStatus.SUCCEEDED,
                    error=None,
                    elapsed_time=0,
                    execution_metadata="{}",
                    created_by_role=CreatorUserRole.ACCOUNT,
                    created_by=str(uuid4()),
                )
            )


@pytest.fixture
def archive_db(sqlite_engine: Engine) -> ArchiveDatabase:
    """Create only the workflow tables used by archive planning."""

    TypeBase.metadata.create_all(
        sqlite_engine,
        tables=[WorkflowRun.__table__, WorkflowNodeExecutionModel.__table__],
    )
    return ArchiveDatabase(
        session_maker=sessionmaker(bind=sqlite_engine, expire_on_commit=False),
        end_before=datetime.datetime(2025, 4, 1, tzinfo=datetime.UTC),
    )


def _tenant_id(prefix: str, suffix: int) -> str:
    return f"{prefix}{suffix:07x}-0000-0000-0000-000000000000"


def test_resolve_archive_tenant_ids_from_plan_uses_fresh_real_sessions(
    archive_db: ArchiveDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    paid_a = _tenant_id("a", 1)
    free_a = _tenant_id("a", 2)
    paid_b = _tenant_id("b", 1)
    free_b = _tenant_id("b", 2)
    for tenant_id in (paid_a, free_a, paid_b, free_b):
        archive_db.add_run(tenant_id)

    # These decoys verify the real candidate query's time, status, and type filters.
    archive_db.add_run(
        _tenant_id("a", 3),
        created_at=archive_db.end_before + datetime.timedelta(seconds=1),
    )
    archive_db.add_run(_tenant_id("a", 4), status=WorkflowExecutionStatus.RUNNING)
    archive_db.add_run(_tenant_id("a", 5), run_type=WorkflowType.CHAT)

    opened_sessions: list[Session] = []

    def record_session(session: Session, _transaction: object, _connection: object) -> None:
        opened_sessions.append(session)

    event.listen(archive_db.session_maker.class_, "after_begin", record_session)
    monkeypatch.setattr(
        retention,
        "_filter_paid_workflow_archive_tenant_ids",
        lambda tenant_ids: ([paid_a, paid_b], sorted(set(tenant_ids) - {paid_a, paid_b})),
    )
    try:
        tenant_plan = retention._resolve_archive_tenant_ids_from_plan(
            session_maker=archive_db.session_maker,
            tenant_ids=None,
            tenant_prefixes=["a", "b"],
            start_from=None,
            end_before=archive_db.end_before,
        )
    finally:
        event.remove(archive_db.session_maker.class_, "after_begin", record_session)

    assert tenant_plan == {
        "archive_tenant_ids": [paid_a, paid_b],
        "paid_tenant_ids": [paid_a, paid_b],
        "unpaid_tenant_ids": [free_a, free_b],
    }
    assert len(opened_sessions) == 2
    assert opened_sessions[0] is not opened_sessions[1]


def test_safe_remove_scoped_session_recovers_from_real_closed_connection(
    sqlite_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    maker = sessionmaker(bind=sqlite_engine)
    registry = scoped_session(maker)
    registry().execute(text("select 1"))
    sqlite_engine.dispose()
    monkeypatch.setattr(retention, "db", SimpleNamespace(session=registry, engine=sqlite_engine))

    with caplog.at_level(logging.WARNING, logger="commands.retention"):
        retention._safe_remove_scoped_session("archive workflow run command")

    assert not registry.registry.has()
    assert any("Ignoring DB scoped-session cleanup error" in message for message in caplog.messages)


def test_archive_command_db_retry_retries_retryable_db_disconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = iter([_db_disconnect_error(), "ok"])
    sleep = Mock()
    monkeypatch.setattr("services.retention.workflow_run.db_retry.time.sleep", sleep)

    def operation() -> str:
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    assert retention._run_archive_command_db_retry("archive plan", operation) == "ok"
    sleep.assert_called_once_with(1.0)


def test_archive_plan_prefix_stats_retries_with_fresh_session_and_real_counts(
    archive_db: ArchiveDatabase, sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    tenant_id = _tenant_id("a", 1)
    run_ids = [archive_db.add_run(tenant_id) for _ in range(7)]
    for index in range(9):
        archive_db.add_node(run_ids[index % len(run_ids)], tenant_id, index=index)

    # Decoys outside the selected prefix and archive window must not affect counts.
    decoy_run_id = archive_db.add_run(_tenant_id("b", 1))
    archive_db.add_node(decoy_run_id, _tenant_id("b", 1), index=99)
    archive_db.add_run(
        tenant_id,
        created_at=archive_db.end_before + datetime.timedelta(seconds=1),
    )

    fail_next_query = True

    def disconnect_once(
        _connection: object,
        _cursor: object,
        _statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        nonlocal fail_next_query
        if fail_next_query:
            fail_next_query = False
            raise _db_disconnect_error()

    opened_sessions: list[Session] = []

    def record_session(session: Session, _transaction: object, _connection: object) -> None:
        opened_sessions.append(session)

    sleep = Mock()
    monkeypatch.setattr("services.retention.workflow_run.db_retry.time.sleep", sleep)
    event.listen(sqlite_engine, "before_cursor_execute", disconnect_once)
    event.listen(archive_db.session_maker.class_, "after_begin", record_session)
    try:
        stats = retention._get_archive_plan_prefix_stats(
            archive_db.session_maker,
            "a",
            start_from=None,
            end_before=archive_db.end_before,
        )
    finally:
        event.remove(sqlite_engine, "before_cursor_execute", disconnect_once)
        event.remove(archive_db.session_maker.class_, "after_begin", record_session)

    assert stats == {
        "tenant_ids": [tenant_id],
        "workflow_runs": 7,
        "workflow_node_executions": 9,
    }
    assert len(opened_sessions) == 2
    assert opened_sessions[0] is not opened_sessions[1]
    sleep.assert_called_once_with(1.0)


def test_archive_workflow_runs_raises_click_exception_when_tenant_plan_fails(
    sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = scoped_session(sessionmaker(bind=sqlite_engine))
    monkeypatch.setattr(retention, "db", SimpleNamespace(engine=sqlite_engine, session=registry))
    monkeypatch.setattr(
        retention,
        "_resolve_archive_tenant_ids_from_plan",
        Mock(side_effect=RuntimeError("tenant plan failed")),
    )

    with pytest.raises(click.ClickException, match="Failed to resolve workflow archive tenant plan"):
        retention.archive_workflow_runs.callback(
            tenant_ids="tenant-1",
            tenant_prefixes=None,
            before_days=90,
            from_days_ago=None,
            to_days_ago=None,
            start_from=None,
            end_before=None,
            batch_size=10000,
            workers=1,
            run_shard_index=None,
            run_shard_total=None,
            limit=None,
            dry_run=True,
            delete_after_archive=False,
        )

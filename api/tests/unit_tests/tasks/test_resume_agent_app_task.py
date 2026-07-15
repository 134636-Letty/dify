"""Unit tests for resuming an Agent App after human-input submission.

The task reads a runtime form, app, conversation, and user through the scoped
application session.  Account resolution also opens a model-owned session to
load tenant membership.  These tests bind both paths to SQLite and persist the
complete lookup graph; only the Agent App generator remains an external-boundary
mock.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterator
from dataclasses import dataclass
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker

from core.app.entities.app_invoke_entities import InvokeFrom
from core.workflow.nodes.human_input.enums import HumanInputFormKind, HumanInputFormStatus
from models.account import Account, Tenant, TenantAccountJoin, TenantAccountRole
from models.base import TypeBase
from models.enums import ConversationFromSource, EndUserType
from models.human_input import HumanInputForm
from models.model import App, AppMode, Conversation, EndUser
from tasks.app_generate import resume_agent_app_task as mod

MODULE = "tasks.app_generate.resume_agent_app_task"


@dataclass(frozen=True)
class _ScopedDatabaseBinding:
    session: scoped_session[Session]


@dataclass(frozen=True)
class _EngineDatabaseBinding:
    engine: Engine


@dataclass(frozen=True)
class ResumeDatabase:
    """Identifiers and real session handles for one resumable Agent App turn."""

    session_maker: sessionmaker[Session]
    registry: scoped_session[Session]
    tenant_id: str
    app_id: str
    conversation_id: str
    form_id: str
    account_id: str
    end_user_id: str

    def delete(self, model: type[object], object_id: str) -> None:
        with self.session_maker.begin() as session:
            table = model.__table__  # type: ignore[attr-defined]
            session.execute(table.delete().where(table.c.id == object_id))

    def use_end_user(self) -> None:
        with self.session_maker.begin() as session:
            conversation = session.get_one(Conversation, self.conversation_id)
            conversation.from_account_id = None
            conversation.from_end_user_id = self.end_user_id

    def remove_users(self) -> None:
        with self.session_maker.begin() as session:
            conversation = session.get_one(Conversation, self.conversation_id)
            conversation.from_account_id = None
            conversation.from_end_user_id = None


@pytest.fixture
def resume_db(sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[ResumeDatabase]:
    """Persist the form ownership graph and explicitly bind both ORM session paths."""

    tables = [
        Tenant.__table__,
        Account.__table__,
        TenantAccountJoin.__table__,
        App.__table__,
        Conversation.__table__,
        EndUser.__table__,
        HumanInputForm.__table__,
    ]
    TypeBase.metadata.create_all(sqlite_engine, tables=tables)
    maker = sessionmaker(bind=sqlite_engine, expire_on_commit=False)
    registry = scoped_session(maker)
    monkeypatch.setattr(mod, "db", _ScopedDatabaseBinding(session=registry))
    monkeypatch.setattr("models.account.db", _EngineDatabaseBinding(engine=sqlite_engine))

    tenant_id = str(uuid4())
    app_id = str(uuid4())
    conversation_id = str(uuid4())
    form_id = str(uuid4())
    account_id = str(uuid4())
    end_user_id = str(uuid4())
    with maker.begin() as session:
        tenant = Tenant(name="Agent tenant")
        tenant.id = tenant_id
        account = Account(name="Agent owner", email="owner@example.com")
        account.id = account_id
        app = App(
            id=app_id,
            tenant_id=tenant_id,
            name="Agent App",
            description="runtime test app",
            mode=AppMode.AGENT,
            icon_type=None,
            icon=None,
            icon_background=None,
            enable_site=True,
            enable_api=True,
            max_active_requests=None,
            created_by=account_id,
        )
        conversation = Conversation(
            id=conversation_id,
            app_id=app_id,
            mode=AppMode.AGENT,
            name="Agent conversation",
            status="normal",
            invoke_from=InvokeFrom.WEB_APP,
            from_source=ConversationFromSource.CONSOLE,
            from_account_id=account_id,
            from_end_user_id=None,
        )
        conversation._inputs = {}
        end_user = EndUser(
            id=end_user_id,
            tenant_id=tenant_id,
            app_id=app_id,
            type=EndUserType.BROWSER,
            name="Agent visitor",
            is_anonymous=False,
            session_id="browser-session",
        )
        form = HumanInputForm(
            id=form_id,
            tenant_id=tenant_id,
            app_id=app_id,
            workflow_run_id=None,
            conversation_id=conversation_id,
            form_kind=HumanInputFormKind.RUNTIME,
            node_id="ask-human",
            form_definition="{}",
            rendered_content="Please answer",
            status=HumanInputFormStatus.SUBMITTED,
            expiration_time=datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
        )
        session.add_all(
            [
                tenant,
                account,
                TenantAccountJoin(
                    tenant_id=tenant_id,
                    account_id=account_id,
                    current=True,
                    role=TenantAccountRole.OWNER,
                ),
                app,
                conversation,
                end_user,
                form,
            ]
        )

    database = ResumeDatabase(
        session_maker=maker,
        registry=registry,
        tenant_id=tenant_id,
        app_id=app_id,
        conversation_id=conversation_id,
        form_id=form_id,
        account_id=account_id,
        end_user_id=end_user_id,
    )
    try:
        yield database
    finally:
        registry.remove()


def _run(resume_db: ResumeDatabase, monkeypatch: pytest.MonkeyPatch) -> Mock:
    generator = Mock()
    monkeypatch.setattr(mod, "AgentAppGenerator", Mock(return_value=generator))
    mod.resume_agent_app_execution(conversation_id=resume_db.conversation_id, form_id=resume_db.form_id)
    return generator


def test_resume_account_user_loads_tenant_membership_and_runs(
    resume_db: ResumeDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = _run(resume_db, monkeypatch)

    kwargs = generator.resume_after_form_submission.call_args.kwargs
    assert kwargs["conversation_id"] == resume_db.conversation_id
    assert kwargs["app_model"].id == resume_db.app_id
    assert isinstance(kwargs["user"], Account)
    assert kwargs["user"].id == resume_db.account_id
    assert kwargs["user"].current_tenant_id == resume_db.tenant_id
    assert kwargs["user"].current_role == TenantAccountRole.OWNER
    assert kwargs["invoke_from"] == InvokeFrom.WEB_APP


def test_resume_end_user_path_uses_persisted_visitor(
    resume_db: ResumeDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume_db.use_end_user()

    kwargs = _run(resume_db, monkeypatch).resume_after_form_submission.call_args.kwargs

    assert isinstance(kwargs["user"], EndUser)
    assert kwargs["user"].id == resume_db.end_user_id


def test_resume_preserves_debugger_invoke_from(resume_db: ResumeDatabase, monkeypatch: pytest.MonkeyPatch) -> None:
    with resume_db.session_maker.begin() as session:
        session.get_one(Conversation, resume_db.conversation_id).invoke_from = InvokeFrom.DEBUGGER

    kwargs = _run(resume_db, monkeypatch).resume_after_form_submission.call_args.kwargs

    assert kwargs["invoke_from"] == InvokeFrom.DEBUGGER


@pytest.mark.parametrize("missing_model", [HumanInputForm, App, Conversation])
def test_resume_returns_when_required_runtime_row_is_missing(
    resume_db: ResumeDatabase,
    monkeypatch: pytest.MonkeyPatch,
    missing_model: type[object],
) -> None:
    ids = {
        HumanInputForm: resume_db.form_id,
        App: resume_db.app_id,
        Conversation: resume_db.conversation_id,
    }
    resume_db.delete(missing_model, ids[missing_model])

    generator = _run(resume_db, monkeypatch)

    generator.resume_after_form_submission.assert_not_called()


def test_resume_returns_on_persisted_conversation_mismatch(
    resume_db: ResumeDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    with resume_db.session_maker.begin() as session:
        session.get_one(HumanInputForm, resume_db.form_id).conversation_id = str(uuid4())

    generator = _run(resume_db, monkeypatch)

    generator.resume_after_form_submission.assert_not_called()


def test_resume_returns_when_conversation_has_no_user(
    resume_db: ResumeDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume_db.remove_users()

    generator = _run(resume_db, monkeypatch)

    generator.resume_after_form_submission.assert_not_called()


def test_resume_returns_when_referenced_account_was_deleted(
    resume_db: ResumeDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume_db.delete(Account, resume_db.account_id)

    generator = _run(resume_db, monkeypatch)

    generator.resume_after_form_submission.assert_not_called()


def test_generator_exception_rolls_back_attached_app_changes(
    resume_db: ResumeDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = Mock()

    def mutate_then_fail(*, app_model: App, **_kwargs: object) -> None:
        app_model.name = "uncommitted mutation"
        raise RuntimeError("generator failed")

    generator.resume_after_form_submission.side_effect = mutate_then_fail
    monkeypatch.setattr(mod, "AgentAppGenerator", Mock(return_value=generator))

    mod.resume_agent_app_execution(conversation_id=resume_db.conversation_id, form_id=resume_db.form_id)

    with resume_db.session_maker() as verification_session:
        assert verification_session.get_one(App, resume_db.app_id).name == "Agent App"

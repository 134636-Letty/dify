from __future__ import annotations

import uuid
from inspect import unwrap
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from flask import Flask
from sqlalchemy import Engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from werkzeug.exceptions import BadRequest, NotFound

from controllers.console.app import conversation as conversation_module
from models import Account, Conversation
from models.base import TypeBase
from models.enums import ConversationFromSource
from models.model import App, AppMode, IconType
from services.errors.conversation import ConversationNotExistsError


@pytest.fixture
def database_session(sqlite_engine: Engine):
    models = (Account, App, Conversation)
    tables = [model.metadata.tables[model.__tablename__] for model in models]
    TypeBase.metadata.create_all(sqlite_engine, tables=tables)
    database_session = scoped_session(sessionmaker(bind=sqlite_engine, expire_on_commit=False))
    try:
        with patch.object(conversation_module.db, "session", database_session):
            yield database_session()
    finally:
        database_session.remove()


def _persist_conversation_state(
    session: Session,
    *,
    mode: AppMode,
    include_conversation: bool = True,
) -> tuple[Account, App, Conversation | None]:
    tenant_id = str(uuid.uuid4())
    account = Account(name="Account", email=f"{uuid.uuid4()}@example.com", timezone="UTC")
    app_model = App(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        name="Conversation App",
        mode=mode,
        icon_type=IconType.EMOJI,
        icon="chat",
        icon_background="#FFFFFF",
        enable_site=False,
        enable_api=True,
    )
    conversation = None
    models: list[object] = [account, app_model]
    if include_conversation:
        conversation = Conversation(
            id=str(uuid.uuid4()),
            app_id=app_model.id,
            mode=mode,
            name="Conversation",
            _inputs={},
            from_source=ConversationFromSource.CONSOLE,
            from_account_id=account.id,
        )
        models.append(conversation)
    session.add_all(models)
    session.commit()
    return account, app_model, conversation


def test_completion_conversation_list_returns_paginated_result(
    app: Flask, monkeypatch: pytest.MonkeyPatch, database_session: Session
) -> None:
    api = conversation_module.CompletionConversationApi()
    method = unwrap(api.get)

    account, app_model, _ = _persist_conversation_state(database_session, mode=AppMode.COMPLETION)
    monkeypatch.setattr(conversation_module, "parse_time_range", lambda *_args, **_kwargs: (None, None))

    paginate_result = SimpleNamespace(page=1, per_page=20, total=0, has_next=False, items=[])
    monkeypatch.setattr(conversation_module, "paginate_query", lambda *_args, **_kwargs: paginate_result)

    with app.test_request_context("/console/api/apps/app-1/completion-conversations", method="GET"):
        response = method(api, account, app_model=app_model)

    assert response == {"page": 1, "limit": 20, "total": 0, "has_more": False, "data": []}


def test_completion_conversation_list_invalid_time_range(
    app: Flask, monkeypatch: pytest.MonkeyPatch, database_session: Session
) -> None:
    api = conversation_module.CompletionConversationApi()
    method = unwrap(api.get)

    account, app_model, _ = _persist_conversation_state(database_session, mode=AppMode.COMPLETION)
    monkeypatch.setattr(
        conversation_module,
        "parse_time_range",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad range")),
    )

    with app.test_request_context(
        "/console/api/apps/app-1/completion-conversations",
        method="GET",
        query_string={"start": "bad"},
    ):
        with pytest.raises(BadRequest):
            method(api, account, app_model=app_model)


def test_chat_conversation_list_advanced_chat_calls_paginate(
    app: Flask, monkeypatch: pytest.MonkeyPatch, database_session: Session
) -> None:
    api = conversation_module.ChatConversationApi()
    method = unwrap(api.get)

    account, app_model, _ = _persist_conversation_state(database_session, mode=AppMode.ADVANCED_CHAT)
    monkeypatch.setattr(conversation_module, "parse_time_range", lambda *_args, **_kwargs: (None, None))

    paginate_result = SimpleNamespace(page=1, per_page=20, total=0, has_next=False, items=[])
    monkeypatch.setattr(conversation_module, "paginate_query", lambda *_args, **_kwargs: paginate_result)

    with app.test_request_context("/console/api/apps/app-1/chat-conversations", method="GET"):
        response = method(api, account, app_model=app_model)

    assert response == {"page": 1, "limit": 20, "total": 0, "has_more": False, "data": []}


def test_get_conversation_updates_read_at(database_session: Session) -> None:
    account, app_model, conversation = _persist_conversation_state(
        database_session,
        mode=AppMode.COMPLETION,
    )
    assert conversation is not None
    original_updated_at = conversation.updated_at

    result = conversation_module._get_conversation(account, app_model, conversation.id)

    assert result is conversation
    assert result.read_at is not None
    assert result.read_account_id == account.id
    assert result.updated_at == original_updated_at


def test_get_conversation_missing_raises_not_found(database_session: Session) -> None:
    account, app_model, _ = _persist_conversation_state(
        database_session,
        mode=AppMode.COMPLETION,
        include_conversation=False,
    )

    with pytest.raises(NotFound):
        conversation_module._get_conversation(account, app_model, str(uuid.uuid4()))


def test_completion_conversation_delete_maps_not_found(
    monkeypatch: pytest.MonkeyPatch, database_session: Session
) -> None:
    api = conversation_module.CompletionConversationDetailApi()
    method = unwrap(api.delete)

    monkeypatch.setattr(
        conversation_module.ConversationService,
        "delete",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ConversationNotExistsError()),
    )
    account, app_model, conversation = _persist_conversation_state(
        database_session,
        mode=AppMode.COMPLETION,
    )
    assert conversation is not None

    with pytest.raises(NotFound):
        method(api, account, app_model=app_model, conversation_id=conversation.id)

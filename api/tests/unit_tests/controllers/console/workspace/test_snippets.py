from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from inspect import unwrap
from types import SimpleNamespace
from unittest.mock import ANY, Mock

import pytest
from flask import Flask
from sqlalchemy import Engine, event
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from werkzeug.exceptions import NotFound

from controllers.console.workspace import snippets as snippets_module
from models import snippet as snippet_model_module
from models.account import Account, TenantAccountRole
from models.base import TypeBase
from models.model import Tag, TagBinding
from models.snippet import CustomizedSnippet
from services.snippet_dsl_service import ImportStatus, SnippetImportInfo


@pytest.fixture(autouse=True)
def _patch_snippet_service_factory(monkeypatch):
    def factory():
        return snippets_module.SnippetService.__new__(snippets_module.SnippetService)

    monkeypatch.setattr(snippets_module, "_snippet_service", factory)


@dataclass(frozen=True)
class _SQLiteDb:
    engine: Engine
    session: scoped_session[Session]


@pytest.fixture
def snippet_db(sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[Session]:
    """Bind controller sessions and snippet model properties to isolated SQLite."""
    TypeBase.metadata.create_all(
        sqlite_engine,
        tables=[Account.__table__, CustomizedSnippet.__table__, Tag.__table__, TagBinding.__table__],
    )
    session_registry = scoped_session(sessionmaker(bind=sqlite_engine, expire_on_commit=False))
    sqlite_db = _SQLiteDb(engine=sqlite_engine, session=session_registry)
    monkeypatch.setattr(snippets_module, "db", sqlite_db)
    monkeypatch.setattr(snippet_model_module, "db", sqlite_db)
    try:
        yield session_registry()
    finally:
        session_registry.remove()


def _account(account_id: str = "account-1") -> Account:
    account = Account(name="Test User", email=f"{account_id}@example.com")
    account.id = account_id
    account.role = TenantAccountRole.EDITOR
    return account


def _snippet(**overrides) -> SimpleNamespace:
    data = {
        "id": "snippet-1",
        "tenant_id": "tenant-1",
        "name": "Snippet",
        "description": "Description",
        "type": snippets_module.SnippetType.NODE,
        "version": 1,
        "use_count": 0,
        "is_published": False,
        "icon_info": None,
        "graph_dict": {},
        "input_fields_list": [],
        "tags": [],
        "created_by": None,
        "author_name": None,
        "created_by_account": None,
        "created_at": datetime.fromtimestamp(1_704_067_200, UTC),
        "updated_by": None,
        "updated_by_account": None,
        "updated_at": datetime.fromtimestamp(1_704_153_600, UTC),
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _customized_snippet(**overrides) -> CustomizedSnippet:
    data = {
        "id": "snippet-1",
        "tenant_id": "tenant-1",
        "name": "Snippet",
        "description": "Description",
        "type": snippets_module.SnippetType.NODE.value,
        "version": 1,
        "use_count": 0,
        "is_published": False,
        "icon_info": None,
        "input_fields": "[]",
        "created_by": None,
        "updated_by": None,
        "created_at": datetime.fromtimestamp(1_704_067_200, UTC).replace(tzinfo=None),
        "updated_at": datetime.fromtimestamp(1_704_153_600, UTC).replace(tzinfo=None),
    }
    data.update(overrides)
    return CustomizedSnippet(**data)


def test_snippet_list_query_reads_repeated_values(app: Flask):
    tag_id = "11111111-1111-1111-1111-111111111111"
    other_tag_id = "22222222-2222-2222-2222-222222222222"

    with app.test_request_context(
        f"/workspaces/current/customized-snippets?tag_ids={tag_id}&tag_ids={other_tag_id}"
        "&creators=account-a&creators=account-b&keyword=search"
    ):
        query = snippets_module._snippet_list_query_from_request()

    assert query.tag_ids == [tag_id, other_tag_id]
    assert query.creators == ["account-a", "account-b"]
    assert query.keyword == "search"


def test_snippet_list_query_reads_creator_ids_alias(app: Flask):
    with app.test_request_context(
        "/workspaces/current/customized-snippets?creator_ids=account-a&creator_ids=account-b"
    ):
        query = snippets_module._snippet_list_query_from_request()

    assert query.creators == ["account-a", "account-b"]


def test_snippet_list_query_ignores_indexed_values(app: Flask):
    tag_id = "11111111-1111-1111-1111-111111111111"
    with app.test_request_context(f"/workspaces/current/customized-snippets?tag_ids[0]={tag_id}&creators[0]=account-a"):
        query = snippets_module._snippet_list_query_from_request()

    assert query.tag_ids is None
    assert query.creators is None


def test_list_snippets_returns_pagination(app: Flask, monkeypatch: pytest.MonkeyPatch):
    snippets = [_snippet()]
    tag_id = "11111111-1111-1111-1111-111111111111"
    get_snippets = Mock(return_value=(snippets, 1, False))
    monkeypatch.setattr(snippets_module.SnippetService, "get_snippets", get_snippets)

    api = snippets_module.CustomizedSnippetsApi()
    handler = unwrap(api.get)

    with app.test_request_context(
        f"/workspaces/current/customized-snippets?page=2&limit=10&tag_ids={tag_id}&creators=account-2"
    ):
        response, status_code = handler(api, "tenant-1")

    assert status_code == 200
    assert response == {
        "data": [
            {
                "id": "snippet-1",
                "name": "Snippet",
                "description": "Description",
                "type": snippets_module.SnippetType.NODE.value,
                "version": 1,
                "use_count": 0,
                "is_published": False,
                "icon_info": None,
                "tags": [],
                "created_by": None,
                "author_name": None,
                "created_at": 1_704_067_200,
                "updated_by": None,
                "updated_at": 1_704_153_600,
            }
        ],
        "page": 2,
        "limit": 10,
        "total": 1,
        "has_more": False,
    }
    get_snippets.assert_called_once_with(
        tenant_id="tenant-1",
        session=ANY,
        page=2,
        limit=10,
        keyword=None,
        is_published=None,
        creators=["account-2"],
        tag_ids=[tag_id],
    )


def test_create_snippet_defaults_unknown_type_and_returns_created(app: Flask, monkeypatch: pytest.MonkeyPatch):
    user = _account("account-1")
    snippet = _snippet()
    create_snippet = Mock(return_value=snippet)
    monkeypatch.setattr(snippets_module.SnippetService, "create_snippet", create_snippet)
    monkeypatch.setattr(
        snippets_module.CreateSnippetPayload,
        "model_validate",
        Mock(
            return_value=SimpleNamespace(
                name="Snippet",
                type="unknown",
                description="Description",
                graph=None,
                icon_info=None,
                input_fields=[],
            )
        ),
    )

    api = snippets_module.CustomizedSnippetsApi()
    handler = unwrap(api.post)

    with app.test_request_context(
        "/workspaces/current/customized-snippets",
        method="POST",
        json={"name": "Snippet", "type": "node", "description": "Description"},
    ):
        response, status_code = handler(api, "tenant-1", user)

    assert status_code == 201
    assert response["id"] == "snippet-1"
    assert response["type"] == snippets_module.SnippetType.NODE.value
    assert create_snippet.call_args.kwargs["snippet_type"] == snippets_module.SnippetType.NODE


def test_create_snippet_rejects_forbidden_nodes(app: Flask, monkeypatch: pytest.MonkeyPatch):
    user = _account("account-1")
    create_snippet = Mock()
    monkeypatch.setattr(snippets_module.SnippetService, "create_snippet", create_snippet)

    api = snippets_module.CustomizedSnippetsApi()
    handler = unwrap(api.post)

    with app.test_request_context(
        "/workspaces/current/customized-snippets",
        method="POST",
        json={
            "name": "snippet with invalid node",
            "type": "node",
            "graph": {
                "nodes": [
                    {"id": "knowledge-1", "data": {"type": "knowledge-retrieval"}},
                ],
                "edges": [],
            },
        },
    ):
        response, status_code = handler(api, "tenant-1", user)

    assert status_code == 400
    assert "knowledge-retrieval" in response["message"]
    create_snippet.assert_not_called()


def test_get_snippet_detail_raises_when_missing(app: Flask, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(snippets_module.SnippetService, "get_snippet_by_id", Mock(return_value=None))

    api = snippets_module.CustomizedSnippetDetailApi()
    handler = unwrap(api.get)

    with app.test_request_context("/workspaces/current/customized-snippets/snippet-1"):
        with pytest.raises(NotFound, match="Snippet not found"):
            handler(api, "tenant-1", snippet_id="snippet-1")


def test_get_snippet_detail_returns_snippet(app: Flask, monkeypatch: pytest.MonkeyPatch):
    snippet = _snippet()
    monkeypatch.setattr(snippets_module.SnippetService, "get_snippet_by_id", Mock(return_value=snippet))

    api = snippets_module.CustomizedSnippetDetailApi()
    handler = unwrap(api.get)

    with app.test_request_context("/workspaces/current/customized-snippets/snippet-1"):
        response, status_code = handler(api, "tenant-1", snippet_id="snippet-1")

    assert status_code == 200
    assert response["id"] == "snippet-1"
    assert response["name"] == "Snippet"


def test_patch_snippet_returns_400_for_empty_payload(app: Flask, monkeypatch: pytest.MonkeyPatch):
    snippet = _snippet()
    user = _account("user-1")
    monkeypatch.setattr(snippets_module.SnippetService, "get_snippet_by_id", Mock(return_value=snippet))

    api = snippets_module.CustomizedSnippetDetailApi()
    handler = unwrap(api.patch)

    with app.test_request_context(
        "/workspaces/current/customized-snippets/snippet-1",
        method="PATCH",
        json={},
    ):
        response, status_code = handler(api, "tenant-1", user, snippet_id="snippet-1")

    assert status_code == 400
    assert response == {"message": "No valid fields to update"}


def test_patch_snippet_updates_and_commits(
    app: Flask, monkeypatch: pytest.MonkeyPatch, snippet_db: Session, sqlite_engine: Engine
):
    user = _account("account-1")
    snippet = _customized_snippet()
    snippet_db.add_all([user, snippet])
    snippet_db.commit()
    snippet_db.expunge(snippet)
    commits: list[Session] = []
    service_sessions: list[Session] = []

    def update_snippet(*, session: Session, snippet: CustomizedSnippet, account_id: str, data: dict):
        service_sessions.append(session)
        event.listen(session, "after_commit", commits.append)
        snippet.name = data["name"]
        snippet.icon_info = data["icon_info"]
        snippet.updated_by = account_id
        snippet.updated_at = datetime.now(UTC).replace(tzinfo=None)
        return snippet

    monkeypatch.setattr(snippets_module.SnippetService, "get_snippet_by_id", Mock(return_value=snippet))
    monkeypatch.setattr(snippets_module.SnippetService, "update_snippet", Mock(side_effect=update_snippet))

    api = snippets_module.CustomizedSnippetDetailApi()
    handler = unwrap(api.patch)

    with app.test_request_context(
        "/workspaces/current/customized-snippets/snippet-1",
        method="PATCH",
        json={"name": "New", "icon_info": {"icon": "star"}},
    ):
        response, status_code = handler(api, "tenant-1", user, snippet_id="snippet-1")

    assert status_code == 200
    assert response["id"] == "snippet-1"
    assert response["name"] == "New"
    update_mock = snippets_module.SnippetService.update_snippet
    update_mock.assert_called_once()
    assert update_mock.call_args.kwargs["data"] == {
        "name": "New",
        "icon_info": {"icon": "star", "icon_background": None, "icon_type": None, "icon_url": None},
    }
    assert len(service_sessions) == 1
    assert commits == service_sessions
    assert isinstance(service_sessions[0], Session)
    with Session(sqlite_engine) as verification_session:
        persisted = verification_session.get(CustomizedSnippet, "snippet-1")
    assert persisted is not None
    assert persisted.name == "New"
    assert persisted.updated_by == "account-1"


def test_delete_snippet_deletes_and_commits(
    app: Flask, monkeypatch: pytest.MonkeyPatch, snippet_db: Session, sqlite_engine: Engine
):
    snippet = _customized_snippet()
    snippet_db.add(snippet)
    snippet_db.commit()
    snippet_db.expunge(snippet)
    commits: list[Session] = []
    service_sessions: list[Session] = []

    def delete_snippet(*, session: Session, snippet: CustomizedSnippet) -> None:
        service_sessions.append(session)
        event.listen(session, "after_commit", commits.append)
        session.delete(snippet)

    monkeypatch.setattr(snippets_module.SnippetService, "get_snippet_by_id", Mock(return_value=snippet))
    delete_mock = Mock(side_effect=delete_snippet)
    monkeypatch.setattr(snippets_module.SnippetService, "delete_snippet", delete_mock)

    api = snippets_module.CustomizedSnippetDetailApi()
    handler = unwrap(api.delete)

    with app.test_request_context("/workspaces/current/customized-snippets/snippet-1", method="DELETE"):
        response, status_code = handler(api, "tenant-1", snippet_id="snippet-1")

    assert status_code == 204
    assert response == ""
    delete_mock.assert_called_once()
    assert isinstance(delete_mock.call_args.kwargs["session"], Session)
    assert delete_mock.call_args.kwargs["snippet"].id == "snippet-1"
    assert commits == service_sessions
    with Session(sqlite_engine) as verification_session:
        assert verification_session.get(CustomizedSnippet, "snippet-1") is None


def test_export_snippet_returns_yaml_attachment(app: Flask, monkeypatch: pytest.MonkeyPatch, snippet_db: Session):
    snippet = _customized_snippet(name="Snippet One")
    snippet_db.add(snippet)
    snippet_db.commit()
    snippet_db.expunge(snippet)
    export_snippet_dsl = Mock(return_value="version: 0.1.0\nkind: snippet\n")
    dsl_sessions: list[Session] = []

    def dsl_service(session: Session):
        dsl_sessions.append(session)
        return SimpleNamespace(export_snippet_dsl=export_snippet_dsl)

    monkeypatch.setattr(snippets_module.SnippetService, "get_snippet_by_id", Mock(return_value=snippet))
    monkeypatch.setattr(snippets_module, "SnippetDslService", Mock(side_effect=dsl_service))

    api = snippets_module.CustomizedSnippetExportApi()
    handler = unwrap(api.get)

    with app.test_request_context("/workspaces/current/customized-snippets/snippet-1/export?include_secret=true"):
        response = handler(api, "tenant-1", snippet_id="snippet-1")

    assert response.status_code == 200
    assert response.get_data(as_text=True) == "version: 0.1.0\nkind: snippet\n"
    assert response.headers["Content-Type"] == "application/x-yaml"
    assert "Snippet%20One.snippet" in response.headers["Content-Disposition"]
    export_snippet_dsl.assert_called_once_with(snippet=snippet, include_secret=True)
    assert len(dsl_sessions) == 1
    assert isinstance(dsl_sessions[0], Session)


def test_import_snippet_returns_202_for_pending_confirmation(
    app: Flask, monkeypatch: pytest.MonkeyPatch, snippet_db: Session
):
    user = _account("account-1")
    result = SnippetImportInfo(id="import-1", status=ImportStatus.PENDING, imported_dsl_version="999.0.0")
    import_snippet = Mock(return_value=result)
    commits: list[Session] = []
    dsl_sessions: list[Session] = []

    def dsl_service(session: Session):
        dsl_sessions.append(session)
        event.listen(session, "after_commit", commits.append)
        return SimpleNamespace(import_snippet=import_snippet)

    monkeypatch.setattr(snippets_module, "SnippetDslService", Mock(side_effect=dsl_service))

    api = snippets_module.CustomizedSnippetImportApi()
    handler = unwrap(api.post)

    with app.test_request_context(
        "/workspaces/current/customized-snippets/imports",
        method="POST",
        json={"mode": "yaml-content", "yaml_content": "kind: snippet"},
    ):
        response, status_code = handler(api, user)

    assert status_code == 202
    assert response["status"] == ImportStatus.PENDING.value
    import_snippet.assert_called_once()
    assert len(dsl_sessions) == 1
    assert isinstance(dsl_sessions[0], Session)
    assert commits == dsl_sessions


def test_import_snippet_returns_400_for_failed_import(app: Flask, monkeypatch: pytest.MonkeyPatch, snippet_db: Session):
    user = _account("account-1")
    result = SnippetImportInfo(id="import-1", status=ImportStatus.FAILED, error="Invalid DSL")
    import_snippet = Mock(return_value=result)
    commits: list[Session] = []
    dsl_sessions: list[Session] = []

    def dsl_service(session: Session):
        dsl_sessions.append(session)
        event.listen(session, "after_commit", commits.append)
        return SimpleNamespace(import_snippet=import_snippet)

    monkeypatch.setattr(snippets_module, "SnippetDslService", Mock(side_effect=dsl_service))

    api = snippets_module.CustomizedSnippetImportApi()
    handler = unwrap(api.post)

    with app.test_request_context(
        "/workspaces/current/customized-snippets/imports",
        method="POST",
        json={"mode": "yaml-content", "yaml_content": "kind: snippet"},
    ):
        response, status_code = handler(api, user)

    assert status_code == 400
    assert response["error"] == "Invalid DSL"
    assert len(dsl_sessions) == 1
    assert commits == dsl_sessions


def test_import_confirm_returns_200_for_completed_import(
    app: Flask, monkeypatch: pytest.MonkeyPatch, snippet_db: Session
):
    user = _account("account-1")
    result = SnippetImportInfo(id="import-1", status=ImportStatus.COMPLETED, snippet_id="snippet-1")
    confirm_import = Mock(return_value=result)
    commits: list[Session] = []
    dsl_sessions: list[Session] = []

    def dsl_service(session: Session):
        dsl_sessions.append(session)
        event.listen(session, "after_commit", commits.append)
        return SimpleNamespace(confirm_import=confirm_import)

    monkeypatch.setattr(snippets_module, "SnippetDslService", Mock(side_effect=dsl_service))

    api = snippets_module.CustomizedSnippetImportConfirmApi()
    handler = unwrap(api.post)

    with app.test_request_context(
        "/workspaces/current/customized-snippets/imports/import-1/confirm",
        method="POST",
    ):
        response, status_code = handler(api, user, import_id="import-1")

    assert status_code == 200
    assert response["snippet_id"] == "snippet-1"
    confirm_import.assert_called_once_with(import_id="import-1", account=user)
    assert len(dsl_sessions) == 1
    assert commits == dsl_sessions


def test_check_dependencies_raises_when_snippet_missing(app: Flask, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(snippets_module.SnippetService, "get_snippet_by_id", Mock(return_value=None))

    api = snippets_module.CustomizedSnippetCheckDependenciesApi()
    handler = unwrap(api.get)

    with app.test_request_context("/workspaces/current/customized-snippets/snippet-1/check-dependencies"):
        with pytest.raises(NotFound, match="Snippet not found"):
            handler(api, "tenant-1", snippet_id="snippet-1")


def test_check_dependencies_returns_dependency_result(app: Flask, monkeypatch: pytest.MonkeyPatch, snippet_db: Session):
    snippet = _customized_snippet()
    snippet_db.add(snippet)
    snippet_db.commit()
    snippet_db.expunge(snippet)
    check_dependencies = Mock(return_value=SimpleNamespace(model_dump=Mock(return_value={"leaked_dependencies": []})))
    dsl_sessions: list[Session] = []

    def dsl_service(session: Session):
        dsl_sessions.append(session)
        return SimpleNamespace(check_dependencies=check_dependencies)

    monkeypatch.setattr(snippets_module.SnippetService, "get_snippet_by_id", Mock(return_value=snippet))
    monkeypatch.setattr(snippets_module, "SnippetDslService", Mock(side_effect=dsl_service))

    api = snippets_module.CustomizedSnippetCheckDependenciesApi()
    handler = unwrap(api.get)

    with app.test_request_context("/workspaces/current/customized-snippets/snippet-1/check-dependencies"):
        response, status_code = handler(api, "tenant-1", snippet_id="snippet-1")

    assert status_code == 200
    assert response == {"leaked_dependencies": []}
    check_dependencies.assert_called_once_with(snippet=snippet)
    assert len(dsl_sessions) == 1
    assert isinstance(dsl_sessions[0], Session)


def test_increment_use_count_raises_when_snippet_missing(app: Flask, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(snippets_module.SnippetService, "get_snippet_by_id", Mock(return_value=None))

    api = snippets_module.CustomizedSnippetUseCountIncrementApi()
    handler = unwrap(api.post)

    with app.test_request_context(
        "/workspaces/current/customized-snippets/snippet-1/use-count/increment",
        method="POST",
    ):
        with pytest.raises(NotFound, match="Snippet not found"):
            handler(api, "tenant-1", snippet_id="snippet-1")


def test_increment_use_count_returns_refreshed_count(
    app: Flask, monkeypatch: pytest.MonkeyPatch, snippet_db: Session, sqlite_engine: Engine
):
    snippet = _customized_snippet(use_count=2)
    snippet_db.add(snippet)
    snippet_db.commit()
    snippet_db.expunge(snippet)
    commits: list[Session] = []
    service_sessions: list[Session] = []

    def increment_use_count(*, session: Session, snippet: CustomizedSnippet) -> None:
        service_sessions.append(session)
        event.listen(session, "after_commit", commits.append)
        snippet.use_count += 1

    increment_mock = Mock(side_effect=increment_use_count)
    monkeypatch.setattr(snippets_module.SnippetService, "get_snippet_by_id", Mock(return_value=snippet))
    monkeypatch.setattr(snippets_module.SnippetService, "increment_use_count", increment_mock)

    api = snippets_module.CustomizedSnippetUseCountIncrementApi()
    handler = unwrap(api.post)

    with app.test_request_context(
        "/workspaces/current/customized-snippets/snippet-1/use-count/increment",
        method="POST",
    ):
        response, status_code = handler(api, "tenant-1", snippet_id="snippet-1")

    assert status_code == 200
    assert response == {"result": "success", "use_count": 3}
    increment_mock.assert_called_once()
    assert len(service_sessions) == 1
    assert commits == service_sessions
    with Session(sqlite_engine) as verification_session:
        persisted = verification_session.get(CustomizedSnippet, "snippet-1")
    assert persisted is not None
    assert persisted.use_count == 3

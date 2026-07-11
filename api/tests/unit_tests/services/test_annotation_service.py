"""Persistence-focused tests for the application annotation service.

All ORM lookups and writes use an isolated SQLite session. Queue, vector-index,
Redis, billing, and authentication boundaries remain mocked.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import NotFound

import extensions.ext_database as ext_database_module
import models.model as model_module
from models.base import TypeBase
from models.dataset import DatasetCollectionBinding
from models.enums import ConversationFromSource
from models.model import App, AppAnnotationHitHistory, AppAnnotationSetting, AppMode, Message, MessageAnnotation
from services import annotation_service as annotation_service_module
from services.annotation_service import AppAnnotationService
from services.app_ref_service import AnnotationRef, AppRef


@dataclass(frozen=True)
class _Database:
    """Expose the real session through Flask-SQLAlchemy's used surface."""

    session: Session


@dataclass(frozen=True)
class _Auth:
    user_id: str
    tenant_id: str


@pytest.fixture
def database(sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Database]:
    """Create only annotation-service tables and bind their model helpers."""

    models = (
        App,
        Message,
        MessageAnnotation,
        AppAnnotationHitHistory,
        AppAnnotationSetting,
        DatasetCollectionBinding,
    )
    tables = [TypeBase.metadata.tables[model.__tablename__] for model in models]
    TypeBase.metadata.create_all(sqlite_engine, tables=tables)
    with Session(sqlite_engine, expire_on_commit=False) as session:
        database = _Database(session)
        monkeypatch.setattr(annotation_service_module, "db", database)
        monkeypatch.setattr(ext_database_module, "db", database)
        monkeypatch.setattr(model_module, "db", database)
        yield database


@pytest.fixture
def auth(monkeypatch: pytest.MonkeyPatch) -> _Auth:
    auth = _Auth(user_id=str(uuid4()), tenant_id=str(uuid4()))
    user = SimpleNamespace(id=auth.user_id)
    monkeypatch.setattr(annotation_service_module, "current_account_with_tenant", lambda: (user, auth.tenant_id))
    return auth


def _app(database: _Database, *, tenant_id: str, name: str = "Annotation app") -> App:
    app = App(
        id=str(uuid4()),
        tenant_id=tenant_id,
        name=name,
        description="",
        mode=AppMode.CHAT,
        icon_type=None,
        icon="",
        icon_background=None,
        enable_site=True,
        enable_api=True,
    )
    database.session.add(app)
    database.session.commit()
    return app


def _message(database: _Database, app: App, *, query: str = "original question") -> Message:
    message = Message(
        id=str(uuid4()),
        app_id=app.id,
        conversation_id=str(uuid4()),
        _inputs={},
        query=query,
        message={},
        message_unit_price=Decimal(0),
        answer="answer",
        answer_unit_price=Decimal(0),
        currency="USD",
        from_source=ConversationFromSource.API,
    )
    database.session.add(message)
    database.session.commit()
    return message


def _annotation(
    database: _Database,
    app: App,
    auth: _Auth,
    *,
    question: str = "question",
    content: str = "answer",
    message: Message | None = None,
) -> MessageAnnotation:
    annotation = MessageAnnotation(
        app_id=app.id,
        question=question,
        content=content,
        account_id=auth.user_id,
        conversation_id=message.conversation_id if message else None,
        message_id=message.id if message else None,
    )
    database.session.add(annotation)
    database.session.commit()
    return annotation


def _setting(database: _Database, app: App, auth: _Auth) -> tuple[AppAnnotationSetting, DatasetCollectionBinding]:
    binding = DatasetCollectionBinding(
        provider_name="provider-a",
        model_name="model-a",
        type="dataset",
        collection_name="annotations",
    )
    setting = AppAnnotationSetting(
        app_id=app.id,
        score_threshold=0.5,
        collection_binding_id=binding.id,
        created_user_id=auth.user_id,
        updated_user_id=auth.user_id,
    )
    database.session.add_all([binding, setting])
    database.session.commit()
    return setting, binding


def _history(database: _Database, app: App, annotation: MessageAnnotation, auth: _Auth) -> AppAnnotationHitHistory:
    history = AppAnnotationHitHistory(
        app_id=app.id,
        annotation_id=annotation.id,
        source="api",
        question="matched query",
        account_id=auth.user_id,
        score=0.9,
        message_id=str(uuid4()),
        annotation_question=annotation.question,
        annotation_content=annotation.content,
    )
    database.session.add(history)
    database.session.commit()
    return history


def _ref(app: App, annotation: MessageAnnotation) -> AnnotationRef:
    return AnnotationRef(tenant_id=app.tenant_id, app_id=app.id, annotation_id=annotation.id)


class TestUpsertAndDirectMutation:
    def test_upsert_from_message_creates_and_persists(self, database: _Database, auth: _Auth) -> None:
        app = _app(database, tenant_id=auth.tenant_id)
        message = _message(database, app)

        with patch("services.annotation_service.add_annotation_to_index_task") as index_task:
            result = AppAnnotationService.up_insert_app_annotation_from_message(
                {"answer": "created answer", "message_id": message.id}, app.id, session=database.session
            )

        persisted = database.session.get(MessageAnnotation, result.id)
        assert persisted is not None
        assert persisted.question == "original question"
        assert persisted.content == "created answer"
        assert persisted.message_id == message.id
        index_task.delay.assert_not_called()

    def test_upsert_updates_existing_annotation_and_indexes(self, database: _Database, auth: _Auth) -> None:
        app = _app(database, tenant_id=auth.tenant_id)
        message = _message(database, app)
        annotation = _annotation(database, app, auth, message=message)
        setting, _ = _setting(database, app, auth)

        with patch("services.annotation_service.add_annotation_to_index_task") as index_task:
            result = AppAnnotationService.up_insert_app_annotation_from_message(
                {"answer": "updated", "message_id": message.id}, app.id, session=database.session
            )

        database.session.refresh(annotation)
        assert result.id == annotation.id
        assert annotation.content == "updated"
        assert annotation.question == message.query
        index_task.delay.assert_called_once_with(
            annotation.id, message.query, auth.tenant_id, app.id, setting.collection_binding_id
        )

    @pytest.mark.parametrize("missing", ["app", "message"])
    def test_upsert_not_found_and_tenant_scope(self, database: _Database, auth: _Auth, missing: str) -> None:
        app = _app(database, tenant_id=str(uuid4()) if missing == "app" else auth.tenant_id)
        message_id = str(uuid4())
        if missing == "app":
            message_id = _message(database, app).id

        with pytest.raises(NotFound):
            AppAnnotationService.up_insert_app_annotation_from_message(
                {"answer": "answer", "message_id": message_id}, app.id, session=database.session
            )

    def test_direct_insert_and_update_persist(self, database: _Database, auth: _Auth) -> None:
        app = _app(database, tenant_id=auth.tenant_id)
        with (
            patch("services.annotation_service.add_annotation_to_index_task"),
            patch("services.annotation_service.update_annotation_to_index_task") as update_task,
        ):
            annotation = AppAnnotationService.insert_app_annotation_directly(
                {"question": "first", "answer": "one"}, app.id, session=database.session
            )
            updated = AppAnnotationService.update_app_annotation_directly(
                {"question": "second", "answer": "two"}, _ref(app, annotation), database.session
            )

        database.session.refresh(annotation)
        assert updated.id == annotation.id
        assert (annotation.question, annotation.content) == ("second", "two")
        update_task.delay.assert_not_called()

    def test_cross_app_update_is_not_found(self, database: _Database, auth: _Auth) -> None:
        app = _app(database, tenant_id=auth.tenant_id)
        other = _app(database, tenant_id=auth.tenant_id)
        annotation = _annotation(database, app, auth)
        cross_ref = AnnotationRef(tenant_id=auth.tenant_id, app_id=other.id, annotation_id=annotation.id)

        with pytest.raises(NotFound):
            AppAnnotationService.update_app_annotation_directly(
                {"question": "q", "answer": "a"}, cross_ref, database.session
            )

    def test_commit_failure_can_be_rolled_back(self, database: _Database, auth: _Auth) -> None:
        app = _app(database, tenant_id=auth.tenant_id)

        def fail_commit(_session: Session) -> None:
            raise IntegrityError("forced commit failure", {}, RuntimeError("constraint"))

        event.listen(database.session, "before_commit", fail_commit)
        try:
            with pytest.raises(IntegrityError):
                AppAnnotationService.insert_app_annotation_directly(
                    {"question": "not persisted", "answer": "answer"}, app.id, session=database.session
                )
        finally:
            event.remove(database.session, "before_commit", fail_commit)
            database.session.rollback()

        assert database.session.scalar(select(MessageAnnotation).where(MessageAnnotation.app_id == app.id)) is None


class TestListExportAndLookup:
    def test_list_filters_keyword_and_tenant(self, database: _Database, auth: _Auth) -> None:
        app = _app(database, tenant_id=auth.tenant_id)
        _annotation(database, app, auth, question="matching question")
        _annotation(database, app, auth, question="other")
        foreign_app = _app(database, tenant_id=str(uuid4()))
        _annotation(database, foreign_app, auth, question="matching foreign")

        items, total = AppAnnotationService.get_annotation_list_by_app_id(
            app.id, 1, 10, "matching", session=database.session
        )

        assert total == 1
        assert [item.question for item in items] == ["matching question"]
        with pytest.raises(NotFound):
            AppAnnotationService.get_annotation_list_by_app_id(foreign_app.id, 1, 10, "", session=database.session)

    def test_export_reads_real_rows_and_sanitizes_csv(self, database: _Database, auth: _Auth) -> None:
        app = _app(database, tenant_id=auth.tenant_id)
        annotation = _annotation(database, app, auth, question="=SUM(A1)", content="+formula")

        exported = AppAnnotationService.export_annotation_list_by_app_id(app.id, session=database.session)

        assert [item.id for item in exported] == [annotation.id]
        assert exported[0].question.startswith("'")
        assert exported[0].content.startswith("'")

    def test_get_by_id_returns_persisted_or_none(self, database: _Database, auth: _Auth) -> None:
        app = _app(database, tenant_id=auth.tenant_id)
        annotation = _annotation(database, app, auth)

        assert AppAnnotationService.get_annotation_by_id(annotation.id, session=database.session) is annotation
        assert AppAnnotationService.get_annotation_by_id(str(uuid4()), session=database.session) is None


class TestDeletion:
    def test_delete_removes_annotation_histories_and_index(self, database: _Database, auth: _Auth) -> None:
        app = _app(database, tenant_id=auth.tenant_id)
        annotation = _annotation(database, app, auth)
        history = _history(database, app, annotation, auth)
        setting, _ = _setting(database, app, auth)

        with patch("services.annotation_service.delete_annotation_index_task") as delete_task:
            AppAnnotationService.delete_app_annotation(_ref(app, annotation), database.session)

        assert database.session.get(MessageAnnotation, annotation.id) is None
        assert database.session.get(AppAnnotationHitHistory, history.id) is None
        delete_task.delay.assert_called_once_with(annotation.id, app.id, auth.tenant_id, setting.collection_binding_id)

    def test_batch_delete_counts_only_scoped_rows(self, database: _Database, auth: _Auth) -> None:
        app = _app(database, tenant_id=auth.tenant_id)
        other = _app(database, tenant_id=auth.tenant_id)
        first = _annotation(database, app, auth)
        second = _annotation(database, app, auth)
        foreign = _annotation(database, other, auth)

        result = AppAnnotationService.delete_app_annotations_in_batch(
            AppRef(tenant_id=auth.tenant_id, app_id=app.id),
            [first.id, second.id, foreign.id],
            session=database.session,
        )

        assert result == {"deleted_count": 2}
        assert database.session.get(MessageAnnotation, first.id) is None
        assert database.session.get(MessageAnnotation, foreign.id) is not None

    def test_clear_all_deletes_rows_and_histories(self, database: _Database, auth: _Auth) -> None:
        app = _app(database, tenant_id=auth.tenant_id)
        first = _annotation(database, app, auth)
        second = _annotation(database, app, auth)
        history = _history(database, app, first, auth)

        result = AppAnnotationService.clear_all_annotations(app.id, session=database.session)

        assert result == {"result": "success"}
        assert (
            database.session.scalar(
                select(func.count()).select_from(MessageAnnotation).where(MessageAnnotation.app_id == app.id)
            )
            == 0
        )
        assert database.session.get(AppAnnotationHitHistory, history.id) is None
        assert database.session.get(MessageAnnotation, second.id) is None


class TestHistoryAndSettings:
    def test_add_history_increments_count_and_persists_history(self, database: _Database, auth: _Auth) -> None:
        app = _app(database, tenant_id=auth.tenant_id)
        annotation = _annotation(database, app, auth)

        AppAnnotationService.add_annotation_history(
            annotation.id,
            app.id,
            annotation.question,
            annotation.content,
            "query",
            auth.user_id,
            str(uuid4()),
            "api",
            0.8,
            session=database.session,
        )

        database.session.refresh(annotation)
        assert annotation.hit_count == 1
        history = database.session.scalar(
            select(AppAnnotationHitHistory).where(AppAnnotationHitHistory.annotation_id == annotation.id)
        )
        assert history is not None
        assert history.score == 0.8

    def test_setting_get_and_update_use_persisted_binding(self, database: _Database, auth: _Auth) -> None:
        app = _app(database, tenant_id=auth.tenant_id)
        setting, _ = _setting(database, app, auth)

        current = AppAnnotationService.get_app_annotation_setting_by_app_id(app.id, session=database.session)
        updated = AppAnnotationService.update_app_annotation_setting(
            app.id, setting.id, {"score_threshold": 0.9}, session=database.session
        )

        assert current["enabled"] is True
        assert current["embedding_model"] == {
            "embedding_provider_name": "provider-a",
            "embedding_model_name": "model-a",
        }
        assert updated["score_threshold"] == 0.9
        database.session.refresh(setting)
        assert setting.score_threshold == 0.9

    def test_setting_disabled_and_missing_app(self, database: _Database, auth: _Auth) -> None:
        app = _app(database, tenant_id=auth.tenant_id)
        assert AppAnnotationService.get_app_annotation_setting_by_app_id(app.id, session=database.session) == {
            "enabled": False
        }

        foreign = _app(database, tenant_id=str(uuid4()))
        with pytest.raises(NotFound):
            AppAnnotationService.get_app_annotation_setting_by_app_id(foreign.id, session=database.session)


class TestBatchImport:
    def test_batch_import_dispatches_valid_rows(self, database: _Database, auth: _Auth) -> None:
        app = _app(database, tenant_id=auth.tenant_id)
        file = FileStorage(stream=BytesIO(b"question,answer\nq1,a1\nq2,a2\n"))
        features = SimpleNamespace(billing=SimpleNamespace(enabled=False))

        with (
            patch("services.annotation_service.FeatureService.get_features", return_value=features),
            patch("services.annotation_service.redis_client") as redis,
            patch("services.annotation_service.batch_import_annotations_task") as task,
        ):
            result = AppAnnotationService.batch_import_app_annotations(app.id, file, session=database.session)

        assert result["job_status"] == "waiting"
        assert result["record_count"] == 2
        payload = task.delay.call_args.args[1]
        assert payload == [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}]
        redis.zadd.assert_called_once()

    @pytest.mark.parametrize(
        "content",
        [b"", b"only-one-column\nvalue\n", b"question,answer\n,\n"],
    )
    def test_batch_import_rejects_invalid_csv(self, database: _Database, auth: _Auth, content: bytes) -> None:
        app = _app(database, tenant_id=auth.tenant_id)
        result = AppAnnotationService.batch_import_app_annotations(
            app.id, FileStorage(stream=BytesIO(content)), session=database.session
        )
        assert "error_msg" in result

    def test_batch_import_missing_or_cross_tenant_app(self, database: _Database, auth: _Auth) -> None:
        foreign = _app(database, tenant_id=str(uuid4()))
        with pytest.raises(NotFound):
            AppAnnotationService.batch_import_app_annotations(
                foreign.id,
                FileStorage(stream=BytesIO(b"question,answer\nq,a\n")),
                session=database.session,
            )

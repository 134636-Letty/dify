"""Unit tests for document indexing tasks.

The task deliberately uses several short-lived sessions: validation errors are
committed immediately, parsing state is committed before the indexing runner is
called, and summary eligibility is read again afterwards.  These tests bind the
task-owned session factory to SQLite so those transaction boundaries and fresh
reads are exercised rather than simulated by query-result doubles.
"""

import logging
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from sqlalchemy import event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from core.indexing_runner import DocumentIsPausedError
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from core.rag.pipeline.queue import TaskWrapper, TenantIsolatedTaskQueue
from enums.cloud_plan import CloudPlan
from extensions.ext_redis import redis_client
from models.base import TypeBase
from models.dataset import Dataset, Document
from models.enums import DataSourceType, DocumentCreatedFrom, IndexingStatus
from services.document_indexing_proxy.document_indexing_task_proxy import DocumentIndexingTaskProxy
from tasks.document_indexing_task import (
    _document_indexing,
    _document_indexing_with_tenant_queue,
    document_indexing_task,
    normal_document_indexing_task,
    priority_document_indexing_task,
)


@dataclass(frozen=True)
class IndexingDatabase:
    """Handles persisted task records while production code owns its sessions."""

    session_maker: sessionmaker[Session]
    tenant_id: str
    dataset_id: str
    document_ids: tuple[str, ...]
    decoy_dataset_id: str
    decoy_document_id: str

    def documents(self, ids: Sequence[str] | None = None) -> list[Document]:
        with self.session_maker() as session:
            stmt = select(Document).order_by(Document.position)
            if ids is not None:
                stmt = stmt.where(Document.id.in_(ids))
            return list(session.scalars(stmt).all())

    def dataset(self) -> Dataset | None:
        with self.session_maker() as session:
            return session.get(Dataset, self.dataset_id)


def _dataset(*, dataset_id: str, tenant_id: str, summary_enabled: bool = False) -> Dataset:
    return Dataset(
        id=dataset_id,
        tenant_id=tenant_id,
        name=f"dataset-{dataset_id}",
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique=IndexTechniqueType.HIGH_QUALITY,
        summary_index_setting={"enable": True} if summary_enabled else None,
        created_by=str(uuid4()),
    )


def _document(
    *,
    document_id: str,
    dataset_id: str,
    tenant_id: str,
    position: int,
    status: IndexingStatus = IndexingStatus.WAITING,
    doc_form: IndexStructureType = IndexStructureType.PARAGRAPH_INDEX,
    need_summary: bool = False,
) -> Document:
    return Document(
        id=document_id,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        position=position,
        data_source_type=DataSourceType.UPLOAD_FILE,
        batch="batch-1",
        name=f"document-{position}",
        created_from=DocumentCreatedFrom.WEB,
        created_by=str(uuid4()),
        indexing_status=status,
        doc_form=doc_form,
        need_summary=need_summary,
    )


@pytest.fixture
def indexing_db(sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> IndexingDatabase:
    """Persist a target dataset and cross-tenant decoys, then bind task sessions."""

    TypeBase.metadata.create_all(sqlite_engine, tables=[Dataset.__table__, Document.__table__])
    maker = sessionmaker(bind=sqlite_engine, expire_on_commit=False)
    monkeypatch.setattr("tasks.document_indexing_task.session_factory.create_session", maker)

    tenant_id = str(uuid4())
    dataset_id = str(uuid4())
    document_ids = tuple(str(uuid4()) for _ in range(3))
    decoy_tenant_id = str(uuid4())
    decoy_dataset_id = str(uuid4())
    decoy_document_id = str(uuid4())
    with maker.begin() as session:
        session.add(_dataset(dataset_id=dataset_id, tenant_id=tenant_id))
        session.add(_dataset(dataset_id=decoy_dataset_id, tenant_id=decoy_tenant_id))
        session.add_all(
            [
                _document(
                    document_id=document_id,
                    dataset_id=dataset_id,
                    tenant_id=tenant_id,
                    position=position,
                )
                for position, document_id in enumerate(document_ids)
            ]
        )
        session.add(
            _document(
                document_id=decoy_document_id,
                dataset_id=decoy_dataset_id,
                tenant_id=decoy_tenant_id,
                position=99,
            )
        )

    return IndexingDatabase(
        session_maker=maker,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_ids=document_ids,
        decoy_dataset_id=decoy_dataset_id,
        decoy_document_id=decoy_document_id,
    )


@pytest.fixture
def mock_redis() -> Mock:
    """Reset the external Redis boundary used by tenant-isolated queues."""

    redis_client.reset_mock()
    redis_client.get.return_value = None
    redis_client.setex.return_value = True
    redis_client.delete.return_value = True
    redis_client.lpush.return_value = 1
    redis_client.rpop.side_effect = None
    redis_client.rpop.return_value = None
    return redis_client


def _features(
    *,
    billing_enabled: bool = False,
    plan: CloudPlan = CloudPlan.PROFESSIONAL,
    vector_limit: int = 1_000,
    vector_size: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        billing=SimpleNamespace(enabled=billing_enabled, subscription=SimpleNamespace(plan=plan)),
        vector_space=SimpleNamespace(limit=vector_limit, size=vector_size),
    )


def _run_task(
    monkeypatch: pytest.MonkeyPatch,
    database: IndexingDatabase,
    *,
    document_ids: Sequence[str] | None = None,
    features: SimpleNamespace | None = None,
    runner: Mock | None = None,
) -> Mock:
    runner = runner or Mock()
    monkeypatch.setattr(
        "tasks.document_indexing_task.FeatureService.get_features", Mock(return_value=features or _features())
    )
    monkeypatch.setattr("tasks.document_indexing_task.IndexingRunner", Mock(return_value=runner))
    _document_indexing(database.dataset_id, document_ids if document_ids is not None else database.document_ids)
    return runner


class TestTaskEnqueuing:
    """Document proxies choose a direct task or an isolated tenant queue."""

    @pytest.mark.parametrize(
        ("billing_enabled", "plan", "task_attribute"),
        [
            (False, CloudPlan.PROFESSIONAL, "PRIORITY_TASK_FUNC"),
            (True, CloudPlan.SANDBOX, "NORMAL_TASK_FUNC"),
            (True, CloudPlan.PROFESSIONAL, "PRIORITY_TASK_FUNC"),
        ],
    )
    def test_proxy_dispatches_first_task(
        self,
        mock_redis: Mock,
        billing_enabled: bool,
        plan: CloudPlan,
        task_attribute: str,
    ) -> None:
        tenant_id, dataset_id = str(uuid4()), str(uuid4())
        document_ids = [str(uuid4())]
        task = Mock()
        with (
            patch.object(DocumentIndexingTaskProxy, "features", _features(billing_enabled=billing_enabled, plan=plan)),
            patch.object(DocumentIndexingTaskProxy, task_attribute, task),
        ):
            DocumentIndexingTaskProxy(tenant_id, dataset_id, document_ids).delay()

        task.delay.assert_called_once_with(tenant_id=tenant_id, dataset_id=dataset_id, document_ids=document_ids)
        if billing_enabled:
            mock_redis.setex.assert_called()

    def test_proxy_queues_work_when_tenant_is_busy(self, mock_redis: Mock) -> None:
        mock_redis.get.return_value = b"1"
        task = Mock()
        with (
            patch.object(DocumentIndexingTaskProxy, "features", _features(billing_enabled=True)),
            patch.object(DocumentIndexingTaskProxy, "PRIORITY_TASK_FUNC", task),
        ):
            DocumentIndexingTaskProxy(str(uuid4()), str(uuid4()), [str(uuid4())]).delay()

        mock_redis.lpush.assert_called_once()
        task.delay.assert_not_called()


class TestDocumentIndexing:
    """The core task persists validation and parsing outcomes across sessions."""

    def test_legacy_task_persists_parsing_before_runner(
        self, indexing_db: IndexingDatabase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = Mock()
        monkeypatch.setattr("tasks.document_indexing_task.FeatureService.get_features", Mock(return_value=_features()))
        monkeypatch.setattr("tasks.document_indexing_task.IndexingRunner", Mock(return_value=runner))

        document_indexing_task(indexing_db.dataset_id, list(indexing_db.document_ids))

        passed_documents = runner.run.call_args.args[0]
        assert {document.id for document in passed_documents} == set(indexing_db.document_ids)
        persisted = indexing_db.documents()
        target_documents = [document for document in persisted if document.dataset_id == indexing_db.dataset_id]
        assert all(document.indexing_status == IndexingStatus.PARSING for document in target_documents)
        assert all(document.processing_started_at is not None for document in target_documents)
        decoy = next(document for document in persisted if document.id == indexing_db.decoy_document_id)
        assert decoy.indexing_status == IndexingStatus.WAITING
        assert decoy.processing_started_at is None

    def test_missing_and_foreign_documents_are_not_processed(
        self, indexing_db: IndexingDatabase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requested = [indexing_db.document_ids[0], str(uuid4()), indexing_db.decoy_document_id]
        runner = _run_task(monkeypatch, indexing_db, document_ids=requested)

        assert [document.id for document in runner.run.call_args.args[0]] == [indexing_db.document_ids[0]]
        assert indexing_db.documents([indexing_db.decoy_document_id])[0].indexing_status == IndexingStatus.WAITING

    def test_empty_document_list_reaches_runner(
        self, indexing_db: IndexingDatabase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = _run_task(monkeypatch, indexing_db, document_ids=[])

        runner.run.assert_called_once_with([])
        assert all(document.indexing_status == IndexingStatus.WAITING for document in indexing_db.documents())

    def test_missing_dataset_returns_without_external_calls(
        self, indexing_db: IndexingDatabase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        feature_service = Mock()
        runner_factory = Mock()
        monkeypatch.setattr("tasks.document_indexing_task.FeatureService.get_features", feature_service)
        monkeypatch.setattr("tasks.document_indexing_task.IndexingRunner", runner_factory)

        _document_indexing(str(uuid4()), indexing_db.document_ids)

        feature_service.assert_not_called()
        runner_factory.assert_not_called()

    @pytest.mark.parametrize(
        ("features", "message"),
        [
            (_features(billing_enabled=True, plan=CloudPlan.SANDBOX), "does not support batch upload"),
            (_features(billing_enabled=True, vector_limit=100, vector_size=100), "over the limit"),
        ],
    )
    def test_validation_errors_are_committed_for_target_documents_only(
        self,
        indexing_db: IndexingDatabase,
        monkeypatch: pytest.MonkeyPatch,
        features: SimpleNamespace,
        message: str,
    ) -> None:
        runner = _run_task(monkeypatch, indexing_db, features=features)

        runner.run.assert_not_called()
        target_documents = indexing_db.documents(indexing_db.document_ids)
        assert all(document.indexing_status == IndexingStatus.ERROR for document in target_documents)
        assert all(message in (document.error or "") for document in target_documents)
        assert all(document.stopped_at is not None for document in target_documents)
        assert indexing_db.documents([indexing_db.decoy_document_id])[0].indexing_status == IndexingStatus.WAITING

    def test_batch_limit_error_is_persisted(
        self, indexing_db: IndexingDatabase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("tasks.document_indexing_task.dify_config.BATCH_UPLOAD_LIMIT", "1")
        runner = _run_task(monkeypatch, indexing_db, features=_features(billing_enabled=True))

        runner.run.assert_not_called()
        assert all(
            "batch upload limit" in (document.error or "")
            for document in indexing_db.documents(indexing_db.document_ids)
        )

    @pytest.mark.parametrize("failure", [DocumentIsPausedError("paused"), RuntimeError("runner failed")])
    def test_runner_failure_keeps_committed_parsing_state_and_skips_summary(
        self,
        indexing_db: IndexingDatabase,
        monkeypatch: pytest.MonkeyPatch,
        failure: Exception,
    ) -> None:
        runner = Mock()
        runner.run.side_effect = failure
        summary_delay = Mock()
        monkeypatch.setattr("tasks.document_indexing_task.generate_summary_index_task.delay", summary_delay)

        _run_task(monkeypatch, indexing_db, runner=runner)

        assert all(
            document.indexing_status == IndexingStatus.PARSING
            for document in indexing_db.documents(indexing_db.document_ids)
        )
        summary_delay.assert_not_called()

    def test_parsing_transaction_rolls_back_on_database_failure(
        self, indexing_db: IndexingDatabase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("tasks.document_indexing_task.FeatureService.get_features", Mock(return_value=_features()))

        def reject_parsing(session: Session, _flush_context: object, _instances: object) -> None:
            if any(
                isinstance(item, Document) and item.indexing_status == IndexingStatus.PARSING for item in session.dirty
            ):
                raise RuntimeError("forced flush failure")

        event.listen(indexing_db.session_maker.class_, "before_flush", reject_parsing)
        try:
            with pytest.raises(RuntimeError, match="forced flush failure"):
                _document_indexing(indexing_db.dataset_id, indexing_db.document_ids)
        finally:
            event.remove(indexing_db.session_maker.class_, "before_flush", reject_parsing)

        documents = indexing_db.documents(indexing_db.document_ids)
        assert all(document.indexing_status == IndexingStatus.WAITING for document in documents)
        assert all(document.processing_started_at is None for document in documents)


class TestSummaryGeneration:
    """Summary decisions use the runner's persisted document state."""

    def _enable_summary(self, database: IndexingDatabase) -> None:
        with database.session_maker.begin() as session:
            dataset = session.get_one(Dataset, database.dataset_id)
            dataset.summary_index_setting = {"enable": True}

    def test_only_eligible_completed_documents_are_queued(
        self, indexing_db: IndexingDatabase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._enable_summary(indexing_db)
        with indexing_db.session_maker.begin() as session:
            documents = list(session.scalars(select(Document).where(Document.dataset_id == indexing_db.dataset_id)))
            documents[0].need_summary = True
            documents[1].need_summary = True
            documents[1].doc_form = IndexStructureType.QA_INDEX
            documents[2].need_summary = True

        def complete_selected(documents: Sequence[Document]) -> None:
            with indexing_db.session_maker.begin() as session:
                session.get_one(Document, documents[0].id).indexing_status = IndexingStatus.COMPLETED
                session.get_one(Document, documents[1].id).indexing_status = IndexingStatus.COMPLETED

        runner = Mock()
        runner.run.side_effect = complete_selected
        delay = Mock()
        monkeypatch.setattr("tasks.document_indexing_task.generate_summary_index_task.delay", delay)

        _run_task(monkeypatch, indexing_db, runner=runner)

        delay.assert_called_once_with(indexing_db.dataset_id, indexing_db.document_ids[0], None)

    def test_summary_queue_failure_does_not_roll_back_indexing(
        self, indexing_db: IndexingDatabase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._enable_summary(indexing_db)
        with indexing_db.session_maker.begin() as session:
            session.get_one(Document, indexing_db.document_ids[0]).need_summary = True

        def complete(documents: Sequence[Document]) -> None:
            with indexing_db.session_maker.begin() as session:
                session.get_one(Document, documents[0].id).indexing_status = IndexingStatus.COMPLETED

        runner = Mock()
        runner.run.side_effect = complete
        delay = Mock(side_effect=RuntimeError("queue unavailable"))
        monkeypatch.setattr("tasks.document_indexing_task.generate_summary_index_task.delay", delay)

        _run_task(monkeypatch, indexing_db, document_ids=[indexing_db.document_ids[0]], runner=runner)

        delay.assert_called_once()
        assert indexing_db.documents([indexing_db.document_ids[0]])[0].indexing_status == IndexingStatus.COMPLETED

    def test_dataset_removed_by_runner_skips_summary_refresh(
        self, indexing_db: IndexingDatabase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._enable_summary(indexing_db)

        def remove_dataset(_documents: Sequence[Document]) -> None:
            with indexing_db.session_maker.begin() as session:
                session.delete(session.get_one(Dataset, indexing_db.dataset_id))

        runner = Mock()
        runner.run.side_effect = remove_dataset
        delay = Mock()
        monkeypatch.setattr("tasks.document_indexing_task.generate_summary_index_task.delay", delay)

        _run_task(monkeypatch, indexing_db, runner=runner)

        assert indexing_db.dataset() is None
        delay.assert_not_called()


class TestTenantQueue:
    """Tenant queue cleanup and continuation remain external boundary tests."""

    def test_processes_waiting_tasks_fifo(self, mock_redis: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
        tenant_id = str(uuid4())
        queued = [
            {"tenant_id": tenant_id, "dataset_id": str(uuid4()), "document_ids": [f"doc-{index}"]} for index in range(2)
        ]
        mock_redis.rpop.side_effect = [TaskWrapper(data=task).serialize() for task in queued]
        task = Mock()
        monkeypatch.setattr("tasks.document_indexing_task._document_indexing", Mock())
        monkeypatch.setattr(
            "tasks.document_indexing_task.current_app",
            SimpleNamespace(producer_or_acquire=lambda: nullcontext(object())),
        )
        monkeypatch.setattr("tasks.document_indexing_task.dify_config.TENANT_ISOLATED_TASK_CONCURRENCY", 2)

        _document_indexing_with_tenant_queue(tenant_id, str(uuid4()), [str(uuid4())], task)

        assert [call.kwargs["kwargs"]["document_ids"] for call in task.apply_async.call_args_list] == [
            ["doc-0"],
            ["doc-1"],
        ]
        assert mock_redis.setex.call_count == 2

    def test_error_still_cleans_up_empty_queue(
        self, mock_redis: Mock, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        tenant_id = str(uuid4())
        monkeypatch.setattr("tasks.document_indexing_task._document_indexing", Mock(side_effect=RuntimeError("boom")))

        with caplog.at_level(logging.ERROR, logger="tasks.document_indexing_task"):
            _document_indexing_with_tenant_queue(tenant_id, str(uuid4()), [str(uuid4())], Mock())

        mock_redis.delete.assert_called_once_with(f"tenant_document_indexing_task:{tenant_id}")
        assert any("Error processing document indexing" in message for message in caplog.messages)

    def test_tenants_use_distinct_queue_keys(self, mock_redis: Mock) -> None:
        first, second = str(uuid4()), str(uuid4())
        first_queue = TenantIsolatedTaskQueue(first, "document_indexing")
        second_queue = TenantIsolatedTaskQueue(second, "document_indexing")

        assert first_queue._queue != second_queue._queue
        assert first_queue._task_key != second_queue._task_key

    @pytest.mark.parametrize(
        ("task", "expected"),
        [
            (normal_document_indexing_task, normal_document_indexing_task),
            (priority_document_indexing_task, priority_document_indexing_task),
        ],
    )
    def test_celery_entrypoint_delegates(self, task: object, expected: object) -> None:
        with patch("tasks.document_indexing_task._document_indexing_with_tenant_queue") as handler:
            task("tenant-1", "dataset-1", ["doc-1"])  # type: ignore[operator]

        handler.assert_called_once_with("tenant-1", "dataset-1", ["doc-1"], expected)

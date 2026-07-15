"""
Unit tests for document_indexing_update_task summary generation.

After updating a document via the API, the summary index should be
regenerated under the same conditions as during initial creation:
- indexing_technique is HIGH_QUALITY
- summary_index_setting has enable=True
- document.indexing_status is COMPLETED
- document.doc_form is not QA_INDEX
- document.need_summary is True

The indexing runner remains an external boundary, but successful runner doubles
persist their final status so the task's post-indexing session performs a real
database refresh before deciding whether to enqueue the summary task.
"""

from collections.abc import Callable
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from core.indexing_runner import DocumentIsPausedError
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from models.dataset import Dataset, Document, DocumentSegment
from models.enums import DataSourceType, DocumentCreatedFrom, IndexingStatus
from tasks import document_indexing_update_task as task_module


@dataclass(frozen=True)
class TaskDatabase:
    session_maker: sessionmaker[Session]


@pytest.fixture
def task_database(sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> TaskDatabase:
    """Bind every task-owned session to an isolated SQLite database."""
    Dataset.metadata.create_all(
        sqlite_engine,
        tables=[Dataset.__table__, Document.__table__, DocumentSegment.__table__],
    )
    session_maker = sessionmaker(bind=sqlite_engine, expire_on_commit=False)

    def create_session() -> Session:
        return session_maker()

    monkeypatch.setattr(task_module.session_factory, "create_session", create_session)
    return TaskDatabase(session_maker=session_maker)


def _persist_dataset_and_document(
    database: TaskDatabase,
    *,
    indexing_technique: IndexTechniqueType = IndexTechniqueType.HIGH_QUALITY,
    summary_index_setting: dict[str, bool] | None = None,
    doc_form: IndexStructureType = IndexStructureType.PARAGRAPH_INDEX,
    need_summary: bool = True,
) -> None:
    dataset = Dataset(
        id="ds-1",
        tenant_id="tenant-1",
        name="dataset",
        created_by="account-1",
        indexing_technique=indexing_technique,
        summary_index_setting=summary_index_setting,
    )
    document = Document(
        id="doc-1",
        tenant_id="tenant-1",
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        batch="batch-1",
        name="document.txt",
        created_from=DocumentCreatedFrom.API,
        created_by="account-1",
        indexing_status=IndexingStatus.WAITING,
        doc_form=doc_form,
        need_summary=need_summary,
    )
    with database.session_maker.begin() as session:
        session.add_all([dataset, document])


def _persist_segment(database: TaskDatabase) -> None:
    segment = DocumentSegment(
        tenant_id="tenant-1",
        dataset_id="ds-1",
        document_id="doc-1",
        position=1,
        content="segment content",
        word_count=2,
        tokens=2,
        created_by="account-1",
        index_node_id="node-1",
    )
    with database.session_maker.begin() as session:
        session.add(segment)


def _build_runner(
    database: TaskDatabase,
    *,
    final_status: IndexingStatus = IndexingStatus.COMPLETED,
    after_update: Callable[[Session], None] | None = None,
) -> MagicMock:
    """Create an indexing boundary double that commits the runner's database effects."""
    runner = MagicMock()

    def run(_documents: list[Document]) -> None:
        with database.session_maker.begin() as session:
            document = session.get(Document, "doc-1")
            assert document is not None
            document.indexing_status = final_status
            if after_update is not None:
                after_update(session)

    runner.run.side_effect = run
    return runner


def _patch_external_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    *,
    runner: MagicMock,
) -> MagicMock:
    """Keep index processing and queue dispatch as explicit external doubles."""
    processor = MagicMock()
    monkeypatch.setattr(
        task_module,
        "IndexProcessorFactory",
        MagicMock(return_value=MagicMock(init_index_processor=MagicMock(return_value=processor))),
    )
    monkeypatch.setattr(task_module, "IndexingRunner", MagicMock(return_value=runner))
    return processor


def _patch_summary_delay(
    monkeypatch: pytest.MonkeyPatch,
    *,
    side_effect: Exception | None = None,
) -> MagicMock:
    delay = MagicMock(side_effect=side_effect)
    monkeypatch.setattr(task_module.generate_summary_index_task, "delay", delay)
    return delay


def _load_document(database: TaskDatabase) -> Document:
    with database.session_maker() as session:
        document = session.get(Document, "doc-1")
        assert document is not None
        return document


class TestUpdateTaskSummaryGeneration:
    """Exercise summary decisions across the task's independent real sessions."""

    def test_should_queue_summary_when_conditions_met(
        self,
        monkeypatch: pytest.MonkeyPatch,
        task_database: TaskDatabase,
    ) -> None:
        """Summary task is queued when all conditions are met."""
        _persist_dataset_and_document(task_database, summary_index_setting={"enable": True})
        runner = _build_runner(task_database)
        _patch_external_boundaries(monkeypatch, runner=runner)
        delay = _patch_summary_delay(monkeypatch)

        task_module.document_indexing_update_task("ds-1", "doc-1")

        delay.assert_called_once_with("ds-1", "doc-1", None)
        assert _load_document(task_database).indexing_status == IndexingStatus.COMPLETED

    def test_should_not_queue_when_not_high_quality(
        self,
        monkeypatch: pytest.MonkeyPatch,
        task_database: TaskDatabase,
    ) -> None:
        """Summary is skipped when indexing_technique is not high_quality."""
        _persist_dataset_and_document(
            task_database,
            indexing_technique=IndexTechniqueType.ECONOMY,
            summary_index_setting={"enable": True},
        )
        runner = _build_runner(task_database)
        _patch_external_boundaries(monkeypatch, runner=runner)
        delay = _patch_summary_delay(monkeypatch)

        task_module.document_indexing_update_task("ds-1", "doc-1")

        delay.assert_not_called()

    def test_should_not_queue_when_summary_setting_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        task_database: TaskDatabase,
    ) -> None:
        """Summary is skipped when summary_index_setting has enable=False."""
        _persist_dataset_and_document(task_database, summary_index_setting={"enable": False})
        runner = _build_runner(task_database)
        _patch_external_boundaries(monkeypatch, runner=runner)
        delay = _patch_summary_delay(monkeypatch)

        task_module.document_indexing_update_task("ds-1", "doc-1")

        delay.assert_not_called()

    def test_should_not_queue_when_summary_setting_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
        task_database: TaskDatabase,
    ) -> None:
        """Summary is skipped when summary_index_setting is None."""
        _persist_dataset_and_document(task_database, summary_index_setting=None)
        runner = _build_runner(task_database)
        _patch_external_boundaries(monkeypatch, runner=runner)
        delay = _patch_summary_delay(monkeypatch)

        task_module.document_indexing_update_task("ds-1", "doc-1")

        delay.assert_not_called()

    def test_should_not_queue_when_need_summary_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
        task_database: TaskDatabase,
    ) -> None:
        """Summary is skipped when document.need_summary is False."""
        _persist_dataset_and_document(
            task_database,
            summary_index_setting={"enable": True},
            need_summary=False,
        )
        runner = _build_runner(task_database)
        _patch_external_boundaries(monkeypatch, runner=runner)
        delay = _patch_summary_delay(monkeypatch)

        task_module.document_indexing_update_task("ds-1", "doc-1")

        delay.assert_not_called()

    def test_should_not_queue_when_qa_index_form(
        self,
        monkeypatch: pytest.MonkeyPatch,
        task_database: TaskDatabase,
    ) -> None:
        """Summary is skipped when doc_form is QA_INDEX."""
        _persist_dataset_and_document(
            task_database,
            summary_index_setting={"enable": True},
            doc_form=IndexStructureType.QA_INDEX,
        )
        runner = _build_runner(task_database)
        _patch_external_boundaries(monkeypatch, runner=runner)
        delay = _patch_summary_delay(monkeypatch)

        task_module.document_indexing_update_task("ds-1", "doc-1")

        delay.assert_not_called()

    @pytest.mark.parametrize(
        "failure",
        [Exception("indexing failed"), DocumentIsPausedError("doc-1 is paused")],
        ids=["indexing-error", "paused"],
    )
    def test_should_not_queue_when_indexing_does_not_complete(
        self,
        monkeypatch: pytest.MonkeyPatch,
        task_database: TaskDatabase,
        failure: Exception,
    ) -> None:
        """Summary is skipped when the indexing runner fails or pauses."""
        _persist_dataset_and_document(task_database, summary_index_setting={"enable": True})
        runner = MagicMock()
        runner.run.side_effect = failure
        _patch_external_boundaries(monkeypatch, runner=runner)
        delay = _patch_summary_delay(monkeypatch)

        task_module.document_indexing_update_task("ds-1", "doc-1")

        delay.assert_not_called()
        assert _load_document(task_database).indexing_status == IndexingStatus.PARSING

    def test_should_not_queue_when_dataset_not_found_after_indexing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        task_database: TaskDatabase,
    ) -> None:
        """Summary is skipped when the dataset disappears after indexing."""
        _persist_dataset_and_document(task_database, summary_index_setting={"enable": True})

        def delete_dataset(session: Session) -> None:
            dataset = session.get(Dataset, "ds-1")
            assert dataset is not None
            session.delete(dataset)

        runner = _build_runner(task_database, after_update=delete_dataset)
        _patch_external_boundaries(monkeypatch, runner=runner)
        delay = _patch_summary_delay(monkeypatch)

        task_module.document_indexing_update_task("ds-1", "doc-1")

        delay.assert_not_called()
        with task_database.session_maker() as session:
            assert session.get(Dataset, "ds-1") is None

    def test_should_not_queue_when_document_not_completed_after_indexing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        task_database: TaskDatabase,
    ) -> None:
        """Summary is skipped when the persisted indexing status is not COMPLETED."""
        _persist_dataset_and_document(task_database, summary_index_setting={"enable": True})
        runner = _build_runner(task_database, final_status=IndexingStatus.ERROR)
        _patch_external_boundaries(monkeypatch, runner=runner)
        delay = _patch_summary_delay(monkeypatch)

        task_module.document_indexing_update_task("ds-1", "doc-1")

        delay.assert_not_called()
        assert _load_document(task_database).indexing_status == IndexingStatus.ERROR

    def test_should_swallow_summary_queue_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        task_database: TaskDatabase,
    ) -> None:
        """Task should not raise when generate_summary_index_task.delay raises."""
        _persist_dataset_and_document(task_database, summary_index_setting={"enable": True})
        runner = _build_runner(task_database)
        _patch_external_boundaries(monkeypatch, runner=runner)
        delay = _patch_summary_delay(monkeypatch, side_effect=Exception("queue full"))

        task_module.document_indexing_update_task("ds-1", "doc-1")

        delay.assert_called_once_with("ds-1", "doc-1", None)

    def test_should_queue_summary_with_segments_and_delete_them(
        self,
        monkeypatch: pytest.MonkeyPatch,
        task_database: TaskDatabase,
    ) -> None:
        """A cleaned segment is committed as deleted before summary generation."""
        _persist_dataset_and_document(task_database, summary_index_setting={"enable": True})
        _persist_segment(task_database)
        runner = _build_runner(task_database)
        processor = _patch_external_boundaries(monkeypatch, runner=runner)
        delay = _patch_summary_delay(monkeypatch)

        task_module.document_indexing_update_task("ds-1", "doc-1")

        delay.assert_called_once_with("ds-1", "doc-1", None)
        processor.clean.assert_called_once()
        assert processor.clean.call_args.args[1] == ["node-1"]
        with task_database.session_maker() as session:
            assert session.scalars(select(DocumentSegment)).all() == []

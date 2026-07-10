"""State-based tests for :mod:`services.summary_index_service`."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from sqlalchemy import Engine, event, select
from sqlalchemy.orm import Session

import services.summary_index_service as summary_module
from core.db.session_factory import session_factory
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from models.base import TypeBase
from models.dataset import Dataset, Document, DocumentSegment, DocumentSegmentSummary
from models.enums import DataSourceType, DocumentCreatedFrom, SegmentStatus, SummaryStatus
from services.summary_index_service import SummaryIndexService


@pytest.fixture
def orm_session(sqlite_engine: Engine) -> Iterator[Session]:
    """Provide a real session and bind service-owned sessions to the same database."""

    tables = [model.__table__ for model in (Dataset, Document, DocumentSegment, DocumentSegmentSummary)]
    TypeBase.metadata.create_all(sqlite_engine, tables=tables)
    session_factory.configure(sqlite_engine, expire_on_commit=False)
    with Session(sqlite_engine, expire_on_commit=False) as session:
        yield session


@dataclass(frozen=True)
class SummaryRows:
    dataset: Dataset
    document: Document
    segment: DocumentSegment


@pytest.fixture
def summary_rows(orm_session: Session) -> SummaryRows:
    dataset = Dataset(
        id="dataset-1",
        tenant_id="tenant-1",
        name="Knowledge",
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique=IndexTechniqueType.HIGH_QUALITY,
        created_by="user-1",
        embedding_model_provider="openai",
        embedding_model="text-embedding",
    )
    document = Document(
        id="doc-1",
        tenant_id="tenant-1",
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        batch="batch-1",
        name="Document",
        created_from=DocumentCreatedFrom.WEB,
        created_by="user-1",
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
        doc_language="en",
    )
    segment = DocumentSegment(
        tenant_id="tenant-1",
        dataset_id=dataset.id,
        document_id=document.id,
        position=1,
        content="hello world",
        word_count=2,
        tokens=2,
        created_by="user-1",
        status=SegmentStatus.COMPLETED,
    )
    orm_session.add_all([dataset, document, segment])
    orm_session.commit()
    return SummaryRows(dataset=dataset, document=document, segment=segment)


def _summary(
    session: Session,
    rows: SummaryRows,
    *,
    content: str = "summary",
    status: SummaryStatus = SummaryStatus.GENERATING,
    enabled: bool = True,
    node_id: str | None = None,
) -> DocumentSegmentSummary:
    record = DocumentSegmentSummary(
        dataset_id=rows.dataset.id,
        document_id=rows.document.id,
        chunk_id=rows.segment.id,
        summary_content=content,
        summary_index_node_id=node_id,
        status=status,
        enabled=enabled,
    )
    session.add(record)
    session.commit()
    return record


@contextmanager
def _raise_on_sql(engine: Engine, table_name: str, operation: str) -> Iterator[None]:
    def raise_error(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith(operation) and table_name in statement:
            raise RuntimeError(f"forced {operation}")

    event.listen(engine, "before_cursor_execute", raise_error)
    try:
        yield
    finally:
        event.remove(engine, "before_cursor_execute", raise_error)


def _mock_vectorization(monkeypatch: pytest.MonkeyPatch, *, add_effect=None) -> MagicMock:
    monkeypatch.setattr(summary_module.helper, "generate_text_hash", MagicMock(return_value="hash-1"))
    monkeypatch.setattr(summary_module.uuid, "uuid4", MagicMock(return_value="node-1"))
    model_manager = MagicMock()
    model_manager.get_model_instance.return_value = None
    monkeypatch.setattr(summary_module.ModelManager, "for_tenant", MagicMock(return_value=model_manager))
    vector = MagicMock()
    vector.add_texts.side_effect = add_effect
    monkeypatch.setattr(summary_module, "Vector", MagicMock(return_value=vector))
    return vector


def test_generate_summary_for_segment_passes_document_language(
    monkeypatch: pytest.MonkeyPatch, summary_rows: SummaryRows
) -> None:
    usage = MagicMock()
    processor = SimpleNamespace(generate_summary=MagicMock(return_value=("sum", usage)))
    monkeypatch.setitem(
        sys.modules,
        "core.rag.index_processor.processor.paragraph_index_processor",
        SimpleNamespace(ParagraphIndexProcessor=processor),
    )
    with patch.object(DocumentSegment, "document", new_callable=PropertyMock, return_value=summary_rows.document):
        content, result_usage = SummaryIndexService.generate_summary_for_segment(
            summary_rows.segment, summary_rows.dataset, {"enable": True}
        )
    assert content == "sum"
    assert result_usage is usage
    assert processor.generate_summary.call_args.kwargs["document_language"] == "en"


def test_generate_summary_for_segment_rejects_empty(monkeypatch: pytest.MonkeyPatch, summary_rows: SummaryRows) -> None:
    processor = SimpleNamespace(generate_summary=MagicMock(return_value=("", MagicMock())))
    monkeypatch.setitem(
        sys.modules,
        "core.rag.index_processor.processor.paragraph_index_processor",
        SimpleNamespace(ParagraphIndexProcessor=processor),
    )
    with (
        patch.object(DocumentSegment, "document", new_callable=PropertyMock, return_value=summary_rows.document),
        pytest.raises(ValueError, match="Generated summary is empty"),
    ):
        SummaryIndexService.generate_summary_for_segment(summary_rows.segment, summary_rows.dataset, {"enable": True})


def test_create_summary_record_creates_and_reenables_existing(orm_session: Session, summary_rows: SummaryRows) -> None:
    created = SummaryIndexService.create_summary_record(
        summary_rows.segment, summary_rows.dataset, "first", session=orm_session
    )
    created.enabled = False
    created.disabled_by = "user-1"
    orm_session.commit()

    updated = SummaryIndexService.create_summary_record(
        summary_rows.segment, summary_rows.dataset, "second", session=orm_session
    )
    assert updated.id == created.id
    assert updated.summary_content == "second"
    assert updated.enabled is True
    assert updated.disabled_by is None
    assert orm_session.scalar(select(DocumentSegmentSummary).where(DocumentSegmentSummary.chunk_id == created.chunk_id))


def test_create_summary_record_flush_failure_leaves_no_row(
    orm_session: Session, sqlite_engine: Engine, summary_rows: SummaryRows
) -> None:
    with (
        _raise_on_sql(sqlite_engine, "document_segment_summaries", "INSERT"),
        pytest.raises(RuntimeError, match="forced INSERT"),
    ):
        SummaryIndexService.create_summary_record(
            summary_rows.segment, summary_rows.dataset, "summary", session=orm_session
        )
    orm_session.rollback()
    assert orm_session.scalar(select(DocumentSegmentSummary)) is None


def test_vectorize_summary_skips_economy_and_rejects_blank(
    monkeypatch: pytest.MonkeyPatch, orm_session: Session, summary_rows: SummaryRows
) -> None:
    record = _summary(orm_session, summary_rows)
    summary_rows.dataset.indexing_technique = IndexTechniqueType.ECONOMY
    vector_cls = MagicMock()
    monkeypatch.setattr(summary_module, "Vector", vector_cls)
    SummaryIndexService.vectorize_summary(record, summary_rows.segment, summary_rows.dataset, session=orm_session)
    vector_cls.assert_not_called()
    summary_rows.dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
    record.summary_content = " "
    with pytest.raises(ValueError, match="Summary content is empty"):
        SummaryIndexService.vectorize_summary(record, summary_rows.segment, summary_rows.dataset, session=orm_session)


def test_vectorize_summary_retries_and_flushes_caller_transaction(
    monkeypatch: pytest.MonkeyPatch, orm_session: Session, summary_rows: SummaryRows
) -> None:
    record = _summary(orm_session, summary_rows)
    vector = _mock_vectorization(monkeypatch, add_effect=[RuntimeError("connection timeout"), None])
    monkeypatch.setattr(summary_module.time, "sleep", MagicMock())
    SummaryIndexService.vectorize_summary(record, summary_rows.segment, summary_rows.dataset, session=orm_session)
    assert vector.add_texts.call_count == 2
    assert record.status == SummaryStatus.COMPLETED
    assert record.summary_index_node_id == "node-1"
    assert record.summary_index_node_hash == "hash-1"
    assert orm_session.in_transaction()


def test_vectorize_summary_owned_session_updates_persisted_row(
    monkeypatch: pytest.MonkeyPatch, orm_session: Session, summary_rows: SummaryRows
) -> None:
    record = _summary(orm_session, summary_rows, node_id="existing-node")
    record_id = record.id
    _mock_vectorization(monkeypatch)
    SummaryIndexService.vectorize_summary(record, summary_rows.segment, summary_rows.dataset)
    orm_session.expire_all()
    persisted = orm_session.get(DocumentSegmentSummary, record_id)
    assert persisted is not None
    assert persisted.status == SummaryStatus.COMPLETED
    assert persisted.summary_index_node_id == "existing-node"


def test_vectorize_summary_final_failure_persists_error(
    monkeypatch: pytest.MonkeyPatch, orm_session: Session, summary_rows: SummaryRows
) -> None:
    record = _summary(orm_session, summary_rows)
    record_id = record.id
    _mock_vectorization(monkeypatch, add_effect=RuntimeError("boom"))
    monkeypatch.setattr(summary_module.time, "sleep", MagicMock())
    with pytest.raises(RuntimeError, match="boom"):
        SummaryIndexService.vectorize_summary(record, summary_rows.segment, summary_rows.dataset)
    orm_session.expire_all()
    persisted = orm_session.get(DocumentSegmentSummary, record_id)
    assert persisted is not None
    assert persisted.status == SummaryStatus.ERROR
    assert "Vectorization failed" in (persisted.error or "")


def test_batch_create_records_creates_updates_and_reenables(orm_session: Session, summary_rows: SummaryRows) -> None:
    existing = _summary(orm_session, summary_rows, enabled=False)
    existing.disabled_by = "user-1"
    orm_session.commit()
    second = DocumentSegment(
        tenant_id="tenant-1",
        dataset_id=summary_rows.dataset.id,
        document_id=summary_rows.document.id,
        position=2,
        content="second",
        word_count=1,
        tokens=1,
        created_by="user-1",
        status=SegmentStatus.COMPLETED,
    )
    orm_session.add(second)
    orm_session.commit()
    SummaryIndexService.batch_create_summary_records(
        [summary_rows.segment, second], summary_rows.dataset, SummaryStatus.NOT_STARTED
    )
    orm_session.expire_all()
    records = orm_session.scalars(select(DocumentSegmentSummary).order_by(DocumentSegmentSummary.chunk_id)).all()
    assert len(records) == 2
    assert all(record.status == SummaryStatus.NOT_STARTED for record in records)
    assert all(record.enabled for record in records)


def test_update_summary_record_error_updates_only_matching_dataset(
    orm_session: Session, summary_rows: SummaryRows
) -> None:
    record = _summary(orm_session, summary_rows)
    SummaryIndexService.update_summary_record_error(summary_rows.segment, summary_rows.dataset, "failure")
    orm_session.expire_all()
    persisted = orm_session.get(DocumentSegmentSummary, record.id)
    assert persisted is not None
    assert persisted.status == SummaryStatus.ERROR
    assert persisted.error == "failure"


def test_generate_and_vectorize_summary_persists_success(
    monkeypatch: pytest.MonkeyPatch, orm_session: Session, summary_rows: SummaryRows
) -> None:
    monkeypatch.setattr(
        SummaryIndexService,
        "generate_summary_for_segment",
        MagicMock(return_value=("generated", MagicMock(total_tokens=0))),
    )
    monkeypatch.setattr(SummaryIndexService, "vectorize_summary", MagicMock(return_value=None))
    result = SummaryIndexService.generate_and_vectorize_summary(
        summary_rows.segment, summary_rows.dataset, {"enable": True}, session=orm_session
    )
    assert result.summary_content == "generated"
    assert orm_session.get(DocumentSegmentSummary, result.id) is not None


def test_generate_and_vectorize_summary_persists_vector_error(
    monkeypatch: pytest.MonkeyPatch, orm_session: Session, summary_rows: SummaryRows
) -> None:
    record = _summary(orm_session, summary_rows, content="")
    monkeypatch.setattr(
        SummaryIndexService,
        "generate_summary_for_segment",
        MagicMock(return_value=("generated", MagicMock(total_tokens=0))),
    )
    monkeypatch.setattr(SummaryIndexService, "vectorize_summary", MagicMock(side_effect=RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        SummaryIndexService.generate_and_vectorize_summary(
            summary_rows.segment, summary_rows.dataset, {"enable": True}, session=orm_session
        )
    orm_session.expire_all()
    persisted = orm_session.get(DocumentSegmentSummary, record.id)
    assert persisted is not None
    assert persisted.status == SummaryStatus.ERROR


def test_generate_summaries_for_document_reads_real_segments(
    monkeypatch: pytest.MonkeyPatch, orm_session: Session, summary_rows: SummaryRows
) -> None:
    monkeypatch.setattr(SummaryIndexService, "batch_create_summary_records", MagicMock())
    generated = MagicMock()
    monkeypatch.setattr(SummaryIndexService, "generate_and_vectorize_summary", MagicMock(return_value=generated))
    with patch.object(DocumentSegment, "document", new_callable=PropertyMock, return_value=summary_rows.document):
        results = SummaryIndexService.generate_summaries_for_document(
            summary_rows.dataset, summary_rows.document, {"enable": True}
        )
    assert results == [generated]


def test_disable_enable_and_delete_summaries_persist_state(
    monkeypatch: pytest.MonkeyPatch, orm_session: Session, summary_rows: SummaryRows
) -> None:
    record = _summary(orm_session, summary_rows, status=SummaryStatus.COMPLETED, node_id="node-1")
    record_id = record.id
    SummaryIndexService.disable_summaries_for_segments(
        summary_rows.dataset, segment_ids=[summary_rows.segment.id], disabled_by="user-1"
    )
    orm_session.expire_all()
    persisted = orm_session.get(DocumentSegmentSummary, record_id)
    assert persisted is not None
    assert persisted.enabled is False
    assert persisted.disabled_by == "user-1"

    monkeypatch.setattr(SummaryIndexService, "vectorize_summary", MagicMock(return_value=None))
    SummaryIndexService.enable_summaries_for_segments(summary_rows.dataset, segment_ids=[summary_rows.segment.id])
    orm_session.expire_all()
    assert orm_session.get(DocumentSegmentSummary, record_id).enabled is True

    vector = MagicMock()
    monkeypatch.setattr(summary_module, "Vector", MagicMock(return_value=vector))
    SummaryIndexService.delete_summaries_for_segments(summary_rows.dataset, segment_ids=[summary_rows.segment.id])
    orm_session.expire_all()
    assert orm_session.get(DocumentSegmentSummary, record_id) is None
    vector.delete_by_ids.assert_called_once_with(["node-1"])


def test_update_summary_for_segment_deletes_empty_summary(
    monkeypatch: pytest.MonkeyPatch, orm_session: Session, summary_rows: SummaryRows
) -> None:
    record = _summary(orm_session, summary_rows, node_id="node-1")
    record_id = record.id
    vector = MagicMock()
    monkeypatch.setattr(summary_module, "Vector", MagicMock(return_value=vector))
    with patch.object(DocumentSegment, "document", new_callable=PropertyMock, return_value=summary_rows.document):
        assert (
            SummaryIndexService.update_summary_for_segment(
                summary_rows.segment, summary_rows.dataset, " ", session=orm_session
            )
            is None
        )
    assert orm_session.get(DocumentSegmentSummary, record_id) is None
    vector.delete_by_ids.assert_called_once_with(["node-1"])


def test_update_summary_for_segment_persists_success_and_vector_error(
    monkeypatch: pytest.MonkeyPatch, orm_session: Session, summary_rows: SummaryRows
) -> None:
    record = _summary(orm_session, summary_rows, node_id="node-1")
    with (
        patch.object(DocumentSegment, "document", new_callable=PropertyMock, return_value=summary_rows.document),
        patch.object(SummaryIndexService, "vectorize_summary", return_value=None),
    ):
        updated = SummaryIndexService.update_summary_for_segment(
            summary_rows.segment, summary_rows.dataset, "new", session=orm_session
        )
    assert updated is not None
    assert updated.summary_content == "new"

    with (
        patch.object(DocumentSegment, "document", new_callable=PropertyMock, return_value=summary_rows.document),
        patch.object(SummaryIndexService, "vectorize_summary", side_effect=RuntimeError("boom")),
    ):
        failed = SummaryIndexService.update_summary_for_segment(
            summary_rows.segment, summary_rows.dataset, "again", session=orm_session
        )
    assert failed is not None
    assert failed.status == SummaryStatus.ERROR
    assert "Vectorization failed" in (failed.error or "")


def test_summary_read_helpers_filter_disabled_and_dataset_rows(orm_session: Session, summary_rows: SummaryRows) -> None:
    enabled = _summary(orm_session, summary_rows, status=SummaryStatus.COMPLETED)
    disabled = DocumentSegmentSummary(
        dataset_id=summary_rows.dataset.id,
        document_id=summary_rows.document.id,
        chunk_id="other-segment",
        summary_content="hidden",
        enabled=False,
    )
    orm_session.add(disabled)
    orm_session.commit()
    assert (
        SummaryIndexService.get_segment_summary(
            summary_rows.segment.id, summary_rows.dataset.id, session=orm_session
        ).id
        == enabled.id
    )
    assert (
        SummaryIndexService.get_segment_summary("other-segment", summary_rows.dataset.id, session=orm_session) is None
    )
    assert SummaryIndexService.get_segments_summaries(
        [summary_rows.segment.id, "other-segment"], summary_rows.dataset.id, session=orm_session
    ) == {summary_rows.segment.id: enabled}
    assert SummaryIndexService.get_document_summaries(
        summary_rows.document.id, summary_rows.dataset.id, session=orm_session
    ) == [enabled]


def test_document_summary_status_uses_real_segments_and_summaries(
    orm_session: Session, summary_rows: SummaryRows
) -> None:
    _summary(orm_session, summary_rows, status=SummaryStatus.GENERATING)
    assert (
        SummaryIndexService.get_document_summary_index_status(
            summary_rows.document.id,
            summary_rows.dataset.id,
            summary_rows.dataset.tenant_id,
            session=orm_session,
        )
        == "SUMMARIZING"
    )
    result = SummaryIndexService.get_documents_summary_index_status(
        [summary_rows.document.id, "missing-document"],
        summary_rows.dataset.id,
        summary_rows.dataset.tenant_id,
        session=orm_session,
    )
    assert result == {summary_rows.document.id: "SUMMARIZING", "missing-document": None}


def test_document_summary_status_is_tenant_scoped(orm_session: Session, summary_rows: SummaryRows) -> None:
    foreign_segment = DocumentSegment(
        tenant_id="tenant-2",
        dataset_id=summary_rows.dataset.id,
        document_id=summary_rows.document.id,
        position=2,
        content="foreign",
        word_count=1,
        tokens=1,
        created_by="user-2",
        status=SegmentStatus.COMPLETED,
    )
    orm_session.add(foreign_segment)
    orm_session.commit()
    foreign_summary = DocumentSegmentSummary(
        dataset_id=summary_rows.dataset.id,
        document_id=summary_rows.document.id,
        chunk_id=foreign_segment.id,
        status=SummaryStatus.GENERATING,
    )
    orm_session.add(foreign_summary)
    orm_session.commit()
    assert (
        SummaryIndexService.get_document_summary_index_status(
            summary_rows.document.id,
            summary_rows.dataset.id,
            "tenant-1",
            session=orm_session,
        )
        is None
    )


def test_update_error_logs_when_record_missing(caplog: pytest.LogCaptureFixture, summary_rows: SummaryRows) -> None:
    with caplog.at_level(logging.WARNING, logger="services.summary_index_service"):
        SummaryIndexService.update_summary_record_error(summary_rows.segment, summary_rows.dataset, "missing")
    assert "not found" in caplog.text

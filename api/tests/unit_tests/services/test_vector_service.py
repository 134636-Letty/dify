"""SQLite-backed tests for :mod:`services.vector_service`."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from sqlalchemy import Engine, event, select
from sqlalchemy.orm import Session

import models.dataset as dataset_module
import services.vector_service as vector_module
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from extensions.storage.storage_type import StorageType
from models.base import TypeBase
from models.dataset import (
    ChildChunk,
    Dataset,
    DatasetProcessRule,
    Document,
    DocumentSegment,
    SegmentAttachmentBinding,
)
from models.enums import CreatorUserRole, DataSourceType, DocumentCreatedFrom, ProcessRuleMode, SegmentStatus
from models.model import UploadFile
from services.vector_service import VectorService


@dataclass(frozen=True)
class Rows:
    dataset: Dataset
    document: Document
    rule: DatasetProcessRule
    segment: DocumentSegment


@pytest.fixture
def orm_session(sqlite_engine: Engine) -> Iterator[Session]:
    models = (Dataset, Document, DatasetProcessRule, DocumentSegment, ChildChunk, UploadFile, SegmentAttachmentBinding)
    TypeBase.metadata.create_all(sqlite_engine, tables=[model.__table__ for model in models])
    with Session(sqlite_engine, expire_on_commit=False) as session:
        yield session


@pytest.fixture
def rows(orm_session: Session) -> Rows:
    dataset = Dataset(
        id="dataset-1",
        tenant_id="tenant-1",
        name="Knowledge",
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique=IndexTechniqueType.HIGH_QUALITY,
        created_by="user-1",
        embedding_model_provider="openai",
        embedding_model="text-embedding",
        chunk_structure=IndexStructureType.PARAGRAPH_INDEX,
        is_multimodal=False,
    )
    rule = DatasetProcessRule(
        dataset_id=dataset.id,
        mode=ProcessRuleMode.HIERARCHICAL,
        rules=json.dumps({"segmentation": {"delimiter": "\\n", "max_tokens": 100}}),
        created_by="user-1",
    )
    document = Document(
        id="document-1",
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        dataset_process_rule_id=rule.id,
        batch="batch-1",
        name="Document",
        created_from=DocumentCreatedFrom.WEB,
        created_by="user-1",
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
        doc_language="en",
    )
    segment = DocumentSegment(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        position=1,
        content="hello world",
        word_count=2,
        tokens=2,
        created_by="user-1",
        index_node_id="node-1",
        index_node_hash="hash-1",
        status=SegmentStatus.COMPLETED,
    )
    orm_session.add_all([dataset, rule, document, segment])
    orm_session.commit()
    return Rows(dataset=dataset, document=document, rule=rule, segment=segment)


def _upload(session: Session, *, name: str, tenant_id: str = "tenant-1") -> UploadFile:
    upload = UploadFile(
        tenant_id=tenant_id,
        storage_type=StorageType.LOCAL,
        key=f"key-{name}",
        name=name,
        size=10,
        extension="png",
        mime_type="image/png",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by="user-1",
        created_at=datetime(2024, 1, 1),
        used=False,
    )
    session.add(upload)
    session.commit()
    return upload


def _processor(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    processor = MagicMock()
    factory = MagicMock()
    factory.init_index_processor.return_value = processor
    monkeypatch.setattr(vector_module, "IndexProcessorFactory", MagicMock(return_value=factory))
    return processor


@contextmanager
def _bind_attachment_property(session: Session) -> Iterator[None]:
    with patch.object(dataset_module.db, "session", session):
        yield


@contextmanager
def _raise_before_commit(session: Session) -> Iterator[None]:
    def raise_error(_session: Session) -> None:
        raise RuntimeError("forced binding commit")

    event.listen(session, "before_commit", raise_error)
    try:
        yield
    finally:
        event.remove(session, "before_commit", raise_error)


def test_create_segments_vector_loads_regular_and_multimodal_documents(
    monkeypatch: pytest.MonkeyPatch, orm_session: Session, rows: Rows
) -> None:
    processor = _processor(monkeypatch)
    with _bind_attachment_property(orm_session):
        VectorService.create_segments_vector(
            [["keyword"]], [rows.segment], rows.dataset, IndexStructureType.PARAGRAPH_INDEX, orm_session
        )
    processor.load.assert_called_once()
    assert processor.load.call_args.kwargs["keywords_list"] == [["keyword"]]

    upload = _upload(orm_session, name="image.png")
    orm_session.add(
        SegmentAttachmentBinding(
            tenant_id=rows.dataset.tenant_id,
            dataset_id=rows.dataset.id,
            document_id=rows.document.id,
            segment_id=rows.segment.id,
            attachment_id=upload.id,
        )
    )
    rows.dataset.is_multimodal = True
    orm_session.commit()
    processor.reset_mock()
    with _bind_attachment_property(orm_session):
        VectorService.create_segments_vector(
            None, [rows.segment], rows.dataset, IndexStructureType.PARAGRAPH_INDEX, orm_session
        )
    assert processor.load.call_count == 2
    assert len(processor.load.call_args_list[1].args[2]) == 1


def test_create_segments_vector_empty_does_not_load(
    monkeypatch: pytest.MonkeyPatch, orm_session: Session, rows: Rows
) -> None:
    processor = _processor(monkeypatch)
    VectorService.create_segments_vector(None, [], rows.dataset, IndexStructureType.PARAGRAPH_INDEX, orm_session)
    processor.load.assert_not_called()


def test_parent_child_lookup_uses_persisted_document_and_rule(
    monkeypatch: pytest.MonkeyPatch, orm_session: Session, rows: Rows
) -> None:
    rows.dataset.chunk_structure = IndexStructureType.PARENT_CHILD_INDEX
    orm_session.commit()
    model = MagicMock()
    manager = MagicMock()
    manager.get_model_instance.return_value = model
    monkeypatch.setattr(vector_module.ModelManager, "for_tenant", MagicMock(return_value=manager))
    generate = MagicMock()
    monkeypatch.setattr(VectorService, "generate_child_chunks", generate)
    processor = _processor(monkeypatch)
    VectorService.create_segments_vector(
        None, [rows.segment], rows.dataset, IndexStructureType.PARENT_CHILD_INDEX, orm_session
    )
    generate.assert_called_once_with(rows.segment, rows.document, rows.dataset, model, rows.rule, orm_session, False)
    processor.load.assert_not_called()


def test_parent_child_uses_default_model_and_validates_persisted_state(
    monkeypatch: pytest.MonkeyPatch, orm_session: Session, rows: Rows
) -> None:
    rows.dataset.chunk_structure = IndexStructureType.PARENT_CHILD_INDEX
    rows.dataset.embedding_model_provider = None
    orm_session.commit()
    manager = MagicMock()
    manager.get_default_model_instance.return_value = MagicMock()
    monkeypatch.setattr(vector_module.ModelManager, "for_tenant", MagicMock(return_value=manager))
    monkeypatch.setattr(VectorService, "generate_child_chunks", MagicMock())
    _processor(monkeypatch)
    VectorService.create_segments_vector(
        None, [rows.segment], rows.dataset, IndexStructureType.PARENT_CHILD_INDEX, orm_session
    )
    manager.get_default_model_instance.assert_called_once()

    orm_session.delete(rows.rule)
    orm_session.commit()
    with pytest.raises(ValueError, match="No processing rule found"):
        VectorService.create_segments_vector(
            None, [rows.segment], rows.dataset, IndexStructureType.PARENT_CHILD_INDEX, orm_session
        )


def test_parent_child_missing_document_logs_and_economy_rejects(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    orm_session: Session,
    rows: Rows,
) -> None:
    missing_segment = DocumentSegment(
        tenant_id="tenant-1",
        dataset_id=rows.dataset.id,
        document_id="missing",
        position=2,
        content="missing",
        word_count=1,
        tokens=1,
        created_by="user-1",
        index_node_id="node-2",
        status=SegmentStatus.COMPLETED,
    )
    orm_session.add(missing_segment)
    orm_session.commit()
    _processor(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="services.vector_service"):
        VectorService.create_segments_vector(
            None, [missing_segment], rows.dataset, IndexStructureType.PARENT_CHILD_INDEX, orm_session
        )
    assert "none was found" in caplog.text

    rows.dataset.indexing_technique = IndexTechniqueType.ECONOMY
    orm_session.commit()
    with pytest.raises(ValueError, match="not high quality"):
        VectorService.create_segments_vector(
            None, [rows.segment], rows.dataset, IndexStructureType.PARENT_CHILD_INDEX, orm_session
        )


def test_update_segment_vector_uses_vector_or_keyword(monkeypatch: pytest.MonkeyPatch, rows: Rows) -> None:
    vector = MagicMock()
    monkeypatch.setattr(vector_module, "Vector", MagicMock(return_value=vector))
    VectorService.update_segment_vector(["keyword"], rows.segment, rows.dataset)
    vector.delete_by_ids.assert_called_once_with([rows.segment.index_node_id])
    vector.add_texts.assert_called_once()

    rows.dataset.indexing_technique = IndexTechniqueType.ECONOMY
    keyword = MagicMock()
    monkeypatch.setattr(vector_module, "Keyword", MagicMock(return_value=keyword))
    VectorService.update_segment_vector(["one", "two"], rows.segment, rows.dataset)
    assert keyword.add_texts.call_args.kwargs["keywords_list"] == [["one", "two"]]
    keyword.reset_mock()
    VectorService.update_segment_vector(None, rows.segment, rows.dataset)
    assert "keywords_list" not in keyword.add_texts.call_args.kwargs


def test_generate_child_chunks_persists_children_and_commits(
    monkeypatch: pytest.MonkeyPatch, orm_session: Session, rows: Rows
) -> None:
    child = SimpleNamespace(page_content="child", metadata={"doc_id": "child-node", "doc_hash": "child-hash"})
    transformed = [SimpleNamespace(children=[child])]
    processor = _processor(monkeypatch)
    processor.transform.return_value = transformed
    VectorService.generate_child_chunks(
        rows.segment, rows.document, rows.dataset, MagicMock(), rows.rule, orm_session, regenerate=True
    )
    persisted = orm_session.scalar(select(ChildChunk).where(ChildChunk.segment_id == rows.segment.id))
    assert persisted is not None
    assert persisted.content == "child"
    assert persisted.index_node_id == "child-node"
    processor.clean.assert_called_once()
    processor.load.assert_called_once()


def test_generate_child_chunks_empty_result_persists_nothing(
    monkeypatch: pytest.MonkeyPatch, orm_session: Session, rows: Rows
) -> None:
    processor = _processor(monkeypatch)
    processor.transform.return_value = [SimpleNamespace(children=[])]
    VectorService.generate_child_chunks(
        rows.segment, rows.document, rows.dataset, MagicMock(), rows.rule, orm_session, regenerate=False
    )
    assert orm_session.scalar(select(ChildChunk)) is None
    processor.load.assert_not_called()


def test_child_chunk_vector_create_update_delete_boundaries(
    monkeypatch: pytest.MonkeyPatch, orm_session: Session, rows: Rows
) -> None:
    child = ChildChunk(
        tenant_id=rows.dataset.tenant_id,
        dataset_id=rows.dataset.id,
        document_id=rows.document.id,
        segment_id=rows.segment.id,
        position=1,
        content="child",
        word_count=1,
        created_by="user-1",
        index_node_id="child-node",
        index_node_hash="child-hash",
    )
    orm_session.add(child)
    orm_session.commit()
    vector = MagicMock()
    monkeypatch.setattr(vector_module, "Vector", MagicMock(return_value=vector))
    VectorService.create_child_chunk_vector(child, rows.dataset)
    VectorService.update_child_chunk_vector([child], [child], [child], rows.dataset)
    VectorService.delete_child_chunk_vector(child, rows.dataset)
    assert vector.add_texts.call_count == 2
    assert vector.delete_by_ids.call_count == 2

    rows.dataset.indexing_technique = IndexTechniqueType.ECONOMY
    vector.reset_mock()
    VectorService.create_child_chunk_vector(child, rows.dataset)
    VectorService.update_child_chunk_vector([child], [child], [child], rows.dataset)
    vector.add_texts.assert_not_called()


def test_update_multimodal_vector_replaces_bindings_and_vectors(
    monkeypatch: pytest.MonkeyPatch, orm_session: Session, rows: Rows
) -> None:
    old = _upload(orm_session, name="old.png")
    new = _upload(orm_session, name="new.png")
    foreign = _upload(orm_session, name="foreign.png", tenant_id="tenant-2")
    orm_session.add(
        SegmentAttachmentBinding(
            tenant_id=rows.dataset.tenant_id,
            dataset_id=rows.dataset.id,
            document_id=rows.document.id,
            segment_id=rows.segment.id,
            attachment_id=old.id,
        )
    )
    rows.dataset.is_multimodal = True
    orm_session.commit()
    vector = MagicMock()
    monkeypatch.setattr(vector_module, "Vector", MagicMock(return_value=vector))
    with _bind_attachment_property(orm_session):
        VectorService.update_multimodel_vector(rows.segment, [new.id, "missing", foreign.id], rows.dataset, orm_session)
    bindings = orm_session.scalars(
        select(SegmentAttachmentBinding).where(SegmentAttachmentBinding.segment_id == rows.segment.id)
    ).all()
    assert {binding.attachment_id for binding in bindings} == {new.id, foreign.id}
    vector.delete_by_ids.assert_called_once_with([old.id])
    assert len(vector.create_multimodal.call_args.args[0]) == 2


def test_update_multimodal_vector_empty_ids_deletes_bindings(
    monkeypatch: pytest.MonkeyPatch, orm_session: Session, rows: Rows
) -> None:
    upload = _upload(orm_session, name="old.png")
    orm_session.add(
        SegmentAttachmentBinding(
            tenant_id=rows.dataset.tenant_id,
            dataset_id=rows.dataset.id,
            document_id=rows.document.id,
            segment_id=rows.segment.id,
            attachment_id=upload.id,
        )
    )
    rows.dataset.is_multimodal = True
    orm_session.commit()
    monkeypatch.setattr(vector_module, "Vector", MagicMock(return_value=MagicMock()))
    with _bind_attachment_property(orm_session):
        VectorService.update_multimodel_vector(rows.segment, [], rows.dataset, orm_session)
    assert orm_session.scalar(select(SegmentAttachmentBinding)) is None


def test_update_multimodal_vector_noops_for_economy_or_unchanged(
    monkeypatch: pytest.MonkeyPatch, orm_session: Session, rows: Rows
) -> None:
    vector = MagicMock()
    monkeypatch.setattr(vector_module, "Vector", MagicMock(return_value=vector))
    rows.dataset.indexing_technique = IndexTechniqueType.ECONOMY
    VectorService.update_multimodel_vector(rows.segment, ["new"], rows.dataset, orm_session)
    vector.assert_not_called()
    rows.dataset.indexing_technique = IndexTechniqueType.HIGH_QUALITY
    with patch.object(DocumentSegment, "attachments", new_callable=PropertyMock, return_value=[{"id": "same"}]):
        VectorService.update_multimodel_vector(rows.segment, ["same"], rows.dataset, orm_session)
    vector.create_multimodal.assert_not_called()


def test_update_multimodal_vector_rolls_back_delete_when_insert_fails(
    monkeypatch: pytest.MonkeyPatch,
    orm_session: Session,
    rows: Rows,
) -> None:
    old = _upload(orm_session, name="old.png")
    new = _upload(orm_session, name="new.png")
    binding = SegmentAttachmentBinding(
        tenant_id=rows.dataset.tenant_id,
        dataset_id=rows.dataset.id,
        document_id=rows.document.id,
        segment_id=rows.segment.id,
        attachment_id=old.id,
    )
    orm_session.add(binding)
    rows.dataset.is_multimodal = True
    orm_session.commit()
    binding_id = binding.id
    monkeypatch.setattr(vector_module, "Vector", MagicMock(return_value=MagicMock()))
    with (
        _bind_attachment_property(orm_session),
        _raise_before_commit(orm_session),
        pytest.raises(RuntimeError, match="forced binding commit"),
    ):
        VectorService.update_multimodel_vector(rows.segment, [new.id], rows.dataset, orm_session)
    orm_session.expire_all()
    persisted = orm_session.get(SegmentAttachmentBinding, binding_id)
    assert persisted is not None
    assert persisted.attachment_id == old.id

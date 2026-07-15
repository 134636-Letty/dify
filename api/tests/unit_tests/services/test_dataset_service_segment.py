"""SQLite-backed tests for SegmentService database and transaction behavior.

The service receives a caller-owned SQLAlchemy session. Tests persist the complete dataset,
document, parent-segment, and child-chunk graph; vector, cache, summary, and task dispatch remain
external boundaries and are patched narrowly.
"""

from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass

import pytest
from sqlalchemy import Engine, event, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from models.account import Account, Tenant
from models.base import TypeBase
from models.dataset import ChildChunk, Dataset, Document, DocumentSegment
from models.enums import (
    DataSourceType,
    DocumentCreatedFrom,
    IndexingStatus,
    SegmentStatus,
    SegmentType,
)
from services.dataset_ref_service import DatasetRef, DatasetRefService
from services.dataset_service import SegmentService
from services.entities.knowledge_entities.knowledge_entities import ChildChunkUpdateArgs, SegmentUpdateArgs
from services.errors.chunk import ChildChunkDeleteIndexError, ChildChunkIndexingError


@dataclass(frozen=True)
class _DatasetGraph:
    session: Session
    dataset: Dataset
    document: Document
    segment: DocumentSegment
    rollback_events: list[bool]


@pytest.fixture
def dataset_graph(sqlite_engine: Engine) -> Iterator[_DatasetGraph]:
    """Persist one tenant graph plus cross-tenant decoys in isolated SQLite."""

    TypeBase.metadata.create_all(
        sqlite_engine,
        tables=[Dataset.__table__, Document.__table__, DocumentSegment.__table__, ChildChunk.__table__],
    )
    factory = sessionmaker(bind=sqlite_engine, expire_on_commit=False)
    rollback_events: list[bool] = []

    def _record_rollback(session: Session) -> None:
        if session.get_bind() is sqlite_engine:
            rollback_events.append(True)

    event.listen(Session, "after_rollback", _record_rollback)
    try:
        with factory() as session:
            dataset = _dataset()
            document = _document(dataset)
            segment = _segment(dataset, document)
            decoy_dataset = _dataset(dataset_id="dataset-2", tenant_id="tenant-2")
            decoy_document = _document(decoy_dataset, document_id="doc-2")
            decoy_segment = _segment(decoy_dataset, decoy_document, segment_id="segment-2")
            session.add_all([dataset, document, segment, decoy_dataset, decoy_document, decoy_segment])
            session.commit()
            yield _DatasetGraph(
                session=session,
                dataset=dataset,
                document=document,
                segment=segment,
                rollback_events=rollback_events,
            )
    finally:
        event.remove(Session, "after_rollback", _record_rollback)


@pytest.fixture
def account_context(monkeypatch: pytest.MonkeyPatch) -> Account:
    """Bind service ownership checks to a real account/tenant model pair."""

    import services.dataset_service as dataset_service_module

    account = Account(name="User", email="user@example.com")
    account._current_tenant = Tenant(name="Tenant")
    account._current_tenant.id = "tenant-1"
    monkeypatch.setattr(dataset_service_module, "current_user", account)
    return account


def _dataset(*, dataset_id: str = "dataset-1", tenant_id: str = "tenant-1") -> Dataset:
    return Dataset(
        id=dataset_id,
        tenant_id=tenant_id,
        name=f"Dataset {dataset_id}",
        indexing_technique=IndexTechniqueType.ECONOMY,
        created_by="user-1",
    )


def _document(dataset: Dataset, *, document_id: str = "doc-1") -> Document:
    return Document(
        id=document_id,
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        batch="batch-1",
        name="document.txt",
        created_from=DocumentCreatedFrom.API,
        created_by="user-1",
        word_count=7,
        tokens=2,
        indexing_status=IndexingStatus.COMPLETED,
        enabled=True,
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
    )


def _segment(
    dataset: Dataset,
    document: Document,
    *,
    segment_id: str = "segment-1",
    position: int = 1,
) -> DocumentSegment:
    segment = DocumentSegment(
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        position=position,
        content=f"segment {position}",
        word_count=7,
        tokens=2,
        created_by="user-1",
        status=SegmentStatus.COMPLETED,
        index_node_id=f"node-{segment_id}",
    )
    segment.id = segment_id
    return segment


def _child(graph: _DatasetGraph, *, child_id: str, position: int, content: str = "child") -> ChildChunk:
    child = ChildChunk(
        tenant_id=graph.dataset.tenant_id,
        dataset_id=graph.dataset.id,
        document_id=graph.document.id,
        segment_id=graph.segment.id,
        position=position,
        content=content,
        word_count=len(content),
        created_by="user-1",
        type=SegmentType.CUSTOMIZED,
        index_node_id=f"node-{child_id}",
    )
    child.id = child_id
    return child


def _patch_lock_and_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    import services.dataset_service as dataset_service_module

    monkeypatch.setattr(dataset_service_module.redis_client, "lock", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(dataset_service_module.uuid, "uuid4", lambda: "node-new")
    monkeypatch.setattr(dataset_service_module.helper, "generate_text_hash", lambda _content: "hash-new")


class TestDatasetRefService:
    def test_refs_reject_cross_dataset_document(self, dataset_graph: _DatasetGraph) -> None:
        dataset_ref = DatasetRefService.create_dataset_ref(dataset_graph.dataset)
        assert dataset_ref == DatasetRef("tenant-1", "dataset-1")

        cross_tenant_document = dataset_graph.session.get(Document, "doc-2")
        assert cross_tenant_document is not None
        assert DatasetRefService.create_document_ref(dataset_ref, cross_tenant_document) is None

    def test_segment_ref_carries_full_parent_chain(self, dataset_graph: _DatasetGraph) -> None:
        dataset_ref = DatasetRefService.create_dataset_ref(dataset_graph.dataset)
        document_ref = DatasetRefService.create_document_ref(dataset_ref, dataset_graph.document)
        assert document_ref is not None
        segment_ref = DatasetRefService.create_segment_ref(document_ref, dataset_graph.segment.id)
        assert tuple(segment_ref) == ("tenant-1", "dataset-1", "doc-1", "segment-1")


class TestChildChunkTransactions:
    def test_create_assigns_next_position_and_commits(
        self,
        dataset_graph: _DatasetGraph,
        account_context: Account,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import services.dataset_service as dataset_service_module

        existing = _child(dataset_graph, child_id="child-1", position=2)
        dataset_graph.session.add(existing)
        dataset_graph.session.commit()
        _patch_lock_and_hash(monkeypatch)
        vector_calls: list[str] = []
        monkeypatch.setattr(
            dataset_service_module.VectorService,
            "create_child_chunk_vector",
            lambda child, dataset: vector_calls.append(child.id),
        )

        child = SegmentService.create_child_chunk(
            "new child",
            dataset_graph.segment,
            dataset_graph.document,
            dataset_graph.dataset,
            dataset_graph.session,
        )

        assert child.position == 3
        assert child.index_node_id == "node-new"
        assert child.index_node_hash == "hash-new"
        assert dataset_graph.session.get(ChildChunk, child.id) is child
        assert vector_calls == [child.id]

    def test_create_rolls_back_vector_failure(
        self,
        dataset_graph: _DatasetGraph,
        account_context: Account,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import services.dataset_service as dataset_service_module

        _patch_lock_and_hash(monkeypatch)
        before = dataset_graph.session.scalar(select(func.count(ChildChunk.id)))

        def _raise(*_args, **_kwargs) -> None:
            raise RuntimeError("vector failed")

        monkeypatch.setattr(dataset_service_module.VectorService, "create_child_chunk_vector", _raise)
        with pytest.raises(ChildChunkIndexingError, match="vector failed"):
            SegmentService.create_child_chunk(
                "new child",
                dataset_graph.segment,
                dataset_graph.document,
                dataset_graph.dataset,
                dataset_graph.session,
            )

        assert dataset_graph.rollback_events == [True]
        assert dataset_graph.session.scalar(select(func.count(ChildChunk.id))) == before

    def test_update_children_persists_update_delete_and_create(
        self,
        dataset_graph: _DatasetGraph,
        account_context: Account,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import services.dataset_service as dataset_service_module

        first = _child(dataset_graph, child_id="child-a", position=1, content="old")
        removed = _child(dataset_graph, child_id="child-b", position=2, content="remove")
        dataset_graph.session.add_all([first, removed])
        dataset_graph.session.commit()
        _patch_lock_and_hash(monkeypatch)
        monkeypatch.setattr(dataset_service_module.VectorService, "update_child_chunk_vector", lambda *_args: None)

        result = SegmentService.update_child_chunks(
            [
                ChildChunkUpdateArgs(id="child-a", content="updated"),
                ChildChunkUpdateArgs(content="brand new"),
            ],
            dataset_graph.segment,
            dataset_graph.document,
            dataset_graph.dataset,
            dataset_graph.session,
        )

        assert [chunk.position for chunk in result] == [1, 3]
        assert dataset_graph.session.get(ChildChunk, "child-a").content == "updated"
        assert dataset_graph.session.get(ChildChunk, "child-b") is None
        assert dataset_graph.session.scalar(select(func.count(ChildChunk.id))) == 2

    def test_update_children_rolls_back_all_changes_on_vector_failure(
        self,
        dataset_graph: _DatasetGraph,
        account_context: Account,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import services.dataset_service as dataset_service_module

        child = _child(dataset_graph, child_id="child-a", position=1, content="old")
        dataset_graph.session.add(child)
        dataset_graph.session.commit()
        monkeypatch.setattr(
            dataset_service_module.VectorService,
            "update_child_chunk_vector",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("vector failed")),
        )

        with pytest.raises(ChildChunkIndexingError, match="vector failed"):
            SegmentService.update_child_chunks(
                [ChildChunkUpdateArgs(id="child-a", content="changed")],
                dataset_graph.segment,
                dataset_graph.document,
                dataset_graph.dataset,
                dataset_graph.session,
            )

        dataset_graph.session.expire_all()
        assert dataset_graph.session.get(ChildChunk, "child-a").content == "old"
        assert dataset_graph.rollback_events == [True]

    def test_delete_rolls_back_vector_failure_and_commits_success(
        self,
        dataset_graph: _DatasetGraph,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import services.dataset_service as dataset_service_module

        child = _child(dataset_graph, child_id="child-a", position=1)
        dataset_graph.session.add(child)
        dataset_graph.session.commit()

        monkeypatch.setattr(
            dataset_service_module.VectorService,
            "delete_child_chunk_vector",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("delete failed")),
        )
        with pytest.raises(ChildChunkDeleteIndexError, match="delete failed"):
            SegmentService.delete_child_chunk(child, dataset_graph.dataset, dataset_graph.session)
        assert dataset_graph.session.get(ChildChunk, child.id) is not None

        monkeypatch.setattr(dataset_service_module.VectorService, "delete_child_chunk_vector", lambda *_args: None)
        SegmentService.delete_child_chunk(child, dataset_graph.dataset, dataset_graph.session)
        assert dataset_graph.session.get(ChildChunk, child.id) is None

    def test_primary_key_constraint_failure_rolls_back_duplicate(self, dataset_graph: _DatasetGraph) -> None:
        first = _child(dataset_graph, child_id="duplicate", position=1)
        dataset_graph.session.add(first)
        dataset_graph.session.commit()
        dataset_graph.session.expunge(first)
        duplicate = _child(dataset_graph, child_id="duplicate", position=2)
        dataset_graph.session.add(duplicate)
        with pytest.raises(IntegrityError):
            dataset_graph.session.commit()
        dataset_graph.session.rollback()

        assert dataset_graph.session.scalar(select(func.count(ChildChunk.id))) == 1


class TestScopedQueries:
    def test_child_and_segment_lookup_enforce_tenant_and_parent_chain(self, dataset_graph: _DatasetGraph) -> None:
        child = _child(dataset_graph, child_id="child-a", position=1)
        dataset_graph.session.add(child)
        dataset_graph.session.commit()
        dataset_ref = DatasetRefService.create_dataset_ref(dataset_graph.dataset)
        document_ref = DatasetRefService.create_document_ref(dataset_ref, dataset_graph.document)
        assert document_ref is not None
        segment_ref = DatasetRefService.create_segment_ref(document_ref, dataset_graph.segment.id)

        assert SegmentService.get_segment_by_ref(segment_ref, dataset_graph.session).id == dataset_graph.segment.id
        assert (
            SegmentService.get_child_chunk_by_segment_ref(child.id, segment_ref, dataset_graph.session).id == child.id
        )
        assert SegmentService.get_segment_by_id(dataset_graph.segment.id, "tenant-2", dataset_graph.session) is None
        assert SegmentService.get_child_chunk_by_id(child.id, "tenant-2", dataset_graph.session) is None

    def test_segment_collection_filters_document_dataset_status_and_enabled(self, dataset_graph: _DatasetGraph) -> None:
        disabled = _segment(dataset_graph.dataset, dataset_graph.document, segment_id="disabled", position=2)
        disabled.enabled = False
        waiting = _segment(dataset_graph.dataset, dataset_graph.document, segment_id="waiting", position=3)
        waiting.status = SegmentStatus.WAITING
        dataset_graph.session.add_all([disabled, waiting])
        dataset_graph.session.commit()

        rows = SegmentService.get_segments_by_document_and_dataset(
            document_id=dataset_graph.document.id,
            dataset_id=dataset_graph.dataset.id,
            session=dataset_graph.session,
            status=SegmentStatus.COMPLETED,
            enabled=True,
        )
        assert [row.id for row in rows] == [dataset_graph.segment.id]


class TestParentSegmentTransactions:
    def test_multi_create_persists_segments_and_document_word_count(
        self,
        dataset_graph: _DatasetGraph,
        account_context: Account,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import services.dataset_service as dataset_service_module

        _patch_lock_and_hash(monkeypatch)
        monkeypatch.setattr(dataset_service_module.VectorService, "create_segments_vector", lambda *_args: None)
        before_count = dataset_graph.session.scalar(select(func.count(DocumentSegment.id)))

        rows = SegmentService.multi_create_segment(
            [{"content": "first"}, {"content": "second", "keywords": ["key"]}],
            dataset_graph.document,
            dataset_graph.dataset,
            dataset_graph.session,
        )

        assert rows is not None
        assert [row.position for row in rows] == [2, 3]
        assert dataset_graph.session.scalar(select(func.count(DocumentSegment.id))) == before_count + 2
        assert dataset_graph.session.get(Document, dataset_graph.document.id).word_count == 18

    def test_delete_segment_persists_removal_and_dispatches_index_task(
        self,
        dataset_graph: _DatasetGraph,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import services.dataset_service as dataset_service_module

        child = _child(dataset_graph, child_id="child-a", position=1)
        dataset_graph.session.add(child)
        dataset_graph.session.commit()
        monkeypatch.setattr(dataset_service_module.redis_client, "get", lambda _key: None)
        monkeypatch.setattr(dataset_service_module.redis_client, "setex", lambda *_args: None)
        dispatched: list[tuple[object, ...]] = []
        monkeypatch.setattr(
            dataset_service_module.delete_segment_from_index_task,
            "delay",
            lambda *args: dispatched.append(args),
        )

        SegmentService.delete_segment(
            dataset_graph.segment,
            dataset_graph.document,
            dataset_graph.dataset,
            dataset_graph.session,
        )

        assert dataset_graph.session.get(DocumentSegment, dataset_graph.segment.id) is None
        assert dataset_graph.session.get(Document, dataset_graph.document.id).word_count == 0
        assert dispatched
        assert dispatched[0][0] == ["node-segment-1"]

    def test_disable_segment_commits_state_and_dispatches_task(
        self,
        dataset_graph: _DatasetGraph,
        account_context: Account,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import services.dataset_service as dataset_service_module

        monkeypatch.setattr(dataset_service_module.redis_client, "get", lambda _key: None)
        monkeypatch.setattr(dataset_service_module.redis_client, "setex", lambda *_args: None)
        dispatched: list[str] = []
        monkeypatch.setattr(
            dataset_service_module.disable_segment_from_index_task,
            "delay",
            lambda segment_id: dispatched.append(segment_id),
        )

        result = SegmentService.update_segment(
            SegmentUpdateArgs(enabled=False),
            dataset_graph.segment,
            dataset_graph.document,
            dataset_graph.dataset,
            dataset_graph.session,
        )

        assert result.enabled is False
        assert dataset_graph.session.get(DocumentSegment, result.id).enabled is False
        assert dispatched == [result.id]

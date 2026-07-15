"""SQLite-backed unit tests for the dataset document store."""

from collections.abc import Iterator
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from core.rag.docstore import dataset_docstore as docstore_module
from core.rag.docstore.dataset_docstore import DatasetDocumentStore
from core.rag.index_processor.constant.index_type import IndexTechniqueType
from core.rag.models.document import AttachmentDocument, ChildDocument, Document
from models.dataset import (
    ChildChunk,
    Dataset,
    DocumentSegment,
    SegmentAttachmentBinding,
)
from models.dataset import (
    Document as DatasetDocument,
)
from models.enums import DataSourceType, DocumentCreatedFrom


@dataclass(frozen=True)
class DocstoreDatabase:
    session: Session


@pytest.fixture
def docstore_database(sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[DocstoreDatabase]:
    """Bind the docstore's unit of work to an isolated real SQLite session."""
    Dataset.metadata.create_all(
        sqlite_engine,
        tables=[
            Dataset.__table__,
            DatasetDocument.__table__,
            DocumentSegment.__table__,
            ChildChunk.__table__,
            SegmentAttachmentBinding.__table__,
        ],
    )
    session_maker = sessionmaker(bind=sqlite_engine, expire_on_commit=False)
    with session_maker() as session:
        database = DocstoreDatabase(session=session)
        monkeypatch.setattr(docstore_module, "db", database)
        yield database


def _persist_dataset(
    database: DocstoreDatabase,
    *,
    dataset_id: str = "test-dataset-id",
    tenant_id: str = "tenant-1",
    indexing_technique: IndexTechniqueType = IndexTechniqueType.ECONOMY,
) -> Dataset:
    dataset = Dataset(
        id=dataset_id,
        tenant_id=tenant_id,
        name=f"Dataset {dataset_id}",
        created_by="test-user-id",
        indexing_technique=indexing_technique,
        embedding_model_provider="provider" if indexing_technique == IndexTechniqueType.HIGH_QUALITY else None,
        embedding_model="model" if indexing_technique == IndexTechniqueType.HIGH_QUALITY else None,
    )
    database.session.add(dataset)
    database.session.commit()
    return dataset


def _persist_source_document(
    database: DocstoreDatabase,
    *,
    dataset_id: str = "test-dataset-id",
    document_id: str = "test-doc-id",
    tenant_id: str = "tenant-1",
) -> DatasetDocument:
    document = DatasetDocument(
        id=document_id,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        batch="batch-1",
        name="document.txt",
        created_from=DocumentCreatedFrom.API,
        created_by="test-user-id",
    )
    database.session.add(document)
    database.session.commit()
    return document


def _build_store(
    database: DocstoreDatabase,
    *,
    dataset_id: str = "test-dataset-id",
    tenant_id: str = "tenant-1",
    document_id: str | None = "test-doc-id",
    indexing_technique: IndexTechniqueType = IndexTechniqueType.ECONOMY,
) -> DatasetDocumentStore:
    dataset = _persist_dataset(
        database,
        dataset_id=dataset_id,
        tenant_id=tenant_id,
        indexing_technique=indexing_technique,
    )
    if document_id is not None:
        _persist_source_document(
            database,
            dataset_id=dataset_id,
            document_id=document_id,
            tenant_id=tenant_id,
        )
    return DatasetDocumentStore(dataset=dataset, user_id="test-user-id", document_id=document_id)


def _persist_segment(
    database: DocstoreDatabase,
    *,
    dataset_id: str = "test-dataset-id",
    document_id: str = "test-doc-id",
    tenant_id: str = "tenant-1",
    index_node_id: str = "node-1",
    index_node_hash: str = "hash-1",
    content: str = "Test content",
    position: int = 1,
    answer: str | None = None,
) -> DocumentSegment:
    segment = DocumentSegment(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        document_id=document_id,
        position=position,
        content=content,
        word_count=len(content),
        tokens=0,
        created_by="test-user-id",
        index_node_id=index_node_id,
        index_node_hash=index_node_hash,
        answer=answer,
    )
    database.session.add(segment)
    database.session.commit()
    return segment


def _rag_document(
    *,
    content: str = "Test content",
    doc_id: str = "node-1",
    doc_hash: str = "hash-1",
    answer: str | None = None,
    children: list[ChildDocument] | None = None,
    attachments: list[AttachmentDocument] | None = None,
) -> Document:
    metadata = {"doc_id": doc_id, "doc_hash": doc_hash}
    if answer is not None:
        metadata["answer"] = answer
    return Document(page_content=content, metadata=metadata, children=children, attachments=attachments)


class TestDatasetDocumentStoreInit:
    def test_init_with_all_parameters(self) -> None:
        dataset = Dataset(id="test-dataset-id", tenant_id="tenant-1", name="Dataset", created_by="user-1")

        store = DatasetDocumentStore(dataset=dataset, user_id="test-user-id", document_id="test-doc-id")

        assert store._dataset is dataset
        assert store._document_id == "test-doc-id"
        assert store.dataset_id == "test-dataset-id"
        assert store.user_id == "test-user-id"

    def test_init_without_document_id(self) -> None:
        dataset = Dataset(id="test-dataset-id", tenant_id="tenant-1", name="Dataset", created_by="user-1")

        store = DatasetDocumentStore(dataset=dataset, user_id="test-user-id")

        assert store._document_id is None
        assert store.dataset_id == "test-dataset-id"


class TestDatasetDocumentStoreSerialization:
    def test_to_dict_and_from_dict(self) -> None:
        dataset = Dataset(id="ds-123", tenant_id="tenant-1", name="Dataset", created_by="user-1")
        store = DatasetDocumentStore(dataset=dataset, user_id="test-user", document_id="test-doc")

        assert store.to_dict() == {"dataset_id": "ds-123"}
        restored = DatasetDocumentStore.from_dict(
            {"dataset": dataset, "user_id": "test-user", "document_id": "test-doc"}
        )
        assert restored._dataset is dataset
        assert restored._document_id == "test-doc"


class TestDatasetDocumentStoreDocs:
    def test_docs_returns_only_dataset_segments(self, docstore_database: DocstoreDatabase) -> None:
        store = _build_store(docstore_database)
        _persist_segment(docstore_database)
        _persist_dataset(docstore_database, dataset_id="other-dataset", tenant_id="tenant-2")
        _persist_source_document(
            docstore_database,
            dataset_id="other-dataset",
            document_id="other-document",
            tenant_id="tenant-2",
        )
        _persist_segment(
            docstore_database,
            dataset_id="other-dataset",
            document_id="other-document",
            tenant_id="tenant-2",
            index_node_id="other-node",
        )

        result = store.docs

        assert list(result) == ["node-1"]
        assert result["node-1"].page_content == "Test content"
        assert result["node-1"].metadata["dataset_id"] == "test-dataset-id"

    def test_docs_empty_dataset(self, docstore_database: DocstoreDatabase) -> None:
        store = _build_store(docstore_database)

        assert store.docs == {}


class TestDatasetDocumentStoreAddDocuments:
    def test_add_documents_new_document_with_embedding(
        self,
        monkeypatch: pytest.MonkeyPatch,
        docstore_database: DocstoreDatabase,
    ) -> None:
        store = _build_store(docstore_database, indexing_technique=IndexTechniqueType.HIGH_QUALITY)
        model_instance = MagicMock()
        model_instance.get_text_embedding_num_tokens.return_value = [10]
        manager = MagicMock()
        manager.get_model_instance.return_value = model_instance
        monkeypatch.setattr(docstore_module.ModelManager, "for_tenant", MagicMock(return_value=manager))

        store.add_documents([_rag_document()])

        segment = docstore_database.session.scalar(select(DocumentSegment))
        assert segment is not None
        assert segment.index_node_id == "node-1"
        assert segment.tokens == 10
        assert segment.enabled is False
        assert segment.position == 1

    def test_add_documents_updates_existing_document(self, docstore_database: DocstoreDatabase) -> None:
        store = _build_store(docstore_database)
        existing = _persist_segment(docstore_database, position=5, content="Old", index_node_hash="old-hash")

        store.add_documents([_rag_document(content="Updated content", doc_hash="new-hash", answer="Updated answer")])

        docstore_database.session.expire_all()
        updated = docstore_database.session.get(DocumentSegment, existing.id)
        assert updated is not None
        assert updated.content == "Updated content"
        assert updated.index_node_hash == "new-hash"
        assert updated.answer == "Updated answer"
        assert updated.position == 5

    def test_add_documents_raises_without_update_and_preserves_state(
        self,
        docstore_database: DocstoreDatabase,
    ) -> None:
        store = _build_store(docstore_database)
        existing = _persist_segment(docstore_database, content="Original")

        with pytest.raises(ValueError, match="already exists"):
            store.add_documents([_rag_document(content="Replacement")], allow_update=False)

        docstore_database.session.rollback()
        docstore_database.session.expire_all()
        assert docstore_database.session.get(DocumentSegment, existing.id).content == "Original"

    def test_add_documents_with_answer_metadata(self, docstore_database: DocstoreDatabase) -> None:
        store = _build_store(docstore_database)

        store.add_documents([_rag_document(answer="Test answer")])

        segment = docstore_database.session.scalar(select(DocumentSegment))
        assert segment is not None
        assert segment.answer == "Test answer"

    def test_add_documents_rejects_invalid_document_type(self, docstore_database: DocstoreDatabase) -> None:
        store = _build_store(docstore_database)

        with pytest.raises(ValueError, match="must be a Document"):
            store.add_documents(["not a document"])  # type: ignore[list-item]

        docstore_database.session.rollback()
        assert docstore_database.session.scalar(select(DocumentSegment)) is None

    def test_add_documents_rejects_none_metadata(self, docstore_database: DocstoreDatabase) -> None:
        store = _build_store(docstore_database)
        document = _rag_document()
        document.metadata = None  # type: ignore[assignment]

        with pytest.raises(ValueError, match="metadata must be a dict"):
            store.add_documents([document])

        docstore_database.session.rollback()
        assert docstore_database.session.scalar(select(DocumentSegment)) is None

    def test_add_documents_persists_child_chunks(self, docstore_database: DocstoreDatabase) -> None:
        store = _build_store(docstore_database)
        child = ChildDocument(page_content="Child content", metadata={"doc_id": "child-1", "doc_hash": "child-hash"})

        store.add_documents([_rag_document(children=[child])], save_child=True)

        child_row = docstore_database.session.scalar(select(ChildChunk))
        assert child_row is not None
        assert child_row.content == "Child content"
        assert child_row.index_node_id == "child-1"

    def test_add_documents_rolls_back_flushed_segment_when_binding_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        docstore_database: DocstoreDatabase,
    ) -> None:
        store = _build_store(docstore_database)
        monkeypatch.setattr(store, "add_multimodel_documents_binding", MagicMock(side_effect=RuntimeError("boom")))

        with pytest.raises(RuntimeError, match="boom"):
            store.add_documents([_rag_document()])

        assert docstore_database.session.in_transaction()
        docstore_database.session.rollback()
        assert docstore_database.session.scalar(select(DocumentSegment)) is None

    def test_update_replaces_existing_children(self, docstore_database: DocstoreDatabase) -> None:
        store = _build_store(docstore_database)
        segment = _persist_segment(docstore_database)
        old_child = ChildChunk(
            tenant_id="tenant-1",
            dataset_id="test-dataset-id",
            document_id="test-doc-id",
            segment_id=segment.id,
            position=1,
            content="Old child",
            word_count=2,
            created_by="test-user-id",
            index_node_id="old-child",
        )
        docstore_database.session.add(old_child)
        docstore_database.session.commit()
        new_child = ChildDocument(page_content="New child", metadata={"doc_id": "new-child", "doc_hash": "new"})

        store.add_documents([_rag_document(content="Updated", children=[new_child])], save_child=True)

        children = docstore_database.session.scalars(select(ChildChunk)).all()
        assert [(child.index_node_id, child.content) for child in children] == [("new-child", "New child")]


class TestDatasetDocumentStoreLookup:
    def test_exists_get_and_hash_use_persisted_segment(self, docstore_database: DocstoreDatabase) -> None:
        store = _build_store(docstore_database)
        _persist_segment(docstore_database, index_node_hash="test-hash")

        assert store.document_exists("node-1") is True
        result = store.get_document("node-1", raise_error=False)
        assert result is not None
        assert result.page_content == "Test content"
        assert store.get_document_hash("node-1") == "test-hash"
        assert store.get_document_segment("node-1") is not None

    def test_lookup_is_dataset_scoped(self, docstore_database: DocstoreDatabase) -> None:
        store = _build_store(docstore_database)
        _persist_dataset(docstore_database, dataset_id="other-dataset", tenant_id="tenant-2")
        _persist_source_document(
            docstore_database,
            dataset_id="other-dataset",
            document_id="other-document",
            tenant_id="tenant-2",
        )
        _persist_segment(
            docstore_database,
            dataset_id="other-dataset",
            document_id="other-document",
            tenant_id="tenant-2",
            index_node_id="node-1",
        )

        assert store.document_exists("node-1") is False
        assert store.get_document("node-1", raise_error=False) is None

    def test_missing_document_raises_when_requested(self, docstore_database: DocstoreDatabase) -> None:
        store = _build_store(docstore_database)

        with pytest.raises(ValueError, match="not found"):
            store.get_document("missing", raise_error=True)

    def test_set_document_hash_persists(self, docstore_database: DocstoreDatabase) -> None:
        store = _build_store(docstore_database)
        segment = _persist_segment(docstore_database, index_node_hash="old-hash")

        store.set_document_hash("node-1", "new-hash")

        docstore_database.session.expire_all()
        assert docstore_database.session.get(DocumentSegment, segment.id).index_node_hash == "new-hash"

    def test_missing_hash_update_returns_none(self, docstore_database: DocstoreDatabase) -> None:
        store = _build_store(docstore_database)

        assert store.set_document_hash("missing", "new-hash") is None
        assert store.get_document_hash("missing") is None


class TestDatasetDocumentStoreDeleteDocument:
    def test_delete_document_commits_only_scoped_segment(self, docstore_database: DocstoreDatabase) -> None:
        store = _build_store(docstore_database)
        deleted = _persist_segment(docstore_database, index_node_id="shared-node")
        _persist_dataset(docstore_database, dataset_id="other-dataset", tenant_id="tenant-2")
        _persist_source_document(
            docstore_database,
            dataset_id="other-dataset",
            document_id="other-document",
            tenant_id="tenant-2",
        )
        retained = _persist_segment(
            docstore_database,
            dataset_id="other-dataset",
            document_id="other-document",
            tenant_id="tenant-2",
            index_node_id="shared-node",
        )

        store.delete_document("shared-node")

        assert docstore_database.session.get(DocumentSegment, deleted.id) is None
        assert docstore_database.session.get(DocumentSegment, retained.id) is not None

    def test_delete_missing_returns_none(self, docstore_database: DocstoreDatabase) -> None:
        store = _build_store(docstore_database)

        assert store.delete_document("missing", raise_error=False) is None

    def test_delete_missing_raises(self, docstore_database: DocstoreDatabase) -> None:
        store = _build_store(docstore_database)

        with pytest.raises(ValueError, match="not found"):
            store.delete_document("missing", raise_error=True)


class TestDatasetDocumentStoreMultimodelBinding:
    def test_adds_persisted_attachment_binding(self, docstore_database: DocstoreDatabase) -> None:
        store = _build_store(docstore_database)
        segment = _persist_segment(docstore_database)
        attachment = AttachmentDocument(page_content="image", metadata={"doc_id": "attachment-1"})

        store.add_multimodel_documents_binding(segment.id, [attachment])
        docstore_database.session.flush()

        binding = docstore_database.session.scalar(select(SegmentAttachmentBinding))
        assert binding is not None
        assert binding.segment_id == segment.id
        assert binding.tenant_id == "tenant-1"
        assert binding.attachment_id == "attachment-1"

    @pytest.mark.parametrize("attachments", [None, []], ids=["none", "empty"])
    def test_skips_missing_attachments(
        self,
        docstore_database: DocstoreDatabase,
        attachments: list[AttachmentDocument] | None,
    ) -> None:
        store = _build_store(docstore_database)

        store.add_multimodel_documents_binding("segment-1", attachments)
        docstore_database.session.flush()

        assert docstore_database.session.scalar(select(SegmentAttachmentBinding)) is None

    def test_skips_binding_without_source_document_id(self, docstore_database: DocstoreDatabase) -> None:
        store = _build_store(docstore_database, document_id=None)
        attachment = AttachmentDocument(page_content="image", metadata={"doc_id": "attachment-1"})

        store.add_multimodel_documents_binding("segment-1", [attachment])
        docstore_database.session.flush()

        assert docstore_database.session.scalar(select(SegmentAttachmentBinding)) is None

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, Mock, call, patch
from uuid import uuid4

import pytest
from sqlalchemy import Engine, event, select
from sqlalchemy.orm import Session, sessionmaker

from core.rag.datasource import retrieval_service as retrieval_service_module
from core.rag.datasource.retrieval_service import RetrievalService
from core.rag.index_processor.constant.doc_type import DocType
from core.rag.index_processor.constant.index_type import IndexStructureType
from core.rag.index_processor.constant.query_type import QueryType
from core.rag.models.document import Document
from core.rag.rerank.rerank_type import RerankMode
from core.rag.retrieval.retrieval_methods import RetrievalMethod
from extensions.storage.storage_type import StorageType
from models.dataset import (
    ChildChunk,
    Dataset,
    DocumentSegment,
    DocumentSegmentSummary,
    SegmentAttachmentBinding,
)
from models.dataset import Document as DatasetDocument
from models.enums import (
    CreatorUserRole,
    DataSourceType,
    DocumentCreatedFrom,
    IndexingStatus,
    SegmentStatus,
    SummaryStatus,
)
from models.model import UploadFile


def create_mock_document(
    content: str,
    doc_id: str,
    score: float = 0.8,
    provider: str = "dify",
    additional_metadata: dict[str, Any] | None = None,
) -> Document:
    """
    Create a mock Document object for testing.

    This helper function standardizes document creation across tests,
    ensuring consistent structure and reducing code duplication.

    Args:
        content: The text content of the document
        doc_id: Unique identifier for the document chunk
        score: Relevance score (0.0 to 1.0)
        provider: Document provider ("dify" or "external")
        additional_metadata: Optional extra metadata fields

    Returns:
        Document: A properly structured Document object

    Example:
        >>> doc = create_mock_document("Python is great", "doc1", score=0.95)
        >>> assert doc.metadata["score"] == 0.95
    """
    metadata = {
        "doc_id": doc_id,
        "document_id": str(uuid4()),
        "dataset_id": str(uuid4()),
        "score": score,
    }

    # Merge additional metadata if provided
    if additional_metadata:
        metadata.update(additional_metadata)

    return Document(
        page_content=content,
        metadata=metadata,
        provider=provider,
    )


class _ImmediateFuture:
    def __init__(self, exception: Exception | None = None) -> None:
        self._exception = exception
        self.cancel_called = False

    def exception(self) -> Exception | None:
        return self._exception

    def cancel(self) -> None:
        self.cancel_called = True


class _ImmediateExecutor:
    def __init__(self) -> None:
        self.futures: list[_ImmediateFuture] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def submit(self, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
            future = _ImmediateFuture()
        except Exception as exc:  # pragma: no cover - only for defensive parity with Future semantics
            future = _ImmediateFuture(exc)
        self.futures.append(future)
        return future


@dataclass(frozen=True)
class Database:
    """Typed subset of Flask-SQLAlchemy used by retrieval code under test."""

    engine: Engine
    session: Session


@pytest.fixture
def database(sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[Database]:
    tables = [
        Dataset.__table__,
        DatasetDocument.__table__,
        DocumentSegment.__table__,
        ChildChunk.__table__,
        DocumentSegmentSummary.__table__,
        UploadFile.__table__,
        SegmentAttachmentBinding.__table__,
    ]
    Dataset.metadata.create_all(sqlite_engine, tables=tables)
    session_factory = sessionmaker(bind=sqlite_engine, expire_on_commit=False)
    with session_factory() as session:
        database = Database(engine=sqlite_engine, session=session)
        monkeypatch.setattr(retrieval_service_module, "db", database)
        monkeypatch.setattr(retrieval_service_module.session_factory, "create_session", session_factory)
        yield database


def _persist_dataset(session: Session, *, dataset_id: str = "dataset-id", tenant_id: str = "tenant-id") -> Dataset:
    dataset = Dataset(
        id=dataset_id,
        tenant_id=tenant_id,
        name=f"Dataset {dataset_id}",
        description="Retrieval fixture",
        provider="external",
        created_by="user-id",
        maintainer="user-id",
        chunk_structure=IndexStructureType.PARENT_CHILD_INDEX,
        is_multimodal=False,
    )
    session.add(dataset)
    session.commit()
    return dataset


def _persist_dataset_document(
    session: Session,
    *,
    document_id: str,
    doc_form: IndexStructureType,
    dataset_id: str = "dataset-id",
    tenant_id: str = "tenant-id",
) -> DatasetDocument:
    document = DatasetDocument(
        id=document_id,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        data_source_info=None,
        batch="batch-1",
        name=f"Document {document_id}",
        created_from=DocumentCreatedFrom.WEB,
        created_by="user-id",
        indexing_status=IndexingStatus.COMPLETED,
        enabled=True,
        archived=False,
        doc_metadata=None,
        doc_form=doc_form,
        need_summary=False,
    )
    session.add(document)
    return document


def _persist_upload_file(
    session: Session,
    *,
    upload_id: str,
    tenant_id: str = "tenant-id",
    extension: str = "png",
    size: int = 42,
) -> UploadFile:
    upload_file = UploadFile(
        tenant_id=tenant_id,
        storage_type=StorageType.LOCAL,
        key=f"files/{upload_id}",
        name=f"file-{upload_id}",
        size=size,
        extension=extension,
        mime_type=f"image/{extension}",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by="user-id",
        created_at=datetime.now(UTC).replace(tzinfo=None),
        used=False,
    )
    upload_file.id = upload_id
    session.add(upload_file)
    session.commit()
    return upload_file


class _SimpleRetrievalChildChunk:
    def __init__(self, id: str, content: str, score: float, position: int) -> None:
        self.id = id
        self.content = content
        self.score = score
        self.position = position


class _SimpleRetrievalSegment:
    def __init__(
        self,
        segment,
        child_chunks: list[_SimpleRetrievalChildChunk] | None = None,
        score: float | None = None,
        files: list[dict[str, str | int]] | None = None,
        summary: str | None = None,
    ) -> None:
        self.segment = segment
        self.child_chunks = child_chunks
        self.score = score
        self.files = files
        self.summary = summary


class TestRetrievalServiceInternals:
    @pytest.fixture
    def internal_dataset(self) -> Dataset:
        return Dataset(
            id="dataset-id",
            tenant_id="tenant-id",
            name="Internal dataset",
            description="Retrieval fixture",
            provider="vendor",
            created_by="user-id",
            maintainer="user-id",
            chunk_structure=IndexStructureType.PARENT_CHILD_INDEX,
            is_multimodal=False,
        )

    @pytest.fixture
    def internal_flask_app(self):
        app = MagicMock()
        app.app_context.return_value.__enter__ = Mock()
        app.app_context.return_value.__exit__.return_value = False
        return app

    def test_retrieve_with_attachment_ids_only(self, monkeypatch: pytest.MonkeyPatch, internal_dataset):
        with (
            patch("core.rag.datasource.retrieval_service.RetrievalService._get_dataset", return_value=internal_dataset),
            patch("core.rag.datasource.retrieval_service.RetrievalService._retrieve") as mock_retrieve,
        ):
            executor = _ImmediateExecutor()
            monkeypatch.setattr(retrieval_service_module, "ThreadPoolExecutor", lambda *args, **kwargs: executor)
            monkeypatch.setattr(
                retrieval_service_module.concurrent.futures,
                "as_completed",
                lambda futures, timeout=None: iter(futures),
            )

            def side_effect(
                flask_app,
                retrieval_method,
                dataset,
                all_documents,
                exceptions,
                query=None,
                top_k=4,
                score_threshold=0.0,
                reranking_model=None,
                reranking_mode="reranking_model",
                weights=None,
                document_ids_filter=None,
                attachment_id=None,
            ):
                all_documents.append(create_mock_document(f"content-{attachment_id}", attachment_id or "none", 0.9))

            mock_retrieve.side_effect = side_effect

            results = RetrievalService.retrieve(
                retrieval_method=RetrievalMethod.SEMANTIC_SEARCH,
                dataset_id=internal_dataset.id,
                query="",
                attachment_ids=["att-1", "att-2"],
            )

        assert len(results) == 2
        assert {doc.metadata["doc_id"] for doc in results} == {"att-1", "att-2"}
        assert mock_retrieve.call_count == 2

    @patch("core.rag.datasource.retrieval_service.ExternalDatasetService.fetch_external_knowledge_retrieval")
    @patch("core.rag.datasource.retrieval_service.MetadataFilteringCondition.model_validate")
    def test_external_retrieve_with_metadata_conditions(self, mock_validate, mock_fetch, database: Database):
        dataset = _persist_dataset(database.session, dataset_id="dataset-1", tenant_id="tenant-1")
        mock_validate.return_value = "validated-condition"
        expected_documents = [create_mock_document("external-doc", "external-1", 0.8, provider="external")]
        mock_fetch.return_value = expected_documents

        results = RetrievalService.external_retrieve(
            session=database.session,
            dataset_id=dataset.id,
            query="test query",
            external_retrieval_model={"top_k": 3},
            metadata_filtering_conditions={"field": "source", "operator": "contains", "value": "manual"},
        )

        assert results == expected_documents
        mock_validate.assert_called_once()
        mock_fetch.assert_called_once_with(
            tenant_id="tenant-1",
            dataset_id="dataset-1",
            query="test query",
            external_retrieval_parameters={"top_k": 3},
            metadata_condition="validated-condition",
            session=database.session,
        )

    def test_external_retrieve_returns_empty_when_dataset_not_found(self, database: Database):
        _persist_dataset(database.session, dataset_id="other-dataset", tenant_id="other-tenant")
        results = RetrievalService.external_retrieve(session=database.session, dataset_id="missing", query="q")

        assert results == []

    def test_get_dataset_queries_by_id(self, database: Database):
        expected_dataset = _persist_dataset(database.session, dataset_id="dataset-123", tenant_id="tenant-1")
        _persist_dataset(database.session, dataset_id="dataset-other", tenant_id="tenant-2")

        result = RetrievalService._get_dataset("dataset-123")

        assert result is not None
        assert (result.id, result.tenant_id) == (expected_dataset.id, "tenant-1")

    def test_get_dataset_returns_none_for_empty_result(self, database: Database):
        assert RetrievalService._get_dataset("missing") is None

    @patch("core.rag.datasource.retrieval_service.Keyword")
    @patch("core.rag.datasource.retrieval_service.RetrievalService._get_dataset")
    def test_keyword_search_success(self, mock_get_dataset, mock_keyword_class, internal_dataset, internal_flask_app):
        mock_get_dataset.return_value = internal_dataset
        keyword_instance = Mock()
        keyword_instance.search.return_value = [create_mock_document("keyword-content", "kw-1", 0.91)]
        mock_keyword_class.return_value = keyword_instance
        all_documents: list[Document] = []
        exceptions: list[str] = []

        RetrievalService.keyword_search(
            flask_app=internal_flask_app,
            dataset_id=internal_dataset.id,
            query='query "with quotes"',
            top_k=5,
            all_documents=all_documents,
            exceptions=exceptions,
        )

        assert len(all_documents) == 1
        assert exceptions == []
        keyword_instance.search.assert_called_once()

    @patch("core.rag.datasource.retrieval_service.RetrievalService._get_dataset")
    def test_keyword_search_appends_exception_when_dataset_missing(self, mock_get_dataset, internal_flask_app):
        mock_get_dataset.return_value = None
        all_documents: list[Document] = []
        exceptions: list[str] = []

        RetrievalService.keyword_search(
            flask_app=internal_flask_app,
            dataset_id="dataset-id",
            query="query",
            top_k=2,
            all_documents=all_documents,
            exceptions=exceptions,
        )

        assert all_documents == []
        assert exceptions == ["dataset not found"]

    @patch("core.rag.datasource.retrieval_service.Keyword")
    @patch("core.rag.datasource.retrieval_service.RetrievalService._get_dataset")
    def test_keyword_search_appends_exception_when_search_fails(
        self, mock_get_dataset, mock_keyword_class, internal_dataset, internal_flask_app
    ):
        mock_get_dataset.return_value = internal_dataset
        keyword_instance = Mock()
        keyword_instance.search.side_effect = RuntimeError("keyword failed")
        mock_keyword_class.return_value = keyword_instance
        all_documents: list[Document] = []
        exceptions: list[str] = []

        RetrievalService.keyword_search(
            flask_app=internal_flask_app,
            dataset_id=internal_dataset.id,
            query="query",
            top_k=2,
            all_documents=all_documents,
            exceptions=exceptions,
        )

        assert all_documents == []
        assert exceptions == ["keyword failed"]

    @patch("core.rag.datasource.retrieval_service.Vector")
    @patch("core.rag.datasource.retrieval_service.RetrievalService._get_dataset")
    def test_embedding_search_text_without_reranking(
        self, mock_get_dataset, mock_vector_class, internal_dataset, internal_flask_app
    ):
        internal_dataset.is_multimodal = False
        mock_get_dataset.return_value = internal_dataset
        vector_instance = Mock()
        vector_instance.search_by_vector.return_value = [create_mock_document("vector-content", "vec-1", 0.7)]
        mock_vector_class.return_value = vector_instance
        all_documents: list[Document] = []
        exceptions: list[str] = []

        RetrievalService.embedding_search(
            flask_app=internal_flask_app,
            dataset_id=internal_dataset.id,
            query="query",
            top_k=4,
            score_threshold=0.5,
            reranking_model=None,
            all_documents=all_documents,
            retrieval_method=RetrievalMethod.SEMANTIC_SEARCH,
            exceptions=exceptions,
            document_ids_filter=["doc-1"],
            query_type=QueryType.TEXT_QUERY,
        )

        assert len(all_documents) == 1
        assert exceptions == []
        vector_instance.search_by_vector.assert_called_once()

    @patch("core.rag.datasource.retrieval_service.Vector")
    @patch("core.rag.datasource.retrieval_service.RetrievalService._get_dataset")
    def test_embedding_search_image_non_multimodal_returns_early(
        self, mock_get_dataset, mock_vector_class, internal_dataset, internal_flask_app
    ):
        internal_dataset.is_multimodal = False
        mock_get_dataset.return_value = internal_dataset
        vector_instance = Mock()
        mock_vector_class.return_value = vector_instance
        all_documents: list[Document] = []
        exceptions: list[str] = []

        RetrievalService.embedding_search(
            flask_app=internal_flask_app,
            dataset_id=internal_dataset.id,
            query="file-1",
            top_k=4,
            score_threshold=0.5,
            reranking_model=None,
            all_documents=all_documents,
            retrieval_method=RetrievalMethod.SEMANTIC_SEARCH,
            exceptions=exceptions,
            query_type=QueryType.IMAGE_QUERY,
        )

        assert all_documents == []
        assert exceptions == []
        vector_instance.search_by_file.assert_not_called()

    @patch("core.rag.datasource.retrieval_service.ModelManager.for_tenant")
    @patch("core.rag.datasource.retrieval_service.DataPostProcessor")
    @patch("core.rag.datasource.retrieval_service.Vector")
    @patch("core.rag.datasource.retrieval_service.RetrievalService._get_dataset")
    def test_embedding_search_image_multimodal_with_vision_reranking(
        self,
        mock_get_dataset,
        mock_vector_class,
        mock_processor_class,
        mock_model_manager_class,
        internal_dataset,
        internal_flask_app,
    ):
        internal_dataset.is_multimodal = True
        mock_get_dataset.return_value = internal_dataset
        original_docs = [create_mock_document("image-content", "img-doc", 0.73)]
        reranked_docs = [create_mock_document("image-content-reranked", "img-doc", 0.97)]

        vector_instance = Mock()
        vector_instance.search_by_file.return_value = original_docs
        mock_vector_class.return_value = vector_instance

        processor_instance = Mock()
        processor_instance.invoke.return_value = reranked_docs
        mock_processor_class.return_value = processor_instance

        model_manager = Mock()
        model_manager.check_model_support_vision.return_value = True
        mock_model_manager_class.return_value = model_manager

        all_documents: list[Document] = []
        exceptions: list[str] = []

        RetrievalService.embedding_search(
            flask_app=internal_flask_app,
            dataset_id=internal_dataset.id,
            query="file-id",
            top_k=4,
            score_threshold=0.5,
            reranking_model={
                "reranking_provider_name": "provider",
                "reranking_model_name": "model",
            },
            all_documents=all_documents,
            retrieval_method=RetrievalMethod.SEMANTIC_SEARCH,
            exceptions=exceptions,
            query_type=QueryType.IMAGE_QUERY,
        )

        assert all_documents == reranked_docs
        assert exceptions == []
        processor_instance.invoke.assert_called_once()
        mock_model_manager_class.assert_called_once_with(tenant_id=internal_dataset.tenant_id)
        model_manager.check_model_support_vision.assert_called_once()

    @patch("core.rag.datasource.retrieval_service.ModelManager.for_tenant")
    @patch("core.rag.datasource.retrieval_service.DataPostProcessor")
    @patch("core.rag.datasource.retrieval_service.Vector")
    @patch("core.rag.datasource.retrieval_service.RetrievalService._get_dataset")
    def test_embedding_search_image_multimodal_without_vision_support(
        self,
        mock_get_dataset,
        mock_vector_class,
        mock_processor_class,
        mock_model_manager_class,
        internal_dataset,
        internal_flask_app,
    ):
        internal_dataset.is_multimodal = True
        mock_get_dataset.return_value = internal_dataset
        original_docs = [create_mock_document("image-content", "img-doc", 0.73)]

        vector_instance = Mock()
        vector_instance.search_by_file.return_value = original_docs
        mock_vector_class.return_value = vector_instance

        processor_instance = Mock()
        processor_instance.invoke.return_value = [create_mock_document("unused", "unused", 0.1)]
        mock_processor_class.return_value = processor_instance

        model_manager = Mock()
        model_manager.check_model_support_vision.return_value = False
        mock_model_manager_class.return_value = model_manager

        all_documents: list[Document] = []
        exceptions: list[str] = []

        RetrievalService.embedding_search(
            flask_app=internal_flask_app,
            dataset_id=internal_dataset.id,
            query="file-id",
            top_k=4,
            score_threshold=0.5,
            reranking_model={
                "reranking_provider_name": "provider",
                "reranking_model_name": "model",
            },
            all_documents=all_documents,
            retrieval_method=RetrievalMethod.SEMANTIC_SEARCH,
            exceptions=exceptions,
            query_type=QueryType.IMAGE_QUERY,
        )

        assert all_documents == original_docs
        assert exceptions == []
        mock_model_manager_class.assert_called_once_with(tenant_id=internal_dataset.tenant_id)
        processor_instance.invoke.assert_not_called()

    @patch("core.rag.datasource.retrieval_service.DataPostProcessor")
    @patch("core.rag.datasource.retrieval_service.Vector")
    @patch("core.rag.datasource.retrieval_service.RetrievalService._get_dataset")
    def test_embedding_search_text_with_reranking_non_multimodal(
        self, mock_get_dataset, mock_vector_class, mock_processor_class, internal_dataset, internal_flask_app
    ):
        internal_dataset.is_multimodal = False
        mock_get_dataset.return_value = internal_dataset
        original_docs = [create_mock_document("vector-content", "vec-doc", 0.62)]
        reranked_docs = [create_mock_document("vector-content-reranked", "vec-doc", 0.89)]

        vector_instance = Mock()
        vector_instance.search_by_vector.return_value = original_docs
        mock_vector_class.return_value = vector_instance

        processor_instance = Mock()
        processor_instance.invoke.return_value = reranked_docs
        mock_processor_class.return_value = processor_instance

        all_documents: list[Document] = []
        exceptions: list[str] = []

        RetrievalService.embedding_search(
            flask_app=internal_flask_app,
            dataset_id=internal_dataset.id,
            query="query",
            top_k=4,
            score_threshold=0.5,
            reranking_model={
                "reranking_provider_name": "provider",
                "reranking_model_name": "model",
            },
            all_documents=all_documents,
            retrieval_method=RetrievalMethod.SEMANTIC_SEARCH,
            exceptions=exceptions,
            query_type=QueryType.TEXT_QUERY,
        )

        assert all_documents == reranked_docs
        assert exceptions == []
        processor_instance.invoke.assert_called_once()

    @patch("core.rag.datasource.retrieval_service.Vector")
    @patch("core.rag.datasource.retrieval_service.RetrievalService._get_dataset")
    def test_embedding_search_appends_exception_when_vector_fails(
        self, mock_get_dataset, mock_vector_class, internal_dataset, internal_flask_app
    ):
        mock_get_dataset.return_value = internal_dataset
        vector_instance = Mock()
        vector_instance.search_by_vector.side_effect = RuntimeError("vector failed")
        mock_vector_class.return_value = vector_instance
        all_documents: list[Document] = []
        exceptions: list[str] = []

        RetrievalService.embedding_search(
            flask_app=internal_flask_app,
            dataset_id=internal_dataset.id,
            query="query",
            top_k=4,
            score_threshold=0.5,
            reranking_model=None,
            all_documents=all_documents,
            retrieval_method=RetrievalMethod.SEMANTIC_SEARCH,
            exceptions=exceptions,
            query_type=QueryType.TEXT_QUERY,
        )

        assert all_documents == []
        assert exceptions == ["vector failed"]

    @patch("core.rag.datasource.retrieval_service.Vector")
    @patch("core.rag.datasource.retrieval_service.RetrievalService._get_dataset")
    def test_full_text_index_search_without_reranking(
        self, mock_get_dataset, mock_vector_class, internal_dataset, internal_flask_app
    ):
        mock_get_dataset.return_value = internal_dataset
        vector_instance = Mock()
        vector_instance.search_by_full_text.return_value = [create_mock_document("fulltext", "ft-1", 0.68)]
        mock_vector_class.return_value = vector_instance
        all_documents: list[Document] = []
        exceptions: list[str] = []

        RetrievalService.full_text_index_search(
            flask_app=internal_flask_app,
            dataset_id=internal_dataset.id,
            query='query "x"',
            top_k=4,
            score_threshold=0.4,
            reranking_model=None,
            all_documents=all_documents,
            retrieval_method=RetrievalMethod.FULL_TEXT_SEARCH,
            exceptions=exceptions,
        )

        assert len(all_documents) == 1
        assert exceptions == []
        vector_instance.search_by_full_text.assert_called_once()

    @patch("core.rag.datasource.retrieval_service.DataPostProcessor")
    @patch("core.rag.datasource.retrieval_service.Vector")
    @patch("core.rag.datasource.retrieval_service.RetrievalService._get_dataset")
    def test_full_text_index_search_with_reranking(
        self, mock_get_dataset, mock_vector_class, mock_processor_class, internal_dataset, internal_flask_app
    ):
        mock_get_dataset.return_value = internal_dataset
        original_docs = [create_mock_document("fulltext", "ft-1", 0.68)]
        reranked_docs = [create_mock_document("fulltext-reranked", "ft-1", 0.9)]

        vector_instance = Mock()
        vector_instance.search_by_full_text.return_value = original_docs
        mock_vector_class.return_value = vector_instance

        processor_instance = Mock()
        processor_instance.invoke.return_value = reranked_docs
        mock_processor_class.return_value = processor_instance

        all_documents: list[Document] = []
        exceptions: list[str] = []

        RetrievalService.full_text_index_search(
            flask_app=internal_flask_app,
            dataset_id=internal_dataset.id,
            query="query",
            top_k=4,
            score_threshold=0.4,
            reranking_model={
                "reranking_provider_name": "provider",
                "reranking_model_name": "model",
            },
            all_documents=all_documents,
            retrieval_method=RetrievalMethod.FULL_TEXT_SEARCH,
            exceptions=exceptions,
        )

        assert all_documents == reranked_docs
        assert exceptions == []
        processor_instance.invoke.assert_called_once()

    @patch("core.rag.datasource.retrieval_service.RetrievalService._get_dataset")
    def test_full_text_index_search_dataset_not_found(self, mock_get_dataset, internal_flask_app):
        mock_get_dataset.return_value = None
        all_documents: list[Document] = []
        exceptions: list[str] = []

        RetrievalService.full_text_index_search(
            flask_app=internal_flask_app,
            dataset_id="dataset-id",
            query="query",
            top_k=4,
            score_threshold=0.4,
            reranking_model=None,
            all_documents=all_documents,
            retrieval_method=RetrievalMethod.FULL_TEXT_SEARCH,
            exceptions=exceptions,
        )

        assert all_documents == []
        assert exceptions == ["dataset not found"]

    @patch("core.rag.datasource.retrieval_service.Vector")
    @patch("core.rag.datasource.retrieval_service.RetrievalService._get_dataset")
    def test_full_text_index_search_appends_exception_when_search_fails(
        self, mock_get_dataset, mock_vector_class, internal_dataset, internal_flask_app
    ):
        mock_get_dataset.return_value = internal_dataset
        vector_instance = Mock()
        vector_instance.search_by_full_text.side_effect = RuntimeError("fulltext failed")
        mock_vector_class.return_value = vector_instance
        all_documents: list[Document] = []
        exceptions: list[str] = []

        RetrievalService.full_text_index_search(
            flask_app=internal_flask_app,
            dataset_id=internal_dataset.id,
            query="query",
            top_k=4,
            score_threshold=0.4,
            reranking_model=None,
            all_documents=all_documents,
            retrieval_method=RetrievalMethod.FULL_TEXT_SEARCH,
            exceptions=exceptions,
        )

        assert all_documents == []
        assert exceptions == ["fulltext failed"]

    def test_format_retrieval_documents_with_empty_input_returns_empty_list(self):
        assert RetrievalService.format_retrieval_documents([]) == []

    def test_format_retrieval_documents_without_document_id_returns_empty_list(self):
        documents = [Document(page_content="content", metadata={"doc_id": "doc-1", "score": 0.4}, provider="dify")]

        assert RetrievalService.format_retrieval_documents(documents) == []

    def test_format_retrieval_documents_with_parent_child_summary_and_attachments(
        self, monkeypatch: pytest.MonkeyPatch, database: Database
    ):
        session = database.session
        _persist_dataset(session)
        _persist_dataset_document(session, document_id="doc-parent", doc_form=IndexStructureType.PARENT_CHILD_INDEX)
        _persist_dataset_document(session, document_id="doc-text", doc_form=IndexStructureType.PARAGRAPH_INDEX)
        _persist_dataset_document(
            session, document_id="doc-parent-summary", doc_form=IndexStructureType.PARENT_CHILD_INDEX
        )

        segments = [
            DocumentSegment(
                tenant_id="tenant-id",
                dataset_id="dataset-id",
                document_id=document_id,
                position=position,
                content=f"Content for {segment_id}",
                word_count=4,
                tokens=4,
                created_by="user-id",
                index_node_id=index_node_id,
                status=SegmentStatus.COMPLETED,
            )
            for position, (segment_id, document_id, index_node_id) in enumerate(
                [
                    ("segment-parent", "doc-parent", "parent-node"),
                    ("segment-text", "doc-text", "index-node-1"),
                    ("segment-summary", "doc-text", "summary-node"),
                    ("segment-parent-summary", "doc-parent-summary", "summary-parent-node"),
                ],
                start=1,
            )
        ]
        for segment, segment_id in zip(
            segments,
            ["segment-parent", "segment-text", "segment-summary", "segment-parent-summary"],
            strict=True,
        ):
            segment.id = segment_id
        child_chunk = ChildChunk(
            tenant_id="tenant-id",
            dataset_id="dataset-id",
            document_id="doc-parent",
            segment_id="segment-parent",
            position=2,
            content="child details",
            word_count=2,
            created_by="user-id",
            index_node_id="child-node-1",
        )
        child_chunk.id = "child-chunk-1"
        summaries = [
            DocumentSegmentSummary(
                dataset_id="dataset-id",
                document_id=document_id,
                chunk_id=chunk_id,
                summary_content=summary,
                status=SummaryStatus.COMPLETED,
                enabled=True,
            )
            for document_id, chunk_id, summary in [
                ("doc-text", "segment-summary", "summary for text"),
                ("doc-parent-summary", "segment-parent-summary", "summary for parent"),
            ]
        ]
        session.add_all([*segments, child_chunk, *summaries])
        session.commit()

        monkeypatch.setattr(retrieval_service_module, "RetrievalChildChunk", _SimpleRetrievalChildChunk)
        monkeypatch.setattr(retrieval_service_module, "RetrievalSegments", _SimpleRetrievalSegment)

        input_documents = [
            Document(
                page_content="child node content",
                metadata={"document_id": "doc-parent", "doc_id": "child-node-1", "score": 0.7},
                provider="dify",
            ),
            Document(
                page_content="parent image",
                metadata={
                    "document_id": "doc-parent",
                    "doc_id": "attach-node-1",
                    "doc_type": DocType.IMAGE,
                    "score": 0.8,
                },
                provider="dify",
            ),
            Document(
                page_content="text index node",
                metadata={"document_id": "doc-text", "doc_id": "index-node-1", "score": 0.6},
                provider="dify",
            ),
            Document(
                page_content="text image node",
                metadata={
                    "document_id": "doc-text",
                    "doc_id": "attach-text-1",
                    "doc_type": DocType.IMAGE,
                    "score": 0.65,
                },
                provider="dify",
            ),
            Document(
                page_content="summary candidate 1",
                metadata={
                    "document_id": "doc-text",
                    "doc_id": "summary-node-1",
                    "is_summary": True,
                    "original_chunk_id": "segment-summary",
                    "score": "0.9",
                },
                provider="dify",
            ),
            Document(
                page_content="summary candidate 2",
                metadata={
                    "document_id": "doc-text",
                    "doc_id": "summary-node-2",
                    "is_summary": True,
                    "original_chunk_id": "segment-summary",
                    "score": "0.95",
                },
                provider="dify",
            ),
            Document(
                page_content="invalid score summary",
                metadata={
                    "document_id": "doc-parent-summary",
                    "doc_id": "summary-parent-invalid",
                    "is_summary": True,
                    "original_chunk_id": "segment-parent-summary",
                    "score": "invalid",
                },
                provider="dify",
            ),
            Document(
                page_content="valid parent summary",
                metadata={
                    "document_id": "doc-parent-summary",
                    "doc_id": "summary-parent-valid",
                    "is_summary": True,
                    "original_chunk_id": "segment-parent-summary",
                    "score": "0.4",
                },
                provider="dify",
            ),
        ]

        monkeypatch.setattr(
            RetrievalService,
            "get_segment_attachment_infos",
            lambda attachment_ids, session: [
                {
                    "attachment_id": "attach-node-1",
                    "attachment_info": {
                        "id": "attach-node-1",
                        "name": "img-parent",
                        "extension": ".png",
                        "mime_type": "image/png",
                        "source_url": "signed://parent",
                        "size": 11,
                    },
                    "segment_id": "segment-parent",
                },
                {
                    "attachment_id": "attach-text-1",
                    "attachment_info": {
                        "id": "attach-text-1",
                        "name": "img-text",
                        "extension": ".png",
                        "mime_type": "image/png",
                        "source_url": "signed://text",
                        "size": 22,
                    },
                    "segment_id": "segment-text",
                },
            ],
        )

        result = RetrievalService.format_retrieval_documents(input_documents)

        assert len(result) == 4
        result_by_segment_id = {item.segment.id: item for item in result}
        assert result_by_segment_id["segment-summary"].score == pytest.approx(0.95)
        assert result_by_segment_id["segment-summary"].summary == "summary for text"
        assert result_by_segment_id["segment-parent"].score == pytest.approx(0.8)
        assert result_by_segment_id["segment-parent"].files is not None
        assert len(result_by_segment_id["segment-parent"].child_chunks or []) == 1
        assert result_by_segment_id["segment-text"].score == pytest.approx(0.65)
        assert result_by_segment_id["segment-parent-summary"].score == pytest.approx(0.4)
        assert result_by_segment_id["segment-parent-summary"].summary == "summary for parent"
        assert result_by_segment_id["segment-parent-summary"].child_chunks == []

    def test_format_retrieval_documents_rolls_back_and_raises_when_db_fails(self, database: Database):
        session = database.session

        def fail_dataset_document_query(orm_execute_state) -> None:
            if orm_execute_state.is_select:
                raise RuntimeError("db error")

        event.listen(session, "do_orm_execute", fail_dataset_document_query)

        documents = [Document(page_content="content", metadata={"document_id": "doc-1"}, provider="dify")]

        try:
            with pytest.raises(RuntimeError, match="db error"):
                RetrievalService.format_retrieval_documents(documents)
        finally:
            event.remove(session, "do_orm_execute", fail_dataset_document_query)

        assert not session.in_transaction()
        assert session.scalar(select(Dataset).where(Dataset.id == "missing")) is None

    def test_retrieve_internal_returns_early_without_query_or_attachment(self, internal_dataset, internal_flask_app):
        all_documents: list[Document] = []
        exceptions: list[str] = []

        RetrievalService()._retrieve(
            flask_app=internal_flask_app,
            retrieval_method=RetrievalMethod.SEMANTIC_SEARCH,
            dataset=internal_dataset,
            all_documents=all_documents,
            exceptions=exceptions,
            query=None,
            attachment_id=None,
        )

        assert all_documents == []
        assert exceptions == []

    def test_retrieve_internal_cancels_futures_when_future_has_exception(self, internal_dataset, internal_flask_app):
        future_error = Mock()
        future_error.exception.return_value = RuntimeError("future failed")
        future_ok = Mock()
        future_ok.exception.return_value = None

        with (
            patch("core.rag.datasource.retrieval_service.ThreadPoolExecutor") as mock_executor,
            patch(
                "core.rag.datasource.retrieval_service.concurrent.futures.as_completed",
                return_value=[future_error, future_ok],
            ),
        ):
            mock_executor_instance = Mock()
            mock_executor_instance.submit.side_effect = [future_error, future_ok]
            mock_executor.return_value.__enter__.return_value = mock_executor_instance
            RetrievalService()._retrieve(
                flask_app=internal_flask_app,
                retrieval_method=RetrievalMethod.SEMANTIC_SEARCH,
                dataset=internal_dataset,
                all_documents=[],
                exceptions=[],
                query="query",
                attachment_id="file-1",
            )

        future_error.cancel.assert_called()
        future_ok.cancel.assert_called()

    def test_retrieve_internal_raises_value_error_when_exceptions_exist(
        self, monkeypatch: pytest.MonkeyPatch, internal_dataset, internal_flask_app
    ):
        executor = _ImmediateExecutor()
        monkeypatch.setattr(retrieval_service_module, "ThreadPoolExecutor", lambda *args, **kwargs: executor)
        monkeypatch.setattr(
            retrieval_service_module.concurrent.futures,
            "as_completed",
            lambda futures, timeout=None: iter(futures),
        )

        with patch("core.rag.datasource.retrieval_service.RetrievalService.keyword_search") as mock_keyword_search:
            mock_keyword_search.side_effect = lambda *args, **kwargs: None
            with pytest.raises(ValueError, match="keyword error"):
                RetrievalService()._retrieve(
                    flask_app=internal_flask_app,
                    retrieval_method=RetrievalMethod.KEYWORD_SEARCH,
                    dataset=internal_dataset,
                    all_documents=[],
                    exceptions=["keyword error"],
                    query="query",
                )

    def test_retrieve_internal_hybrid_weighted_attachment_flow(
        self, monkeypatch: pytest.MonkeyPatch, internal_dataset, internal_flask_app
    ):
        executor = _ImmediateExecutor()
        monkeypatch.setattr(retrieval_service_module, "ThreadPoolExecutor", lambda *args, **kwargs: executor)
        monkeypatch.setattr(
            retrieval_service_module.concurrent.futures,
            "as_completed",
            lambda futures, timeout=None: iter(futures),
        )

        text_doc = create_mock_document("text", "text-doc", 0.81)
        image_doc = create_mock_document("image", "image-doc", 0.72)
        fulltext_doc = create_mock_document("full", "full-doc", 0.65)
        processed_doc = create_mock_document("processed", "processed-doc", 0.99)

        with (
            patch("core.rag.datasource.retrieval_service.RetrievalService.embedding_search") as mock_embedding_search,
            patch("core.rag.datasource.retrieval_service.RetrievalService.full_text_index_search") as mock_fulltext,
            patch("core.rag.datasource.retrieval_service.DataPostProcessor") as mock_processor_class,
        ):

            def embedding_side_effect(
                flask_app,
                dataset_id,
                query,
                top_k,
                score_threshold,
                reranking_model,
                all_documents,
                retrieval_method,
                exceptions,
                document_ids_filter=None,
                query_type=QueryType.TEXT_QUERY,
            ):
                if query_type == QueryType.IMAGE_QUERY:
                    all_documents.append(image_doc)
                else:
                    all_documents.append(text_doc)

            mock_embedding_search.side_effect = embedding_side_effect

            def fulltext_side_effect(
                flask_app,
                dataset_id,
                query,
                top_k,
                score_threshold,
                reranking_model,
                all_documents,
                retrieval_method,
                exceptions,
                document_ids_filter=None,
            ):
                all_documents.append(fulltext_doc)

            mock_fulltext.side_effect = fulltext_side_effect
            processor_instance = Mock()
            processor_instance.invoke.return_value = [processed_doc]
            mock_processor_class.return_value = processor_instance

            all_documents: list[Document] = []
            RetrievalService()._retrieve(
                flask_app=internal_flask_app,
                retrieval_method=RetrievalMethod.HYBRID_SEARCH,
                dataset=internal_dataset,
                all_documents=all_documents,
                exceptions=[],
                query="query",
                attachment_id="file-1",
                reranking_mode=RerankMode.WEIGHTED_SCORE,
                top_k=3,
            )

        assert len(all_documents) == 4
        assert any(doc.metadata["doc_id"] == "processed-doc" for doc in all_documents)
        processor_instance.invoke.assert_called_once()

    @patch("core.rag.datasource.retrieval_service.sign_upload_file_preview_url", return_value="signed://file")
    @patch("core.rag.datasource.retrieval_service.grant_upload_file_access")
    def test_get_segment_attachment_info_success(self, grant_access, mock_sign, database: Database):
        upload_file = _persist_upload_file(database.session, upload_id="upload-1")
        binding = SegmentAttachmentBinding(
            tenant_id="tenant-id",
            dataset_id="dataset-id",
            document_id="document-id",
            segment_id="segment-1",
            attachment_id=upload_file.id,
        )
        database.session.add(binding)
        database.session.commit()

        result = RetrievalService.get_segment_attachment_info("dataset-id", "tenant-id", "upload-1", database.session)

        assert result == {
            "attachment_info": {
                "id": "upload-1",
                "name": "file-upload-1",
                "extension": ".png",
                "mime_type": "image/png",
                "source_url": "signed://file",
                "size": 42,
            },
            "segment_id": "segment-1",
        }
        mock_sign.assert_called_once_with("upload-1", "png")
        grant_access.assert_called_once_with(["upload-1"])

    def test_get_segment_attachment_info_returns_none_when_binding_missing(self, database: Database):
        _persist_upload_file(database.session, upload_id="upload-1")

        result = RetrievalService.get_segment_attachment_info("dataset-id", "tenant-id", "upload-1", database.session)

        assert result is None

    def test_get_segment_attachment_info_returns_none_when_upload_file_missing(self, database: Database):
        result = RetrievalService.get_segment_attachment_info("dataset-id", "tenant-id", "upload-1", database.session)

        assert result is None

    @patch("core.rag.datasource.retrieval_service.grant_upload_file_access")
    def test_get_segment_attachment_infos_returns_empty_when_upload_files_missing(
        self, grant_access, database: Database
    ):
        result = RetrievalService.get_segment_attachment_infos(["upload-1"], database.session)

        assert result == []
        grant_access.assert_called_once_with([])

    @patch("core.rag.datasource.retrieval_service.grant_upload_file_access")
    def test_get_segment_attachment_infos_returns_empty_when_bindings_missing(self, grant_access, database: Database):
        _persist_upload_file(database.session, upload_id="upload-1")

        result = RetrievalService.get_segment_attachment_infos(["upload-1"], database.session)

        assert result == []
        grant_access.assert_called_once_with([])

    @patch("core.rag.datasource.retrieval_service.sign_upload_file_preview_url", return_value="signed://file")
    @patch("core.rag.datasource.retrieval_service.grant_upload_file_access")
    def test_get_segment_attachment_infos_success(self, grant_access, mock_sign, database: Database):
        upload_file_1 = _persist_upload_file(database.session, upload_id="upload-1")
        _persist_upload_file(database.session, upload_id="upload-2", extension="jpg", size=99)
        _persist_upload_file(database.session, upload_id="upload-other", tenant_id="other-tenant")
        binding = SegmentAttachmentBinding(
            tenant_id="tenant-id",
            dataset_id="dataset-id",
            document_id="document-id",
            segment_id="segment-1",
            attachment_id=upload_file_1.id,
        )
        database.session.add(binding)
        database.session.commit()

        result = RetrievalService.get_segment_attachment_infos(["upload-1", "upload-2"], database.session)

        assert result == [
            {
                "attachment_id": "upload-1",
                "attachment_info": {
                    "id": "upload-1",
                    "name": "file-upload-1",
                    "extension": ".png",
                    "mime_type": "image/png",
                    "source_url": "signed://file",
                    "size": 42,
                },
                "segment_id": "segment-1",
            }
        ]
        mock_sign.assert_has_calls(
            [
                call("upload-1", "png"),
                call("upload-2", "jpg"),
            ]
        )
        assert mock_sign.call_count == 2
        grant_access.assert_called_once_with(["upload-1"])

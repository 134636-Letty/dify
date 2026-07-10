"""SQLite-backed coverage for dataset retriever utilities and independent helpers."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import nullcontext
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from yaml import YAMLError

from core.app.app_config.entities import DatasetRetrieveConfigEntity
from core.callback_handler.index_tool_callback_handler import DatasetIndexToolCallbackHandler
from core.rag.index_processor.constant.index_type import IndexTechniqueType
from core.rag.models.document import Document as RagDocument
from core.tools.utils.dataset_retriever import dataset_multi_retriever_tool as multi_retriever_module
from core.tools.utils.dataset_retriever import dataset_retriever_tool as single_retriever_module
from core.tools.utils.dataset_retriever.dataset_multi_retriever_tool import DatasetMultiRetrieverTool
from core.tools.utils.dataset_retriever.dataset_retriever_tool import DatasetRetrieverTool as SingleDatasetRetrieverTool
from core.tools.utils.text_processing_utils import remove_leading_symbols
from core.tools.utils.uuid_utils import is_valid_uuid
from core.tools.utils.yaml_utils import _load_yaml_file, load_yaml_file_cached
from models.base import TypeBase
from models.dataset import Dataset, Document, DocumentSegment
from models.enums import DataSourceType, DocumentCreatedFrom, SegmentStatus


def _retrieve_config() -> DatasetRetrieveConfigEntity:
    return DatasetRetrieveConfigEntity(retrieve_strategy=DatasetRetrieveConfigEntity.RetrieveStrategy.SINGLE)


class _FakeFlaskApp:
    def app_context(self):
        return nullcontext()


class _ImmediateThread:
    def __init__(self, target=None, kwargs=None, **_kwargs):
        self._target = target
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(**self._kwargs)

    def join(self):
        return None


class _TestHitCallback(DatasetIndexToolCallbackHandler):
    def __init__(self):
        self.queries: list[tuple[str, str]] = []
        self.documents: list[RagDocument] | None = None
        self.resources = None

    def on_query(self, query: str, dataset_id: str, session=None):
        self.queries.append((query, dataset_id))

    def on_tool_end(self, documents: list[RagDocument], session=None):
        self.documents = documents

    def return_retriever_resource_info(self, resource):
        self.resources = list(resource)


@pytest.fixture
def orm_session(sqlite_engine: Engine) -> Iterator[Session]:
    """Provide persisted dataset state through a caller-owned SQLAlchemy session."""
    TypeBase.metadata.create_all(
        sqlite_engine,
        tables=[Dataset.__table__, Document.__table__, DocumentSegment.__table__],
    )
    with Session(sqlite_engine, expire_on_commit=False) as session:
        yield session


@pytest.fixture
def bind_retriever_db_sessions(
    monkeypatch: pytest.MonkeyPatch, sqlite_engine: Engine
) -> Iterator[scoped_session[Session]]:
    """Bind both retriever modules' Flask-style session proxy to SQLite."""
    session_proxy = scoped_session(sessionmaker(bind=sqlite_engine, expire_on_commit=False))
    monkeypatch.setattr(single_retriever_module, "db", SimpleNamespace(session=session_proxy))
    monkeypatch.setattr(multi_retriever_module, "db", SimpleNamespace(session=session_proxy))
    try:
        yield session_proxy
    finally:
        session_proxy.remove()


def _dataset(
    *,
    dataset_id: str = "dataset-1",
    tenant_id: str = "tenant-1",
    name: str = "Knowledge Base",
    provider: str = "vendor",
    retrieval_model: dict | None = None,
) -> Dataset:
    return Dataset(
        id=dataset_id,
        tenant_id=tenant_id,
        name=name,
        provider=provider,
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique=IndexTechniqueType.HIGH_QUALITY,
        retrieval_model=retrieval_model,
        created_by="creator",
    )


def _document(
    *,
    document_id: str,
    dataset_id: str,
    tenant_id: str = "tenant-1",
    name: str,
    data_source_type: DataSourceType,
    metadata: dict,
) -> Document:
    return Document(
        id=document_id,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        position=1,
        data_source_type=data_source_type,
        batch="batch-1",
        name=name,
        created_from=DocumentCreatedFrom.WEB,
        created_by="creator",
        doc_metadata=metadata,
    )


def _segment(
    *,
    segment_id: str,
    dataset_id: str,
    document_id: str,
    index_node_id: str,
    content: str,
    answer: str | None,
    hit_count: int,
    word_count: int,
    position: int,
    index_node_hash: str,
) -> DocumentSegment:
    segment = DocumentSegment(
        tenant_id="tenant-1",
        dataset_id=dataset_id,
        document_id=document_id,
        position=position,
        content=content,
        word_count=word_count,
        tokens=word_count,
        created_by="creator",
        index_node_id=index_node_id,
        index_node_hash=index_node_hash,
        answer=answer,
        hit_count=hit_count,
        status=SegmentStatus.COMPLETED,
        completed_at=datetime.now(),
    )
    segment.id = segment_id
    return segment


def test_remove_leading_symbols_preserves_markdown_link_and_strips_punctuation():
    markdown = "[Example](https://example.com) content"
    assert remove_leading_symbols(markdown) == markdown

    assert remove_leading_symbols("...Hello world") == "Hello world"


def test_is_valid_uuid_handles_valid_invalid_and_empty_values():
    assert is_valid_uuid(str(uuid.uuid4())) is True
    assert is_valid_uuid("not-a-uuid") is False
    assert is_valid_uuid("") is False
    assert is_valid_uuid(None) is False


def test_load_yaml_file_valid(tmp_path):
    valid_file = tmp_path / "valid.yaml"
    valid_file.write_text("a: 1\nb: two\n", encoding="utf-8")

    loaded = _load_yaml_file(file_path=str(valid_file))

    assert loaded == {"a": 1, "b": "two"}


def test_load_yaml_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_yaml_file(file_path=str(tmp_path / "missing.yaml"))


def test_load_yaml_file_invalid(tmp_path):
    invalid_file = tmp_path / "invalid.yaml"
    invalid_file.write_text("a: [1, 2\n", encoding="utf-8")

    with pytest.raises(YAMLError):
        _load_yaml_file(file_path=str(invalid_file))


def test_load_yaml_file_cached_hits(tmp_path):
    valid_file = tmp_path / "valid.yaml"
    valid_file.write_text("a: 1\nb: two\n", encoding="utf-8")

    load_yaml_file_cached.cache_clear()
    assert load_yaml_file_cached(str(valid_file)) == {"a": 1, "b": "two"}

    assert load_yaml_file_cached(str(valid_file)) == {"a": 1, "b": "two"}
    assert load_yaml_file_cached.cache_info().hits == 1


def test_single_dataset_retriever_from_dataset_builds_name_and_description(orm_session: Session) -> None:
    dataset = _dataset(name="Knowledge")
    orm_session.add(dataset)
    orm_session.commit()

    tool = SingleDatasetRetrieverTool.from_dataset(
        dataset=dataset,
        retrieve_config=_retrieve_config(),
        return_resource=False,
        retriever_from="prod",
        inputs={},
    )

    assert tool.name == "dataset_dataset_1"
    assert tool.description == "useful for when you want to answer queries about the Knowledge"


def test_single_dataset_retriever_external_run_returns_content_and_resources(
    orm_session: Session, bind_retriever_db_sessions: scoped_session[Session]
) -> None:
    dataset = _dataset(provider="external", retrieval_model={})
    orm_session.add(dataset)
    orm_session.commit()
    callback = _TestHitCallback()
    dataset_retrieval = Mock()
    dataset_retrieval.get_metadata_filter_condition.return_value = (
        {"dataset-1": ["doc-a"]},
        {"logical_operator": "and"},
    )
    external_documents = [
        {"content": "first", "metadata": {"document_id": "doc-a"}, "score": 0.9, "title": "Doc A"},
        {"content": "second", "metadata": {"document_id": "doc-b"}, "score": 0.8, "title": "Doc B"},
    ]

    tool = SingleDatasetRetrieverTool(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        retrieve_config=_retrieve_config(),
        return_resource=True,
        retriever_from="dev",
        hit_callbacks=[callback],
        inputs={"x": 1},
    )

    with patch.object(single_retriever_module, "DatasetRetrieval", return_value=dataset_retrieval):
        with patch.object(
            single_retriever_module.ExternalDatasetService,
            "fetch_external_knowledge_retrieval",
            return_value=external_documents,
        ) as fetch_mock:
            result = tool.run(session=orm_session, query="hello")

    assert result == "first\nsecond"
    assert callback.queries == [("hello", "dataset-1")]
    assert callback.resources is not None
    resource_info = callback.resources
    assert [item.position for item in resource_info] == [1, 2]
    assert resource_info[0].dataset_id == "dataset-1"
    fetch_mock.assert_called_once()
    assert fetch_mock.call_args.kwargs["session"] is orm_session


def test_single_dataset_retriever_returns_empty_when_metadata_filter_finds_no_documents(
    orm_session: Session, bind_retriever_db_sessions: scoped_session[Session]
) -> None:
    dataset = _dataset()
    orm_session.add(dataset)
    orm_session.commit()
    dataset_retrieval = Mock()
    dataset_retrieval.get_metadata_filter_condition.return_value = ({"dataset-1": []}, {"logical_operator": "and"})
    tool = SingleDatasetRetrieverTool(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        retrieve_config=_retrieve_config(),
        return_resource=False,
        retriever_from="prod",
        hit_callbacks=[_TestHitCallback()],
        inputs={},
    )

    with patch.object(single_retriever_module, "DatasetRetrieval", return_value=dataset_retrieval):
        with patch.object(single_retriever_module.RetrievalService, "retrieve") as retrieve_mock:
            result = tool.run(session=orm_session, query="hello")

    assert result == ""
    retrieve_mock.assert_not_called()


def test_single_dataset_retriever_non_economy_run_sorts_context_and_resources(
    orm_session: Session, bind_retriever_db_sessions: scoped_session[Session]
) -> None:
    dataset = _dataset(
        retrieval_model={
            "search_method": "semantic_search",
            "score_threshold_enabled": True,
            "score_threshold": 0.2,
            "reranking_enable": True,
            "reranking_model": {"reranking_provider_name": "provider", "reranking_model_name": "model"},
            "reranking_mode": "reranking_model",
            "weights": {"vector_setting": {"vector_weight": 0.6}},
        },
    )
    callback = _TestHitCallback()
    dataset_retrieval = Mock()
    dataset_retrieval.get_metadata_filter_condition.return_value = (None, None)
    low_document = _document(
        document_id="doc-low",
        dataset_id=dataset.id,
        name="Document Low",
        data_source_type=DataSourceType.UPLOAD_FILE,
        metadata={"lang": "en"},
    )
    high_document = _document(
        document_id="doc-high",
        dataset_id=dataset.id,
        name="Document High",
        data_source_type=DataSourceType.NOTION_IMPORT,
        metadata={"lang": "fr"},
    )
    low_segment = _segment(
        segment_id="seg-low",
        dataset_id=dataset.id,
        document_id=low_document.id,
        index_node_id="node-low",
        content="raw low",
        answer="low answer",
        hit_count=1,
        word_count=10,
        position=3,
        index_node_hash="hash-low",
    )
    high_segment = _segment(
        segment_id="seg-high",
        dataset_id=dataset.id,
        document_id=high_document.id,
        index_node_id="node-high",
        content="raw high",
        answer=None,
        hit_count=9,
        word_count=25,
        position=1,
        index_node_hash="hash-high",
    )
    orm_session.add_all([dataset, low_document, high_document, low_segment, high_segment])
    orm_session.commit()
    records = [
        SimpleNamespace(segment=low_segment, score=0.2, summary="summary low"),
        SimpleNamespace(segment=high_segment, score=0.9, summary=None),
    ]
    documents = [
        RagDocument(page_content="first", metadata={"doc_id": "node-low", "score": 0.2}),
        RagDocument(page_content="second", metadata={"doc_id": "node-high", "score": 0.9}),
    ]
    tool = SingleDatasetRetrieverTool(
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        retrieve_config=_retrieve_config(),
        return_resource=True,
        retriever_from="dev",
        hit_callbacks=[callback],
        inputs={},
        top_k=2,
    )

    with patch.object(single_retriever_module, "DatasetRetrieval", return_value=dataset_retrieval):
        with patch.object(single_retriever_module.RetrievalService, "retrieve", return_value=documents):
            with patch.object(
                single_retriever_module.RetrievalService,
                "format_retrieval_documents",
                return_value=records,
            ):
                result = tool.run(session=orm_session, query="hello")

    assert result == "raw high\nsummary low\nquestion:raw low answer:low answer"
    assert callback.documents == documents
    assert callback.resources is not None
    resource_info = callback.resources
    assert [item.position for item in resource_info] == [1, 2]
    assert resource_info[0].segment_id == "seg-high"
    assert resource_info[0].hit_count == 9
    assert resource_info[1].summary == "summary low"
    assert resource_info[1].content == "question:raw low \nanswer:low answer"


def test_multi_dataset_retriever_from_dataset_sets_tool_name():
    tool = DatasetMultiRetrieverTool.from_dataset(
        dataset_ids=["dataset-1"],
        tenant_id="tenant-1",
        reranking_provider_name="provider",
        reranking_model_name="model",
        return_resource=False,
        retriever_from="prod",
    )

    assert tool.name == "dataset_tenant_1"


def test_multi_dataset_retriever_retriever_returns_early_when_dataset_is_missing(
    orm_session: Session, bind_retriever_db_sessions: scoped_session[Session]
) -> None:
    callback = _TestHitCallback()
    all_documents: list[RagDocument] = []
    orm_session.add(_dataset(tenant_id="tenant-2"))
    orm_session.commit()
    tool = DatasetMultiRetrieverTool(
        tenant_id="tenant-1",
        dataset_ids=["dataset-1"],
        reranking_provider_name="provider",
        reranking_model_name="model",
        return_resource=False,
        retriever_from="prod",
    )

    with patch.object(multi_retriever_module.RetrievalService, "retrieve") as retrieve_mock:
        result = tool._retriever(
            flask_app=_FakeFlaskApp(),
            dataset_id="dataset-1",
            query="hello",
            all_documents=all_documents,
            hit_callbacks=[callback],
        )

    assert result == []
    assert all_documents == []
    assert callback.queries == []
    retrieve_mock.assert_not_called()


def test_multi_dataset_retriever_retriever_non_economy_uses_retrieval_model(
    orm_session: Session, bind_retriever_db_sessions: scoped_session[Session]
) -> None:
    dataset = _dataset(
        retrieval_model={
            "search_method": "semantic_search",
            "top_k": 6,
            "score_threshold_enabled": True,
            "score_threshold": 0.4,
            "reranking_enable": False,
            "reranking_mode": None,
            "weights": {"balanced": True},
        },
    )
    callback = _TestHitCallback()
    documents = [RagDocument(page_content="retrieved", metadata={"doc_id": "node-1", "score": 0.4})]
    all_documents: list[RagDocument] = []
    orm_session.add(dataset)
    orm_session.commit()
    tool = DatasetMultiRetrieverTool(
        tenant_id="tenant-1",
        dataset_ids=["dataset-1"],
        reranking_provider_name="provider",
        reranking_model_name="model",
        return_resource=False,
        retriever_from="prod",
        top_k=2,
    )

    with patch.object(multi_retriever_module.RetrievalService, "retrieve", return_value=documents) as retrieve_mock:
        tool._retriever(
            flask_app=_FakeFlaskApp(),
            dataset_id="dataset-1",
            query="hello",
            all_documents=all_documents,
            hit_callbacks=[callback],
        )

    assert all_documents == documents
    assert callback.queries == [("hello", "dataset-1")]
    retrieve_mock.assert_called_once_with(
        retrieval_method="semantic_search",
        dataset_id="dataset-1",
        query="hello",
        top_k=6,
        score_threshold=0.4,
        reranking_model=None,
        reranking_mode="reranking_model",
        weights={"balanced": True},
    )


def test_multi_dataset_retriever_run_orders_segments_and_returns_resources(
    orm_session: Session, bind_retriever_db_sessions: scoped_session[Session]
) -> None:
    callback = _TestHitCallback()
    tool = DatasetMultiRetrieverTool(
        tenant_id="tenant-1",
        dataset_ids=["dataset-1", "dataset-2"],
        reranking_provider_name="provider",
        reranking_model_name="model",
        return_resource=True,
        retriever_from="dev",
        hit_callbacks=[callback],
        top_k=2,
        score_threshold=0.1,
    )
    first_doc = RagDocument(page_content="first", metadata={"doc_id": "node-2", "score": 0.4})
    second_doc = RagDocument(page_content="second", metadata={"doc_id": "node-1", "score": 0.9})

    def fake_retriever(**kwargs):
        if kwargs["dataset_id"] == "dataset-1":
            kwargs["all_documents"].append(first_doc)
        else:
            kwargs["all_documents"].append(second_doc)

    dataset_one = _dataset(dataset_id="dataset-1", name="Dataset One")
    dataset_two = _dataset(dataset_id="dataset-2", name="Dataset Two")
    document_one = _document(
        document_id="doc-1",
        dataset_id=dataset_two.id,
        name="Doc One",
        data_source_type=DataSourceType.UPLOAD_FILE,
        metadata={"p": 1},
    )
    document_two = _document(
        document_id="doc-2",
        dataset_id=dataset_one.id,
        name="Doc Two",
        data_source_type=DataSourceType.NOTION_IMPORT,
        metadata={"p": 2},
    )
    segment_for_node_2 = _segment(
        segment_id="seg-2",
        dataset_id=dataset_one.id,
        document_id=document_two.id,
        index_node_id="node-2",
        content="raw two",
        answer="answer two",
        hit_count=2,
        word_count=20,
        position=2,
        index_node_hash="hash-2",
    )
    segment_for_node_1 = _segment(
        segment_id="seg-1",
        dataset_id=dataset_two.id,
        document_id=document_one.id,
        index_node_id="node-1",
        content="raw one",
        answer=None,
        hit_count=7,
        word_count=30,
        position=1,
        index_node_hash="hash-1",
    )
    orm_session.add_all([dataset_one, dataset_two, document_one, document_two, segment_for_node_2, segment_for_node_1])
    orm_session.commit()
    model_manager = Mock()
    model_manager.get_model_instance.return_value = Mock()
    rerank_runner = Mock()
    rerank_runner.run.return_value = [second_doc, first_doc]
    fake_current_app = SimpleNamespace(_get_current_object=lambda: _FakeFlaskApp())

    with patch.object(tool, "_retriever", side_effect=fake_retriever) as retriever_mock:
        with patch.object(multi_retriever_module, "current_app", fake_current_app):
            with patch.object(multi_retriever_module.threading, "Thread", _ImmediateThread):
                with patch.object(multi_retriever_module, "ModelManager", return_value=model_manager):
                    with patch.object(multi_retriever_module, "RerankModelRunner", return_value=rerank_runner):
                        result = tool.run(session=orm_session, query="hello")

    assert result == "raw one\nquestion:raw two answer:answer two"
    assert retriever_mock.call_count == 2
    assert callback.documents == [second_doc, first_doc]
    assert callback.resources is not None
    resource_info = callback.resources
    assert [item.position for item in resource_info] == [1, 2]
    assert [item.dataset_id for item in resource_info] == ["dataset-2", "dataset-1"]
    assert [item.document_id for item in resource_info] == ["doc-1", "doc-2"]
    assert resource_info[0].score == 0.9
    assert resource_info[0].content == "raw one"
    assert resource_info[1].score == 0.4
    assert resource_info[1].content == "question:raw two \nanswer:answer two"

"""SQLite-backed tests for dataset segment console controllers."""

import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
from flask import Flask
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from werkzeug.exceptions import Forbidden, NotFound

import extensions.ext_database as ext_database_module
import models.dataset as dataset_model_module
import services
from controllers.console import console_ns
from controllers.console.datasets import datasets_segments as segments_module
from controllers.console.datasets.datasets_segments import (
    ChildChunkAddApi,
    ChildChunkUpdateApi,
    DatasetDocumentSegmentAddApi,
    DatasetDocumentSegmentBatchImportApi,
    DatasetDocumentSegmentListApi,
    DatasetDocumentSegmentUpdateApi,
)
from controllers.console.datasets.error import ChildChunkDeleteIndexError, ChildChunkIndexingError
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from extensions.storage.storage_type import StorageType
from fields.segment_fields import segment_response_with_summary
from models.base import TypeBase
from models.dataset import ChildChunk, Dataset, Document, DocumentSegment, SegmentAttachmentBinding
from models.enums import CreatorUserRole, DataSourceType, DocumentCreatedFrom, SegmentStatus, SegmentType
from models.model import UploadFile
from services.errors.chunk import ChildChunkDeleteIndexError as ChildChunkDeleteIndexServiceError
from services.errors.chunk import ChildChunkIndexingError as ChildChunkIndexingServiceError


@dataclass(frozen=True)
class _Database:
    """Expose a real callable scoped session to controller and model code."""

    session: scoped_session[Session]


@dataclass(frozen=True)
class _Records:
    dataset: Dataset
    document: Document
    segment: DocumentSegment
    child_chunk: ChildChunk


@pytest.fixture
def database(sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Database]:
    models = (Dataset, Document, DocumentSegment, ChildChunk, SegmentAttachmentBinding, UploadFile)
    tables = [TypeBase.metadata.tables[model.__tablename__] for model in models]
    TypeBase.metadata.create_all(sqlite_engine, tables=tables)
    registry = scoped_session(sessionmaker(bind=sqlite_engine, expire_on_commit=False))
    database = _Database(registry)
    monkeypatch.setattr(segments_module, "db", database)
    monkeypatch.setattr(dataset_model_module, "db", database)
    monkeypatch.setattr(ext_database_module, "db", database)
    monkeypatch.setattr(
        segments_module,
        "dify_config",
        SimpleNamespace(SQLALCHEMY_DATABASE_URI_SCHEME="sqlite"),
    )
    try:
        yield database
    finally:
        registry.remove()


@pytest.fixture
def records(database: _Database) -> _Records:
    tenant_id = str(uuid4())
    user_id = str(uuid4())
    dataset = Dataset(
        id=str(uuid4()),
        tenant_id=tenant_id,
        name="Dataset",
        description="",
        provider="vendor",
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique=IndexTechniqueType.ECONOMY,
        created_by=user_id,
    )
    document = Document(
        id=str(uuid4()),
        tenant_id=tenant_id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        data_source_info=None,
        batch="batch",
        name="Document",
        created_from=DocumentCreatedFrom.WEB,
        created_by=user_id,
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
    )
    segment = DocumentSegment(
        tenant_id=tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        position=1,
        content="segment content",
        word_count=2,
        tokens=2,
        created_by=user_id,
        answer="answer",
        keywords=["keyword"],
        status=SegmentStatus.COMPLETED,
    )
    child_chunk = ChildChunk(
        tenant_id=tenant_id,
        dataset_id=dataset.id,
        document_id=document.id,
        segment_id=segment.id,
        position=1,
        content="child content",
        word_count=2,
        created_by=user_id,
        type=SegmentType.CUSTOMIZED,
    )
    database.session.add_all([dataset, document, segment, child_chunk])
    database.session.commit()
    return _Records(dataset=dataset, document=document, segment=segment, child_chunk=child_chunk)


def _user(records: _Records, *, editor: bool = True) -> SimpleNamespace:
    return SimpleNamespace(id=records.dataset.created_by, is_dataset_editor=editor)


@contextmanager
def _patched_services(records: _Records):
    with (
        patch.object(segments_module.DatasetService, "get_dataset", return_value=records.dataset),
        patch.object(segments_module.DocumentService, "get_document", return_value=records.document),
        patch.object(segments_module.DatasetService, "check_dataset_permission", return_value=None),
        patch.object(segments_module.DatasetService, "check_dataset_model_setting", return_value=None),
    ):
        yield


def _upload(database: _Database, records: _Records, *, name: str = "segments.csv") -> UploadFile:
    upload = UploadFile(
        tenant_id=records.dataset.tenant_id,
        storage_type=StorageType.LOCAL,
        key="segments.csv",
        name=name,
        size=10,
        extension=name.rsplit(".", 1)[-1],
        mime_type="text/csv",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by=records.dataset.created_by,
        created_at=datetime(2026, 1, 1),
        used=True,
    )
    database.session.add(upload)
    database.session.commit()
    return upload


def test_segment_response_with_summary_uses_real_model_relations(records: _Records):
    result = segment_response_with_summary(records.segment, "summary")

    assert result.id == records.segment.id
    assert result.summary == "summary"
    assert result.attachments == []


class TestSegmentList:
    def test_get_reads_persisted_segments(self, app: Flask, database: _Database, records: _Records):
        api = DatasetDocumentSegmentListApi()
        method = inspect.unwrap(api.get)

        with (
            app.test_request_context("/?keyword=segment"),
            _patched_services(records),
            patch.object(segments_module.SummaryIndexService, "get_segments_summaries", return_value={}),
        ):
            response, status = method(
                api,
                records.dataset.tenant_id,
                _user(records),
                records.dataset.id,
                records.document.id,
            )

        assert status == 200
        assert response["total"] == 1
        assert response["data"][0]["id"] == records.segment.id

    def test_get_tenant_filter_excludes_other_tenant_segment(self, app: Flask, database: _Database, records: _Records):
        records.segment.tenant_id = str(uuid4())
        database.session.commit()

        with (
            app.test_request_context("/"),
            _patched_services(records),
            patch.object(segments_module.SummaryIndexService, "get_segments_summaries", return_value={}),
        ):
            response, status = inspect.unwrap(DatasetDocumentSegmentListApi().get)(
                DatasetDocumentSegmentListApi(),
                records.dataset.tenant_id,
                _user(records),
                records.dataset.id,
                records.document.id,
            )

        assert status == 200
        assert response["total"] == 0

    def test_get_missing_dataset_and_permission_denied(self, app: Flask, records: _Records):
        method = inspect.unwrap(DatasetDocumentSegmentListApi().get)
        with (
            app.test_request_context("/"),
            patch.object(segments_module.DatasetService, "get_dataset", return_value=None),
        ):
            with pytest.raises(NotFound):
                method(
                    DatasetDocumentSegmentListApi(),
                    records.dataset.tenant_id,
                    _user(records),
                    records.dataset.id,
                    records.document.id,
                )

        with (
            app.test_request_context("/"),
            patch.object(segments_module.DatasetService, "get_dataset", return_value=records.dataset),
            patch.object(
                segments_module.DatasetService,
                "check_dataset_permission",
                side_effect=services.errors.account.NoPermissionError("denied"),
            ),
        ):
            with pytest.raises(Forbidden):
                method(
                    DatasetDocumentSegmentListApi(),
                    records.dataset.tenant_id,
                    _user(records),
                    records.dataset.id,
                    records.document.id,
                )


class TestSegmentMutation:
    def test_add_returns_persisted_segment(self, app: Flask, records: _Records):
        payload = {"content": "new content"}
        with (
            app.test_request_context("/", json=payload),
            patch.object(type(console_ns), "payload", payload),
            _patched_services(records),
            patch.object(segments_module.SegmentService, "segment_create_args_validate", return_value=None),
            patch.object(segments_module.SegmentService, "create_segment", return_value=records.segment),
            patch.object(segments_module.SummaryIndexService, "get_segment_summary", return_value=None),
        ):
            response, status = inspect.unwrap(DatasetDocumentSegmentAddApi().post)(
                DatasetDocumentSegmentAddApi(),
                records.dataset.tenant_id,
                _user(records),
                records.dataset.id,
                records.document.id,
            )

        assert status == 200
        assert response["data"]["id"] == records.segment.id

    def test_add_forbidden_for_non_editor(self, app: Flask, records: _Records):
        with (
            app.test_request_context("/", json={"content": "x"}),
            _patched_services(records),
        ):
            with pytest.raises(Forbidden):
                inspect.unwrap(DatasetDocumentSegmentAddApi().post)(
                    DatasetDocumentSegmentAddApi(),
                    records.dataset.tenant_id,
                    _user(records, editor=False),
                    records.dataset.id,
                    records.document.id,
                )

    def test_update_uses_real_segment(self, app: Flask, records: _Records):
        payload = {"content": "updated"}
        records.segment.content = "updated"
        with (
            app.test_request_context("/", json=payload),
            patch.object(type(console_ns), "payload", payload),
            _patched_services(records),
            patch.object(segments_module.SegmentService, "get_segment_by_ref", return_value=records.segment),
            patch.object(segments_module.SegmentService, "segment_create_args_validate", return_value=None),
            patch.object(segments_module.SegmentService, "update_segment", return_value=records.segment),
            patch.object(segments_module.SummaryIndexService, "get_segment_summary", return_value=None),
        ):
            response, status = inspect.unwrap(DatasetDocumentSegmentUpdateApi().patch)(
                DatasetDocumentSegmentUpdateApi(),
                records.dataset.tenant_id,
                _user(records),
                records.dataset.id,
                records.document.id,
                records.segment.id,
            )

        assert status == 200
        assert response["data"]["content"] == "updated"

    def test_update_missing_segment(self, app: Flask, records: _Records):
        with (
            app.test_request_context("/", json={"content": "x"}),
            patch.object(type(console_ns), "payload", {"content": "x"}),
            _patched_services(records),
            patch.object(segments_module.SegmentService, "get_segment_by_ref", return_value=None),
        ):
            with pytest.raises(NotFound):
                inspect.unwrap(DatasetDocumentSegmentUpdateApi().patch)(
                    DatasetDocumentSegmentUpdateApi(),
                    records.dataset.tenant_id,
                    _user(records),
                    records.dataset.id,
                    records.document.id,
                    str(uuid4()),
                )


class TestChildChunks:
    def test_add_returns_real_child_chunk(self, app: Flask, records: _Records):
        payload = {"content": "child"}
        with (
            app.test_request_context("/", json=payload),
            patch.object(type(console_ns), "payload", payload),
            _patched_services(records),
            patch.object(segments_module.SegmentService, "get_segment_by_ref", return_value=records.segment),
            patch.object(segments_module.SegmentService, "create_child_chunk", return_value=records.child_chunk),
        ):
            response, status = inspect.unwrap(ChildChunkAddApi().post)(
                ChildChunkAddApi(),
                records.dataset.tenant_id,
                _user(records),
                records.dataset.id,
                records.document.id,
                records.segment.id,
            )

        assert status == 200
        assert response["data"]["id"] == records.child_chunk.id

    def test_add_translates_indexing_error(self, app: Flask, records: _Records):
        with (
            app.test_request_context("/", json={"content": "child"}),
            patch.object(type(console_ns), "payload", {"content": "child"}),
            _patched_services(records),
            patch.object(segments_module.SegmentService, "get_segment_by_ref", return_value=records.segment),
            patch.object(
                segments_module.SegmentService,
                "create_child_chunk",
                side_effect=ChildChunkIndexingServiceError("index failed"),
            ),
        ):
            with pytest.raises(ChildChunkIndexingError):
                inspect.unwrap(ChildChunkAddApi().post)(
                    ChildChunkAddApi(),
                    records.dataset.tenant_id,
                    _user(records),
                    records.dataset.id,
                    records.document.id,
                    records.segment.id,
                )

    def test_delete_real_child_chunk_and_translate_error(self, app: Flask, records: _Records):
        method = inspect.unwrap(ChildChunkUpdateApi().delete)
        with (
            app.test_request_context("/"),
            _patched_services(records),
            patch.object(segments_module.SegmentService, "get_segment_by_ref", return_value=records.segment),
            patch.object(
                segments_module.SegmentService,
                "get_child_chunk_by_segment_ref",
                return_value=records.child_chunk,
            ),
            patch.object(segments_module.SegmentService, "delete_child_chunk", return_value=None),
        ):
            _, status = method(
                ChildChunkUpdateApi(),
                records.dataset.tenant_id,
                _user(records),
                records.dataset.id,
                records.document.id,
                records.segment.id,
                records.child_chunk.id,
            )
        assert status == 204

        with (
            app.test_request_context("/"),
            _patched_services(records),
            patch.object(segments_module.SegmentService, "get_segment_by_ref", return_value=records.segment),
            patch.object(
                segments_module.SegmentService,
                "get_child_chunk_by_segment_ref",
                return_value=records.child_chunk,
            ),
            patch.object(
                segments_module.SegmentService,
                "delete_child_chunk",
                side_effect=ChildChunkDeleteIndexServiceError("delete failed"),
            ),
        ):
            with pytest.raises(ChildChunkDeleteIndexError):
                method(
                    ChildChunkUpdateApi(),
                    records.dataset.tenant_id,
                    _user(records),
                    records.dataset.id,
                    records.document.id,
                    records.segment.id,
                    records.child_chunk.id,
                )


class TestBatchImport:
    def test_post_uses_persisted_upload(self, app: Flask, database: _Database, records: _Records):
        upload = _upload(database, records)
        payload = {"upload_file_id": upload.id}
        with (
            app.test_request_context("/", json=payload),
            patch.object(type(console_ns), "payload", payload),
            _patched_services(records),
            patch.object(segments_module.redis_client, "setnx", return_value=True),
            patch.object(segments_module.batch_create_segment_to_index_task, "delay") as task,
        ):
            response, status = inspect.unwrap(DatasetDocumentSegmentBatchImportApi().post)(
                DatasetDocumentSegmentBatchImportApi(),
                records.dataset.tenant_id,
                _user(records),
                records.dataset.id,
                records.document.id,
            )

        assert status == 200
        assert response["job_status"] == "waiting"
        assert task.call_args.args[1] == upload.id

    @pytest.mark.parametrize("state", ["missing", "invalid"])
    def test_post_rejects_missing_or_invalid_upload(
        self, app: Flask, database: _Database, records: _Records, state: str
    ):
        upload_id = str(uuid4())
        if state == "invalid":
            upload_id = _upload(database, records, name="segments.txt").id
        payload = {"upload_file_id": upload_id}

        with (
            app.test_request_context("/", json=payload),
            patch.object(type(console_ns), "payload", payload),
            _patched_services(records),
        ):
            error = NotFound if state == "missing" else ValueError
            with pytest.raises(error):
                inspect.unwrap(DatasetDocumentSegmentBatchImportApi().post)(
                    DatasetDocumentSegmentBatchImportApi(),
                    records.dataset.tenant_id,
                    _user(records),
                    records.dataset.id,
                    records.document.id,
                )

    def test_get_job_status_and_missing(self, app: Flask):
        job_id = str(uuid4())
        with (
            app.test_request_context("/"),
            patch.object(segments_module.redis_client, "get", return_value=b"completed"),
        ):
            response, status = inspect.unwrap(DatasetDocumentSegmentBatchImportApi().get)(
                DatasetDocumentSegmentBatchImportApi(), job_id
            )
        assert status == 200
        assert response["job_status"] == "completed"

        with (
            app.test_request_context("/"),
            patch.object(segments_module.redis_client, "get", return_value=None),
        ):
            with pytest.raises(ValueError):
                inspect.unwrap(DatasetDocumentSegmentBatchImportApi().get)(
                    DatasetDocumentSegmentBatchImportApi(), job_id
                )

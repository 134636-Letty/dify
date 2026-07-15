import datetime
import uuid
from collections.abc import Iterator
from types import SimpleNamespace
from typing import override
from unittest.mock import Mock, PropertyMock, patch

import httpx
import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from core.entities.knowledge_entities import PreviewDetail
from core.rag.index_processor.constant.doc_type import DocType
from core.rag.index_processor.index_processor_base import BaseIndexProcessor
from core.rag.models.document import AttachmentDocument, Document
from extensions.ext_database import db
from extensions.storage.storage_type import StorageType
from models import Account, ToolFile
from models.base import TypeBase
from models.enums import CreatorUserRole
from models.model import UploadFile


@pytest.fixture
def database_session(sqlite_engine: Engine) -> Iterator[Session]:
    models = (Account, UploadFile, ToolFile)
    tables = [model.metadata.tables[model.__tablename__] for model in models]
    TypeBase.metadata.create_all(sqlite_engine, tables=tables)

    with Session(sqlite_engine, expire_on_commit=False) as session:
        with (
            patch.object(db, "session", session),
            patch.object(type(db), "engine", new_callable=PropertyMock, return_value=sqlite_engine),
        ):
            yield session


def _account() -> Account:
    return Account(name="Indexing User", email="indexing@example.com")


def _persist_account(session: Session) -> Account:
    account = _account()
    session.add(account)
    session.commit()
    return account


def _persist_upload_file(session: Session, account: Account, *, file_id: str, name: str) -> UploadFile:
    upload_file = UploadFile(
        tenant_id=str(uuid.uuid4()),
        storage_type=StorageType.LOCAL,
        key=f"upload_files/{file_id}/{name}",
        name=name,
        size=4,
        extension="png",
        mime_type="image/png",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by=account.id,
        created_at=datetime.datetime.now(tz=datetime.UTC),
        used=False,
    )
    upload_file.id = file_id
    session.add(upload_file)
    return upload_file


def _persist_tool_file(session: Session, account: Account, *, tool_file_id: str) -> ToolFile:
    tool_file = ToolFile(
        user_id=account.id,
        tenant_id=str(uuid.uuid4()),
        conversation_id=None,
        file_key="k1",
        mimetype="image/png",
        name="tool.png",
        size=4,
    )
    tool_file.id = tool_file_id
    session.add(tool_file)
    session.commit()
    return tool_file


class _ForwardingBaseIndexProcessor(BaseIndexProcessor):
    @override
    def extract(self, extract_setting, **kwargs):
        return super().extract(extract_setting, **kwargs)

    @override
    def transform(self, documents, current_user=None, **kwargs):
        return super().transform(documents, current_user=current_user, **kwargs)

    @override
    def generate_summary_preview(self, tenant_id, preview_texts, summary_index_setting, doc_language=None):
        return super().generate_summary_preview(
            tenant_id=tenant_id,
            preview_texts=preview_texts,
            summary_index_setting=summary_index_setting,
            doc_language=doc_language,
        )

    @override
    def load(self, dataset, documents, multimodal_documents=None, with_keywords=True, **kwargs):
        return super().load(
            dataset=dataset,
            documents=documents,
            multimodal_documents=multimodal_documents,
            with_keywords=with_keywords,
            **kwargs,
        )

    @override
    def clean(self, dataset, node_ids, with_keywords=True, **kwargs):
        return super().clean(dataset=dataset, node_ids=node_ids, with_keywords=with_keywords, **kwargs)

    @override
    def index(self, dataset, document, chunks):
        return super().index(dataset=dataset, document=document, chunks=chunks)

    @override
    def format_preview(self, chunks):
        return super().format_preview(chunks)


class TestBaseIndexProcessor:
    @pytest.fixture
    def processor(self) -> _ForwardingBaseIndexProcessor:
        return _ForwardingBaseIndexProcessor()

    def test_abstract_methods_raise_not_implemented(self, processor: _ForwardingBaseIndexProcessor) -> None:
        with pytest.raises(NotImplementedError):
            processor.extract(Mock())
        with pytest.raises(NotImplementedError):
            processor.transform([])
        with pytest.raises(NotImplementedError):
            processor.generate_summary_preview("tenant", [PreviewDetail(content="c")], {"enable": False})
        with pytest.raises(NotImplementedError):
            processor.load(Mock(), [])
        with pytest.raises(NotImplementedError):
            processor.clean(Mock(), None)
        with pytest.raises(NotImplementedError):
            processor.index(Mock(), Mock(), {})
        with pytest.raises(NotImplementedError):
            processor.format_preview([])

    def test_get_splitter_validates_custom_length(self, processor: _ForwardingBaseIndexProcessor) -> None:
        with patch(
            "core.rag.index_processor.index_processor_base.dify_config.INDEXING_MAX_SEGMENTATION_TOKENS_LENGTH", 1000
        ):
            with pytest.raises(ValueError, match="between 50 and 1000"):
                processor._get_splitter("custom", 49, 0, "", None)
            with pytest.raises(ValueError, match="between 50 and 1000"):
                processor._get_splitter("custom", 1001, 0, "", None)

    def test_get_splitter_custom_mode_uses_fixed_splitter(self, processor: _ForwardingBaseIndexProcessor) -> None:
        fixed_splitter = Mock()
        with patch(
            "core.rag.index_processor.index_processor_base.FixedRecursiveCharacterTextSplitter.from_encoder",
            return_value=fixed_splitter,
        ) as mock_fixed:
            splitter = processor._get_splitter("hierarchical", 120, 10, "\\n\\n", None)

        assert splitter is fixed_splitter
        assert mock_fixed.call_args.kwargs["fixed_separator"] == "\n\n"
        assert mock_fixed.call_args.kwargs["chunk_size"] == 120

    def test_get_splitter_automatic_mode_uses_enhance_splitter(self, processor: _ForwardingBaseIndexProcessor) -> None:
        auto_splitter = Mock()
        with patch(
            "core.rag.index_processor.index_processor_base.EnhanceRecursiveCharacterTextSplitter.from_encoder",
            return_value=auto_splitter,
        ) as mock_enhance:
            splitter = processor._get_splitter("automatic", 0, 0, "", None)

        assert splitter is auto_splitter
        assert "chunk_size" in mock_enhance.call_args.kwargs

    def test_extract_markdown_images(self, processor: _ForwardingBaseIndexProcessor) -> None:
        markdown = "text ![a](https://a/img.png) and ![b](/files/123/file-preview)"
        images = processor._extract_markdown_images(markdown)
        assert images == ["https://a/img.png", "/files/123/file-preview"]

    def test_get_content_files_without_images_returns_empty(self, processor: _ForwardingBaseIndexProcessor) -> None:
        document = Document(page_content="no image markdown", metadata={"document_id": "doc-1", "dataset_id": "ds-1"})
        assert processor._get_content_files(document) == []

    def test_get_content_files_handles_all_sources_and_duplicates(
        self, processor: _ForwardingBaseIndexProcessor, database_session: Session
    ) -> None:
        document = Document(page_content="ignored", metadata={"document_id": "doc-1", "dataset_id": "ds-1"})
        tool_upload_id = str(uuid.uuid4())
        remote_upload_id = str(uuid.uuid4())
        images = [
            "/files/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/image-preview",
            "/files/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/image-preview",
            "/files/bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb/file-preview",
            "/files/tools/cccccccc-cccc-cccc-cccc-cccccccccccc.png",
            "https://example.com/remote.png?x=1",
        ]
        account = _persist_account(database_session)
        _persist_upload_file(
            database_session,
            account,
            file_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            name="a.png",
        )
        _persist_upload_file(
            database_session,
            account,
            file_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            name="b.png",
        )
        _persist_upload_file(database_session, account, file_id=tool_upload_id, name="tool.png")
        _persist_upload_file(database_session, account, file_id=remote_upload_id, name="remote.png")
        database_session.commit()

        with (
            patch.object(processor, "_extract_markdown_images", return_value=images),
            patch.object(processor, "_download_tool_file", return_value=tool_upload_id) as mock_tool_download,
            patch.object(processor, "_download_image", return_value=remote_upload_id) as mock_image_download,
        ):
            files = processor._get_content_files(document, current_user=account)

        assert len(files) == 5
        assert all(isinstance(file, AttachmentDocument) for file in files)
        assert files[0].metadata["doc_type"] == DocType.IMAGE
        assert files[0].metadata["document_id"] == "doc-1"
        assert files[0].metadata["dataset_id"] == "ds-1"
        assert [file.metadata["doc_id"] for file in files] == [
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            tool_upload_id,
            remote_upload_id,
        ]
        mock_tool_download.assert_called_once()
        mock_image_download.assert_called_once()

    def test_get_content_files_skips_tool_and_remote_download_without_user(
        self, processor: _ForwardingBaseIndexProcessor
    ) -> None:
        document = Document(page_content="ignored", metadata={"document_id": "doc-1", "dataset_id": "ds-1"})
        images = ["/files/tools/cccccccc-cccc-cccc-cccc-cccccccccccc.png", "https://example.com/remote.png"]

        with patch.object(processor, "_extract_markdown_images", return_value=images):
            files = processor._get_content_files(document, current_user=None)

        assert files == []

    def test_get_content_files_ignores_missing_upload_records(
        self, processor: _ForwardingBaseIndexProcessor, database_session: Session
    ) -> None:
        document = Document(page_content="ignored", metadata={"document_id": "doc-1", "dataset_id": "ds-1"})
        images = ["/files/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/image-preview"]

        with patch.object(processor, "_extract_markdown_images", return_value=images):
            files = processor._get_content_files(document)

        assert files == []

    def test_download_image_success_with_filename_from_content_disposition(
        self, processor: _ForwardingBaseIndexProcessor, database_session: Session
    ) -> None:
        response = Mock()
        response.headers = {
            "Content-Length": "4",
            "content-disposition": "attachment; filename=test-image.png",
            "content-type": "image/png",
        }
        response.raise_for_status.return_value = None
        response.iter_bytes.return_value = [b"data"]
        upload_result = SimpleNamespace(id="upload-id")
        account = _persist_account(database_session)

        with (
            patch("core.rag.index_processor.index_processor_base.remote_fetcher.make_request", return_value=response),
            patch("services.file_service.FileService") as mock_file_service,
        ):
            mock_file_service.return_value.upload_file.return_value = upload_result
            upload_id = processor._download_image("https://example.com/test.png", current_user=account)

        assert upload_id == "upload-id"
        mock_file_service.return_value.upload_file.assert_called_once()

    def test_download_image_validates_size_and_empty_content(self, processor: _ForwardingBaseIndexProcessor) -> None:
        too_large = Mock()
        too_large.headers = {"Content-Length": str(3 * 1024 * 1024), "content-type": "image/png"}
        too_large.raise_for_status.return_value = None

        with patch("core.rag.index_processor.index_processor_base.remote_fetcher.make_request", return_value=too_large):
            assert processor._download_image("https://example.com/too-large.png", current_user=_account()) is None

        empty = Mock()
        empty.headers = {"Content-Length": "0", "content-type": "image/png"}
        empty.raise_for_status.return_value = None
        empty.iter_bytes.return_value = []

        with patch("core.rag.index_processor.index_processor_base.remote_fetcher.make_request", return_value=empty):
            assert processor._download_image("https://example.com/empty.png", current_user=_account()) is None

    def test_download_image_limits_stream_size(self, processor: _ForwardingBaseIndexProcessor) -> None:
        response = Mock()
        response.headers = {"content-type": "image/png"}
        response.raise_for_status.return_value = None
        response.iter_bytes.return_value = [b"a" * (3 * 1024 * 1024)]

        with patch("core.rag.index_processor.index_processor_base.remote_fetcher.make_request", return_value=response):
            assert processor._download_image("https://example.com/big-stream.png", current_user=_account()) is None

    def test_download_image_handles_timeout_request_and_unexpected_errors(
        self, processor: _ForwardingBaseIndexProcessor
    ) -> None:
        request = httpx.Request("GET", "https://example.com/image.png")

        with patch(
            "core.rag.index_processor.index_processor_base.remote_fetcher.make_request",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            assert processor._download_image("https://example.com/image.png", current_user=_account()) is None

        with patch(
            "core.rag.index_processor.index_processor_base.remote_fetcher.make_request",
            side_effect=httpx.RequestError("bad request", request=request),
        ):
            assert processor._download_image("https://example.com/image.png", current_user=_account()) is None

        with patch(
            "core.rag.index_processor.index_processor_base.remote_fetcher.make_request",
            side_effect=RuntimeError("unexpected"),
        ):
            assert processor._download_image("https://example.com/image.png", current_user=_account()) is None

    def test_download_tool_file_returns_none_when_not_found(
        self, processor: _ForwardingBaseIndexProcessor, database_session: Session
    ) -> None:
        account = _persist_account(database_session)
        assert processor._download_tool_file(str(uuid.uuid4()), current_user=account) is None

    def test_download_tool_file_uploads_file_when_found(
        self, processor: _ForwardingBaseIndexProcessor, database_session: Session
    ) -> None:
        account = _persist_account(database_session)
        tool_file_id = str(uuid.uuid4())
        _persist_tool_file(database_session, account, tool_file_id=tool_file_id)
        upload_result = SimpleNamespace(id="upload-id")

        with (
            patch("core.rag.index_processor.index_processor_base.storage.load_once", return_value=b"blob") as mock_load,
            patch("services.file_service.FileService") as mock_file_service,
        ):
            mock_file_service.return_value.upload_file.return_value = upload_result
            result = processor._download_tool_file(tool_file_id, current_user=account)

        assert result == "upload-id"
        mock_load.assert_called_once_with("k1")
        mock_file_service.return_value.upload_file.assert_called_once()

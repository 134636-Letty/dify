"""SQLite-backed tests for multimodal image handling in ``AppRunner``."""

import base64
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

import core.app.apps.base_app_runner as base_app_runner_module
from core.app.apps.base_app_runner import AppRunner
from core.app.entities.app_invoke_entities import InvokeFrom
from core.app.entities.queue_entities import QueueMessageFileEvent
from graphon.file import FileTransferMethod, FileType
from graphon.model_runtime.entities.message_entities import ImagePromptMessageContent
from models.base import TypeBase
from models.enums import CreatorUserRole, MessageFileBelongsTo
from models.model import MessageFile


@pytest.fixture
def user_id() -> str:
    return str(uuid4())


@pytest.fixture
def tenant_id() -> str:
    return str(uuid4())


@pytest.fixture
def message_id() -> str:
    return str(uuid4())


@pytest.fixture
def queue_manager() -> MagicMock:
    manager = MagicMock()
    manager.invoke_from = InvokeFrom.SERVICE_API
    return manager


@pytest.fixture
def tool_file() -> MagicMock:
    file = MagicMock()
    file.id = str(uuid4())
    return file


@pytest.fixture
def message_file_session(sqlite_engine: Engine) -> Iterator[Session]:
    """Create the one ORM table used by multimodal output persistence."""
    TypeBase.metadata.create_all(sqlite_engine, tables=[MessageFile.__table__])
    with Session(sqlite_engine, expire_on_commit=False) as session:
        yield session


@pytest.fixture
def bind_service_sessionmaker(monkeypatch: pytest.MonkeyPatch, sqlite_engine: Engine) -> None:
    """Point the runner-owned sessionmaker at this test's SQLite engine."""
    monkeypatch.setattr(base_app_runner_module, "db", SimpleNamespace(engine=sqlite_engine))


def _invoke(
    *,
    content: ImagePromptMessageContent,
    message_id: str,
    user_id: str,
    tenant_id: str,
    queue_manager: MagicMock,
) -> None:
    AppRunner._handle_multimodal_image_content(
        MagicMock(),
        content=content,
        message_id=message_id,
        user_id=user_id,
        tenant_id=tenant_id,
        queue_manager=queue_manager,
    )


def _persisted_message_file(session: Session) -> MessageFile:
    return session.scalars(select(MessageFile)).one()


def _assert_no_message_files(session: Session) -> None:
    assert session.scalar(select(func.count()).select_from(MessageFile)) == 0


def _assert_published_file_event(queue_manager: MagicMock, message_file: MessageFile) -> None:
    queue_manager.publish.assert_called_once()
    event = queue_manager.publish.call_args.args[0]
    assert isinstance(event, QueueMessageFileEvent)
    assert event.message_file_id == message_file.id


class TestBaseAppRunnerMultimodal:
    """The runner persists image metadata before publishing its queue event."""

    def test_handle_multimodal_image_content_with_url(
        self,
        monkeypatch: pytest.MonkeyPatch,
        user_id: str,
        tenant_id: str,
        message_id: str,
        queue_manager: MagicMock,
        tool_file: MagicMock,
        message_file_session: Session,
        bind_service_sessionmaker: None,
    ) -> None:
        image_url = "http://example.com/image.png"
        content = ImagePromptMessageContent(url=image_url, format="png", mime_type="image/png")
        tool_file_manager = MagicMock()
        tool_file_manager.create_file_by_url.return_value = tool_file
        monkeypatch.setattr(
            base_app_runner_module,
            "ToolFileManager",
            MagicMock(return_value=tool_file_manager),
        )

        _invoke(
            content=content,
            message_id=message_id,
            user_id=user_id,
            tenant_id=tenant_id,
            queue_manager=queue_manager,
        )

        tool_file_manager.create_file_by_url.assert_called_once_with(
            user_id=user_id,
            tenant_id=tenant_id,
            file_url=image_url,
            conversation_id=None,
        )
        message_file = _persisted_message_file(message_file_session)
        assert message_file.message_id == message_id
        assert message_file.type == FileType.IMAGE
        assert message_file.transfer_method == FileTransferMethod.TOOL_FILE
        assert message_file.belongs_to == MessageFileBelongsTo.ASSISTANT
        assert message_file.url == f"/files/tools/{tool_file.id}"
        assert message_file.upload_file_id == tool_file.id
        assert message_file.created_by == user_id
        assert message_file.created_by_role == CreatorUserRole.END_USER
        _assert_published_file_event(queue_manager, message_file)

    def test_handle_multimodal_image_content_with_base64(
        self,
        monkeypatch: pytest.MonkeyPatch,
        user_id: str,
        tenant_id: str,
        message_id: str,
        queue_manager: MagicMock,
        tool_file: MagicMock,
        message_file_session: Session,
        bind_service_sessionmaker: None,
    ) -> None:
        image_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        )
        content = ImagePromptMessageContent(
            base64_data=base64.b64encode(image_bytes).decode(),
            format="png",
            mime_type="image/png",
        )
        tool_file_manager = MagicMock()
        tool_file_manager.create_file_by_raw.return_value = tool_file
        monkeypatch.setattr(
            base_app_runner_module,
            "ToolFileManager",
            MagicMock(return_value=tool_file_manager),
        )

        _invoke(
            content=content,
            message_id=message_id,
            user_id=user_id,
            tenant_id=tenant_id,
            queue_manager=queue_manager,
        )

        tool_file_manager.create_file_by_raw.assert_called_once()
        call_kwargs = tool_file_manager.create_file_by_raw.call_args.kwargs
        assert call_kwargs == {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "conversation_id": None,
            "file_binary": image_bytes,
            "mimetype": "image/png",
            "filename": "generated_image.png",
        }
        message_file = _persisted_message_file(message_file_session)
        assert message_file.upload_file_id == tool_file.id
        _assert_published_file_event(queue_manager, message_file)

    def test_handle_multimodal_image_content_with_base64_data_uri(
        self,
        monkeypatch: pytest.MonkeyPatch,
        user_id: str,
        tenant_id: str,
        message_id: str,
        queue_manager: MagicMock,
        tool_file: MagicMock,
        message_file_session: Session,
        bind_service_sessionmaker: None,
    ) -> None:
        encoded_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        content = ImagePromptMessageContent(
            base64_data=f"data:image/png;base64,{encoded_image}",
            format="png",
            mime_type="image/png",
        )
        tool_file_manager = MagicMock()
        tool_file_manager.create_file_by_raw.return_value = tool_file
        monkeypatch.setattr(
            base_app_runner_module,
            "ToolFileManager",
            MagicMock(return_value=tool_file_manager),
        )

        _invoke(
            content=content,
            message_id=message_id,
            user_id=user_id,
            tenant_id=tenant_id,
            queue_manager=queue_manager,
        )

        assert tool_file_manager.create_file_by_raw.call_args.kwargs["file_binary"] == base64.b64decode(encoded_image)
        message_file = _persisted_message_file(message_file_session)
        _assert_published_file_event(queue_manager, message_file)

    def test_handle_multimodal_image_content_without_url_or_base64(
        self,
        monkeypatch: pytest.MonkeyPatch,
        user_id: str,
        tenant_id: str,
        message_id: str,
        queue_manager: MagicMock,
        message_file_session: Session,
        bind_service_sessionmaker: None,
    ) -> None:
        content = ImagePromptMessageContent(url="", base64_data="", format="png", mime_type="image/png")
        tool_file_manager = MagicMock()
        tool_file_manager_factory = MagicMock(return_value=tool_file_manager)
        monkeypatch.setattr(
            base_app_runner_module,
            "ToolFileManager",
            tool_file_manager_factory,
        )

        _invoke(
            content=content,
            message_id=message_id,
            user_id=user_id,
            tenant_id=tenant_id,
            queue_manager=queue_manager,
        )

        tool_file_manager_factory.assert_not_called()
        _assert_no_message_files(message_file_session)
        queue_manager.publish.assert_not_called()

    def test_handle_multimodal_image_content_with_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        user_id: str,
        tenant_id: str,
        message_id: str,
        queue_manager: MagicMock,
        message_file_session: Session,
        bind_service_sessionmaker: None,
    ) -> None:
        content = ImagePromptMessageContent(
            url="http://example.com/image.png",
            format="png",
            mime_type="image/png",
        )
        tool_file_manager = MagicMock()
        tool_file_manager.create_file_by_url.side_effect = Exception("Network error")
        monkeypatch.setattr(
            base_app_runner_module,
            "ToolFileManager",
            MagicMock(return_value=tool_file_manager),
        )

        _invoke(
            content=content,
            message_id=message_id,
            user_id=user_id,
            tenant_id=tenant_id,
            queue_manager=queue_manager,
        )

        _assert_no_message_files(message_file_session)
        queue_manager.publish.assert_not_called()

    @pytest.mark.parametrize(
        ("invoke_from", "expected_role"),
        [
            (InvokeFrom.DEBUGGER, CreatorUserRole.ACCOUNT),
            (InvokeFrom.SERVICE_API, CreatorUserRole.END_USER),
        ],
    )
    def test_handle_multimodal_image_content_sets_creator_role(
        self,
        monkeypatch: pytest.MonkeyPatch,
        invoke_from: InvokeFrom,
        expected_role: CreatorUserRole,
        user_id: str,
        tenant_id: str,
        message_id: str,
        queue_manager: MagicMock,
        tool_file: MagicMock,
        message_file_session: Session,
        bind_service_sessionmaker: None,
    ) -> None:
        content = ImagePromptMessageContent(
            url="http://example.com/image.png",
            format="png",
            mime_type="image/png",
        )
        queue_manager.invoke_from = invoke_from
        tool_file_manager = MagicMock()
        tool_file_manager.create_file_by_url.return_value = tool_file
        monkeypatch.setattr(
            base_app_runner_module,
            "ToolFileManager",
            MagicMock(return_value=tool_file_manager),
        )

        _invoke(
            content=content,
            message_id=message_id,
            user_id=user_id,
            tenant_id=tenant_id,
            queue_manager=queue_manager,
        )

        message_file = _persisted_message_file(message_file_session)
        assert message_file.created_by_role == expected_role
        _assert_published_file_event(queue_manager, message_file)

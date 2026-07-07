"""Unit tests for the summary-index generation task session lifecycle."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tasks.generate_summary_index_task as task_module
from core.rag.index_processor.constant.index_type import IndexTechniqueType


class _TrackedSessionContext:
    def __init__(self, session: MagicMock) -> None:
        self.session = session
        self.active = False
        self.exited = False

    def __enter__(self) -> MagicMock:
        self.active = True
        return self.session

    def __exit__(self, exc_type, exc, tb) -> None:
        self.active = False
        self.exited = True


def test_generate_summary_index_task_releases_loader_session_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = SimpleNamespace(
        id="dataset-1",
        indexing_technique=IndexTechniqueType.HIGH_QUALITY,
        summary_index_setting={"enable": True},
        chunk_structure="text_model",
    )
    document = SimpleNamespace(id="document-1", need_summary=True)
    loader_session = MagicMock()
    loader_session.scalar.side_effect = [dataset, document]
    loader_context = _TrackedSessionContext(loader_session)
    monkeypatch.setattr(task_module.session_factory, "create_session", MagicMock(return_value=loader_context))

    observed: dict[str, object] = {}

    def generate_summaries_for_document(**kwargs):
        observed["loader_active"] = loader_context.active
        observed["session"] = kwargs.get("session")
        return []

    monkeypatch.setattr(
        task_module.SummaryIndexService,
        "generate_summaries_for_document",
        generate_summaries_for_document,
    )

    task_module.generate_summary_index_task.run("dataset-1", "document-1")

    assert loader_context.exited is True
    assert observed == {"loader_active": False, "session": None}

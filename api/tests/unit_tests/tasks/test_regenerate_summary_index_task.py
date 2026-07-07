"""Regression tests for regenerate-summary task transaction rollover."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tasks.regenerate_summary_index_task as task_module
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType


class _TrackedSessionContext:
    def __init__(self, session: MagicMock) -> None:
        self.session = session
        self.active = False
        self.transaction_active = False

    def __enter__(self) -> MagicMock:
        self.active = True
        return self.session

    def __exit__(self, exc_type, exc, tb) -> None:
        self.active = False
        self.transaction_active = False

    def query(self, result: object) -> object:
        self.transaction_active = True
        return result


def test_revectorization_rolls_over_loader_transaction_before_external_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = SimpleNamespace(
        id="dataset-1",
        indexing_technique=IndexTechniqueType.HIGH_QUALITY,
        summary_index_setting={"enable": True},
    )
    segment = SimpleNamespace(id="segment-1", document_id="document-1", position=1)
    summary = SimpleNamespace(id="summary-1", summary_index_node_id=None)
    session = MagicMock()
    context = _TrackedSessionContext(session)
    session.scalar.side_effect = lambda *_args: context.query(dataset)
    session.execute.side_effect = lambda *_args: context.query(SimpleNamespace(all=lambda: [(segment, summary)]))
    session.in_transaction.side_effect = lambda: context.transaction_active
    session.commit.side_effect = lambda: setattr(context, "transaction_active", False)
    monkeypatch.setattr(task_module.session_factory, "create_session", MagicMock(return_value=context))

    callback_state: list[tuple[bool, bool]] = []

    def vectorize(*_args: object) -> None:
        callback_state.append((context.active, session.in_transaction()))
        raise RuntimeError("external vectorization failed")

    vectorize_mock = MagicMock(side_effect=vectorize)
    monkeypatch.setattr(task_module.SummaryIndexService, "vectorize_summary", vectorize_mock)

    task_module.regenerate_summary_index_task.run("dataset-1", regenerate_vectors_only=True)

    vectorize_mock.assert_called_once_with(summary, segment, dataset)
    assert callback_state == [(True, False)]
    session.commit.assert_called_once()
    session.add.assert_not_called()


def test_regeneration_rolls_over_lookup_transaction_before_external_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setting = {"enable": True, "model_name": "summary-model"}
    dataset = SimpleNamespace(
        id="dataset-1",
        indexing_technique=IndexTechniqueType.HIGH_QUALITY,
        summary_index_setting=setting,
    )
    document = SimpleNamespace(id="document-1", doc_form=IndexStructureType.PARAGRAPH_INDEX)
    segment = SimpleNamespace(id="segment-1", position=1)
    summary = SimpleNamespace(id="summary-1")
    session = MagicMock()
    context = _TrackedSessionContext(session)
    scalar_results = iter([dataset, summary])
    session.scalar.side_effect = lambda *_args: context.query(next(scalar_results))
    scalar_batches = iter([[document], [segment]])
    session.scalars.side_effect = lambda *_args: context.query(SimpleNamespace(all=lambda: next(scalar_batches)))
    session.in_transaction.side_effect = lambda: context.transaction_active
    session.commit.side_effect = lambda: setattr(context, "transaction_active", False)
    monkeypatch.setattr(task_module.session_factory, "create_session", MagicMock(return_value=context))

    callback_state: list[tuple[bool, bool]] = []

    def generate(*_args: object) -> None:
        callback_state.append((context.active, session.in_transaction()))
        raise RuntimeError("external generation failed")

    generate_mock = MagicMock(side_effect=generate)
    monkeypatch.setattr(task_module.SummaryIndexService, "generate_and_vectorize_summary", generate_mock)

    task_module.regenerate_summary_index_task.run("dataset-1")

    generate_mock.assert_called_once_with(segment, dataset, setting)
    assert callback_state == [(True, False)]
    session.commit.assert_called_once()
    session.add.assert_not_called()

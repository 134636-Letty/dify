from collections.abc import Iterator
from types import SimpleNamespace
from typing import cast

import pytest
from pytest_mock import MockerFixture
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from core.app.entities.app_invoke_entities import InvokeFrom
from models.base import TypeBase
from models.dataset import Document, Pipeline
from models.enums import DataSourceType, DocumentCreatedFrom
from models.model import Account, App, EndUser
from models.workflow import Workflow
from services.rag_pipeline.pipeline_generate_service import PipelineGenerateService


@pytest.fixture
def orm_session(sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[Session]:
    """Provide pipeline generation helpers with persisted SQLite state."""

    TypeBase.metadata.create_all(sqlite_engine, tables=[Pipeline.__table__, Workflow.__table__, Document.__table__])
    monkeypatch.setattr(
        "services.rag_pipeline.rag_pipeline.db",
        SimpleNamespace(engine=sqlite_engine),
    )
    with Session(sqlite_engine, expire_on_commit=False) as session:
        yield session


def _pipeline(*, workflow_id: str | None = None) -> Pipeline:
    pipeline = Pipeline(tenant_id="tenant-1", name="Pipeline", description="description")
    pipeline.id = "pipeline-1"
    pipeline.workflow_id = workflow_id
    return pipeline


def _workflow(*, workflow_id: str, version: str) -> Workflow:
    return Workflow(
        id=workflow_id,
        tenant_id="tenant-1",
        app_id="pipeline-1",
        type="rag-pipeline",
        version=version,
        graph='{"nodes": [], "edges": []}',
        features="{}",
        created_by="user-1",
        environment_variables=[],
        conversation_variables=[],
        rag_pipeline_variables=[],
    )


def _document(document_id: str = "doc-1") -> Document:
    return Document(
        id=document_id,
        tenant_id="tenant-1",
        dataset_id="dataset-1",
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        batch="batch-1",
        name="Document",
        created_from=DocumentCreatedFrom.API,
        created_by="user-1",
    )


def test_get_max_active_requests_uses_smallest_non_zero_limit(mocker: MockerFixture) -> None:
    mocker.patch("services.rag_pipeline.pipeline_generate_service.dify_config.APP_DEFAULT_ACTIVE_REQUESTS", 5)
    mocker.patch("services.rag_pipeline.pipeline_generate_service.dify_config.APP_MAX_ACTIVE_REQUESTS", 3)

    app_model = cast(App, SimpleNamespace(max_active_requests=10))

    result = PipelineGenerateService._get_max_active_requests(app_model)

    assert result == 3


def test_get_max_active_requests_returns_zero_when_all_unlimited(mocker: MockerFixture) -> None:
    mocker.patch("services.rag_pipeline.pipeline_generate_service.dify_config.APP_DEFAULT_ACTIVE_REQUESTS", 0)
    mocker.patch("services.rag_pipeline.pipeline_generate_service.dify_config.APP_MAX_ACTIVE_REQUESTS", 0)

    app_model = cast(App, SimpleNamespace(max_active_requests=0))

    result = PipelineGenerateService._get_max_active_requests(app_model)

    assert result == 0


@pytest.mark.parametrize(
    ("invoke_from", "has_workflow", "expected_error"),
    [
        (InvokeFrom.DEBUGGER, False, "Workflow not initialized"),
        (InvokeFrom.WEB_APP, False, "Workflow not published"),
        (InvokeFrom.DEBUGGER, True, None),
    ],
)
def test_get_workflow(
    invoke_from: InvokeFrom, has_workflow: bool, expected_error: str | None, orm_session: Session
) -> None:
    workflow = _workflow(workflow_id="wf-1", version=Workflow.VERSION_DRAFT)
    pipeline = _pipeline(workflow_id="wf-1" if invoke_from != InvokeFrom.DEBUGGER else None)
    if has_workflow:
        orm_session.add(workflow)
        orm_session.commit()

    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            PipelineGenerateService._get_workflow(pipeline, invoke_from, orm_session)
    else:
        result = PipelineGenerateService._get_workflow(pipeline, invoke_from, orm_session)
        assert result == workflow


def test_generate_updates_document_status_and_returns_event_stream(mocker: MockerFixture, orm_session: Session) -> None:
    pipeline = cast(Pipeline, SimpleNamespace(id="pipeline-1"))
    user = cast(Account | EndUser, SimpleNamespace(id="user-1"))
    args = {"original_document_id": "doc-1", "query": "hello"}

    mocker.patch.object(PipelineGenerateService, "_get_workflow", return_value=SimpleNamespace(id="wf-1"))
    update_status_mock = mocker.patch.object(PipelineGenerateService, "update_document_status")

    generator_cls = mocker.patch("services.rag_pipeline.pipeline_generate_service.PipelineGenerator")
    generator_instance = generator_cls.return_value
    generator_instance.generate.return_value = "raw-events"
    generator_cls.convert_to_event_stream.return_value = "stream-events"

    result = PipelineGenerateService.generate(
        pipeline=pipeline,
        user=user,
        args=args,
        invoke_from=InvokeFrom.WEB_APP,
        streaming=True,
        session=orm_session,
    )

    assert result == "stream-events"
    update_status_mock.assert_called_once_with("doc-1", session=orm_session)


def test_update_document_status_updates_existing_document(orm_session: Session) -> None:
    document = _document()
    document.indexing_status = "completed"
    orm_session.add(document)
    orm_session.commit()

    PipelineGenerateService.update_document_status("doc-1", session=orm_session)

    assert document.indexing_status == "waiting"
    assert document in orm_session


def test_update_document_status_skips_when_document_missing(orm_session: Session) -> None:
    PipelineGenerateService.update_document_status("missing", session=orm_session)
    assert orm_session.get(Document, "missing") is None


# --- generate_single_iteration ---


def test_generate_single_iteration_delegates(mocker: MockerFixture, orm_session: Session) -> None:
    mocker.patch.object(PipelineGenerateService, "_get_workflow", return_value=SimpleNamespace(id="wf-1"))

    generator_cls = mocker.patch("services.rag_pipeline.pipeline_generate_service.PipelineGenerator")
    generator_instance = generator_cls.return_value
    generator_instance.single_iteration_generate.return_value = "raw-iter"
    generator_cls.convert_to_event_stream.return_value = "stream-iter"

    pipeline = cast(Pipeline, SimpleNamespace(id="p1"))
    user = cast(Account, SimpleNamespace(id="u1"))
    result = PipelineGenerateService.generate_single_iteration(pipeline, user, "node-1", {"key": "val"}, orm_session)

    assert result == "stream-iter"
    generator_instance.single_iteration_generate.assert_called_once()


# --- generate_single_loop ---


def test_generate_single_loop_delegates(mocker: MockerFixture, orm_session: Session) -> None:
    mocker.patch.object(PipelineGenerateService, "_get_workflow", return_value=SimpleNamespace(id="wf-1"))

    generator_cls = mocker.patch("services.rag_pipeline.pipeline_generate_service.PipelineGenerator")
    generator_instance = generator_cls.return_value
    generator_instance.single_loop_generate.return_value = "raw-loop"
    generator_cls.convert_to_event_stream.return_value = "stream-loop"

    pipeline = cast(Pipeline, SimpleNamespace(id="p1"))
    user = cast(Account, SimpleNamespace(id="u1"))
    result = PipelineGenerateService.generate_single_loop(pipeline, user, "node-1", {"key": "val"}, orm_session)

    assert result == "stream-loop"
    generator_instance.single_loop_generate.assert_called_once()

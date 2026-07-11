import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import Engine, event, select
from sqlalchemy.orm import Session, scoped_session, sessionmaker

import core.app.apps.pipeline.pipeline_generator as module
from core.app.apps.exc import GenerateTaskStoppedError
from core.app.entities.app_invoke_entities import InvokeFrom
from core.datasource.entities.datasource_entities import DatasourceProviderType
from models.base import TypeBase
from models.dataset import Dataset, Document, DocumentPipelineExecutionLog, Pipeline
from models.enums import DataSourceType, EndUserType
from models.model import EndUser
from models.workflow import Workflow, WorkflowType


class _Patcher:
    def __init__(self, request: pytest.FixtureRequest) -> None:
        self._request = request

    def __call__(self, target: str, *args, **kwargs):
        return self._start(patch(target, *args, **kwargs))

    def object(self, target, attribute: str, *args, **kwargs):
        return self._start(patch.object(target, attribute, *args, **kwargs))

    def _start(self, patcher):
        value = patcher.start()
        self._request.addfinalizer(patcher.stop)
        return value


class _Mocker:
    def __init__(self, request: pytest.FixtureRequest) -> None:
        self.patch = _Patcher(request)


@pytest.fixture
def mocker(request: pytest.FixtureRequest) -> _Mocker:
    """Provide the small patching surface these tests use without a plugin dependency."""
    return _Mocker(request)


@dataclass(frozen=True)
class _SQLiteDb:
    engine: Engine
    session: scoped_session[Session]


@pytest.fixture
def pipeline_db(sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[scoped_session[Session]]:
    """Bind generator request and owned sessions to isolated SQLite."""
    TypeBase.metadata.create_all(
        sqlite_engine,
        tables=[
            Pipeline.__table__,
            Workflow.__table__,
            Dataset.__table__,
            Document.__table__,
            DocumentPipelineExecutionLog.__table__,
            EndUser.__table__,
        ],
    )
    session_registry = scoped_session(sessionmaker(bind=sqlite_engine, expire_on_commit=False))
    monkeypatch.setattr(module, "db", _SQLiteDb(engine=sqlite_engine, session=session_registry))
    try:
        yield session_registry
    finally:
        session_registry.remove()


class FakeRagPipelineGenerateEntity(SimpleNamespace):
    class SingleIterationRunEntity(SimpleNamespace):
        pass

    class SingleLoopRunEntity(SimpleNamespace):
        pass

    def model_dump(self):
        return dict(self.__dict__)


@pytest.fixture
def generator(mocker: _Mocker):
    gen = module.PipelineGenerator()

    mocker.patch.object(module, "RagPipelineGenerateEntity", FakeRagPipelineGenerateEntity)
    mocker.patch.object(module, "RagPipelineInvokeEntity", side_effect=lambda **kwargs: kwargs)
    mocker.patch.object(module.contexts, "plugin_tool_providers", SimpleNamespace(set=MagicMock()))
    mocker.patch.object(module.contexts, "plugin_tool_providers_lock", SimpleNamespace(set=MagicMock()))

    return gen


def _build_pipeline_dataset():
    dataset = Dataset(
        id="ds",
        tenant_id="tenant",
        name="dataset",
        description="desc",
        created_by="user",
        chunk_structure="text_model",
        built_in_field_enabled=True,
        pipeline_id="pipe",
    )
    return dataset


def _build_pipeline():
    pipeline = Pipeline(tenant_id="tenant", name="Pipeline", description="desc")
    pipeline.id = "pipe"
    return pipeline


def _build_workflow():
    return Workflow(
        id="wf",
        tenant_id="tenant",
        app_id="pipe",
        type=WorkflowType.RAG_PIPELINE,
        version=Workflow.VERSION_DRAFT,
        graph='{"nodes": [], "edges": []}',
        features="{}",
        created_by="user",
        environment_variables=[],
        conversation_variables=[],
        rag_pipeline_variables=[],
    )


def _build_user():
    return SimpleNamespace(id="user", name="User", session_id="session")


def _build_end_user() -> EndUser:
    user = EndUser(
        tenant_id="tenant",
        app_id="pipe",
        type=EndUserType.BROWSER,
        name="User",
        session_id="session",
    )
    user.id = "user"
    return user


def _build_args():
    return {
        "inputs": {"k": "v"},
        "start_node_id": "start",
        "datasource_type": DatasourceProviderType.LOCAL_FILE.value,
        "datasource_info_list": [{"name": "file"}],
    }


def _persist_pipeline_records(
    session: Session,
    *,
    include_dataset: bool = True,
    include_workflow: bool = True,
) -> tuple[Pipeline, Workflow | None, Dataset | None]:
    pipeline = _build_pipeline()
    records: list[object] = [pipeline]
    workflow = _build_workflow() if include_workflow else None
    dataset = _build_pipeline_dataset() if include_dataset else None
    if workflow is not None:
        records.append(workflow)
    if dataset is not None:
        records.append(dataset)
    session.add_all(records)
    session.commit()
    return pipeline, workflow, dataset


def _dummy_preserve(*args, **kwargs):
    return contextlib.nullcontext()


def test_generate_dataset_missing(
    generator, mocker: _Mocker, pipeline_db: scoped_session[Session], sqlite_engine: Engine
):
    pipeline, workflow, _ = _persist_pipeline_records(pipeline_db(), include_dataset=False)
    rollbacks: list[object] = []

    def record_rollback(connection) -> None:
        rollbacks.append(connection)

    event.listen(sqlite_engine, "rollback", record_rollback)
    try:
        with pytest.raises(ValueError):
            generator.generate(
                pipeline=pipeline,
                workflow=workflow,
                user=_build_user(),
                args=_build_args(),
                invoke_from=InvokeFrom.WEB_APP,
                streaming=False,
            )
    finally:
        event.remove(sqlite_engine, "rollback", record_rollback)

    assert len(rollbacks) == 1
    assert list(pipeline_db().scalars(select(Document)).all()) == []


def test_generate_debugger_calls_generate(generator, mocker: _Mocker, pipeline_db: scoped_session[Session]):
    pipeline, workflow, _ = _persist_pipeline_records(pipeline_db())

    mocker.patch.object(
        generator,
        "_format_datasource_info_list",
        return_value=[{"name": "file"}],
    )
    mocker.patch.object(
        module.PipelineConfigManager,
        "get_pipeline_config",
        return_value=SimpleNamespace(app_id="pipe", rag_pipeline_variables=[]),
    )
    mocker.patch.object(generator, "_prepare_user_inputs", return_value={"k": "v"})

    mocker.patch.object(
        module.DifyCoreRepositoryFactory,
        "create_workflow_execution_repository",
        return_value=MagicMock(),
    )
    mocker.patch.object(
        module.DifyCoreRepositoryFactory,
        "create_workflow_node_execution_repository",
        return_value=MagicMock(),
    )

    mocker.patch.object(generator, "_generate", return_value={"result": "ok"})

    result = generator.generate(
        pipeline=pipeline,
        workflow=workflow,
        user=_build_user(),
        args=_build_args(),
        invoke_from=InvokeFrom.DEBUGGER,
        streaming=True,
    )

    assert result == {"result": "ok"}


def test_generate_published_pipeline_creates_documents_and_delay(
    generator, mocker: _Mocker, pipeline_db: scoped_session[Session]
):
    pipeline, workflow, _ = _persist_pipeline_records(pipeline_db())

    datasource_info_list = [{"name": "file1"}, {"name": "file2"}]

    mocker.patch.object(
        generator,
        "_format_datasource_info_list",
        return_value=datasource_info_list,
    )
    mocker.patch.object(
        module.PipelineConfigManager,
        "get_pipeline_config",
        return_value=SimpleNamespace(app_id="pipe", rag_pipeline_variables=[]),
    )
    mocker.patch.object(generator, "_prepare_user_inputs", return_value={"k": "v"})

    mocker.patch("services.dataset_service.DocumentService.get_documents_position", side_effect=[1, 2])
    features = SimpleNamespace()
    mocker.patch("services.feature_service.FeatureService.get_features", return_value=features)
    check_limits = mocker.patch("services.dataset_service.DocumentService.check_document_creation_limits")

    mocker.patch.object(
        module.DifyCoreRepositoryFactory,
        "create_workflow_execution_repository",
        return_value=MagicMock(),
    )
    mocker.patch.object(
        module.DifyCoreRepositoryFactory,
        "create_workflow_node_execution_repository",
        return_value=MagicMock(),
    )

    task_proxy = MagicMock()
    mocker.patch.object(module, "RagPipelineTaskProxy", return_value=task_proxy)

    result = generator.generate(
        pipeline=pipeline,
        workflow=workflow,
        user=_build_user(),
        args=_build_args(),
        invoke_from=InvokeFrom.PUBLISHED_PIPELINE,
        streaming=False,
    )

    assert result["batch"]
    assert len(result["documents"]) == 2
    check_limits.assert_called_once_with(len(datasource_info_list), features)
    task_proxy.delay.assert_called_once()
    database_session = pipeline_db()
    database_session.expire_all()
    documents = list(database_session.scalars(select(Document).order_by(Document.position)).all())
    logs = list(database_session.scalars(select(DocumentPipelineExecutionLog)).all())
    assert [document.name for document in documents] == ["file1", "file2"]
    assert {log.document_id for log in logs} == {document.id for document in documents}


def test_generate_published_pipeline_rejects_when_document_creation_limits_exceeded(
    generator, mocker: _Mocker, pipeline_db: scoped_session[Session]
):
    pipeline, workflow, _ = _persist_pipeline_records(pipeline_db())

    datasource_info_list = [{"name": "file1"}, {"name": "file2"}]
    mocker.patch.object(
        generator,
        "_format_datasource_info_list",
        return_value=datasource_info_list,
    )
    mocker.patch.object(
        module.PipelineConfigManager,
        "get_pipeline_config",
        return_value=SimpleNamespace(app_id="pipe", rag_pipeline_variables=[]),
    )

    features = SimpleNamespace()
    mocker.patch("services.feature_service.FeatureService.get_features", return_value=features)
    check_limits = mocker.patch(
        "services.dataset_service.DocumentService.check_document_creation_limits",
        side_effect=ValueError("document limit exceeded"),
    )

    with pytest.raises(ValueError, match="document limit exceeded"):
        generator.generate(
            pipeline=pipeline,
            workflow=workflow,
            user=_build_user(),
            args=_build_args(),
            invoke_from=InvokeFrom.PUBLISHED_PIPELINE,
            streaming=False,
        )

    check_limits.assert_called_once_with(len(datasource_info_list), features)
    assert list(pipeline_db().scalars(select(Document)).all()) == []
    assert list(pipeline_db().scalars(select(DocumentPipelineExecutionLog)).all()) == []


def test_generate_is_retry_calls_generate(generator, mocker: _Mocker, pipeline_db: scoped_session[Session]):
    pipeline, workflow, _ = _persist_pipeline_records(pipeline_db())

    mocker.patch.object(
        generator,
        "_format_datasource_info_list",
        return_value=[{"name": "file"}],
    )
    mocker.patch.object(
        module.PipelineConfigManager,
        "get_pipeline_config",
        return_value=SimpleNamespace(app_id="pipe", rag_pipeline_variables=[]),
    )
    mocker.patch.object(generator, "_prepare_user_inputs", return_value={"k": "v"})

    mocker.patch.object(
        module.DifyCoreRepositoryFactory,
        "create_workflow_execution_repository",
        return_value=MagicMock(),
    )
    mocker.patch.object(
        module.DifyCoreRepositoryFactory,
        "create_workflow_node_execution_repository",
        return_value=MagicMock(),
    )

    mocker.patch.object(generator, "_generate", return_value={"result": "ok"})

    result = generator.generate(
        pipeline=pipeline,
        workflow=workflow,
        user=_build_user(),
        args=_build_args(),
        invoke_from=InvokeFrom.PUBLISHED_PIPELINE,
        streaming=True,
        is_retry=True,
    )

    assert result == {"result": "ok"}


def test_generate_worker_handles_errors(generator, mocker: _Mocker, pipeline_db: scoped_session[Session]):
    flask_app = MagicMock()
    flask_app.app_context.return_value = contextlib.nullcontext()
    mocker.patch.object(module, "preserve_flask_contexts", _dummy_preserve)
    database_session = pipeline_db()
    database_session.add_all([_build_workflow(), _build_end_user()])
    database_session.commit()

    application_generate_entity = FakeRagPipelineGenerateEntity(
        app_config=SimpleNamespace(tenant_id="tenant", app_id="pipe", workflow_id="wf"),
        invoke_from=InvokeFrom.WEB_APP,
        user_id="user",
    )

    runner_instance = MagicMock()
    runner_instance.run.side_effect = ValueError("bad")
    mocker.patch.object(module, "PipelineRunner", return_value=runner_instance)

    queue_manager = MagicMock()
    generator._generate_worker(
        flask_app=flask_app,
        application_generate_entity=application_generate_entity,
        queue_manager=queue_manager,
        context=contextlib.nullcontext(),
        variable_loader=MagicMock(),
        workflow_execution_repository=MagicMock(),
        workflow_node_execution_repository=MagicMock(),
    )

    queue_manager.publish_error.assert_called_once()
    assert database_session.in_transaction() is False


def test_generate_worker_sets_system_user_id_for_external_call(
    generator, mocker: _Mocker, pipeline_db: scoped_session[Session]
):
    flask_app = MagicMock()
    flask_app.app_context.return_value = contextlib.nullcontext()
    mocker.patch.object(module, "preserve_flask_contexts", _dummy_preserve)
    database_session = pipeline_db()
    database_session.add_all([_build_workflow(), _build_end_user()])
    database_session.commit()

    application_generate_entity = FakeRagPipelineGenerateEntity(
        app_config=SimpleNamespace(tenant_id="tenant", app_id="pipe", workflow_id="wf"),
        invoke_from=InvokeFrom.WEB_APP,
        user_id="user",
    )

    runner_instance = MagicMock()
    mocker.patch.object(module, "PipelineRunner", return_value=runner_instance)

    generator._generate_worker(
        flask_app=flask_app,
        application_generate_entity=application_generate_entity,
        queue_manager=MagicMock(),
        context=contextlib.nullcontext(),
        variable_loader=MagicMock(),
        workflow_execution_repository=MagicMock(),
        workflow_node_execution_repository=MagicMock(),
    )

    assert module.PipelineRunner.call_args.kwargs["system_user_id"] == "session"
    assert isinstance(module.PipelineRunner.call_args.kwargs["workflow"], Workflow)
    assert database_session.in_transaction() is False


def test_generate_raises_when_workflow_not_found(generator, mocker: _Mocker, pipeline_db: scoped_session[Session]):
    flask_app = MagicMock()
    mocker.patch.object(module, "preserve_flask_contexts", _dummy_preserve)
    pipeline, _, _ = _persist_pipeline_records(pipeline_db(), include_dataset=False, include_workflow=False)

    with pytest.raises(ValueError):
        generator._generate(
            flask_app=flask_app,
            context=contextlib.nullcontext(),
            pipeline=pipeline,
            workflow_id="wf",
            user=_build_user(),
            application_generate_entity=FakeRagPipelineGenerateEntity(
                task_id="t",
                app_config=SimpleNamespace(app_id="pipe"),
                user_id="user",
                invoke_from=InvokeFrom.DEBUGGER,
            ),
            invoke_from=InvokeFrom.DEBUGGER,
            workflow_execution_repository=MagicMock(),
            workflow_node_execution_repository=MagicMock(),
            streaming=True,
        )


def test_generate_success_returns_converted(generator, mocker: _Mocker, pipeline_db: scoped_session[Session]):
    flask_app = MagicMock()
    mocker.patch.object(module, "preserve_flask_contexts", _dummy_preserve)
    pipeline, workflow, _ = _persist_pipeline_records(pipeline_db())

    queue_manager = MagicMock()
    mocker.patch.object(module, "PipelineQueueManager", return_value=queue_manager)

    worker_thread = MagicMock()
    mocker.patch.object(module.threading, "Thread", return_value=worker_thread)

    mocker.patch.object(generator, "_get_draft_var_saver_factory", return_value=MagicMock())
    mocker.patch.object(generator, "_handle_response", return_value="response")
    mocker.patch.object(module.WorkflowAppGenerateResponseConverter, "convert", return_value="converted")

    result = generator._generate(
        flask_app=flask_app,
        context=contextlib.nullcontext(),
        pipeline=pipeline,
        workflow_id="wf",
        user=_build_user(),
        application_generate_entity=FakeRagPipelineGenerateEntity(
            task_id="t",
            app_config=SimpleNamespace(app_id="pipe"),
            user_id="user",
            invoke_from=InvokeFrom.DEBUGGER,
        ),
        invoke_from=InvokeFrom.DEBUGGER,
        workflow_execution_repository=MagicMock(),
        workflow_node_execution_repository=MagicMock(),
        streaming=True,
    )

    assert result == "converted"
    assert module.WorkflowAppGenerateResponseConverter.convert.call_args.kwargs["response"] == "response"


def test_single_iteration_generate_validates_inputs(generator, mocker: _Mocker):
    with pytest.raises(ValueError):
        generator.single_iteration_generate(_build_pipeline(), _build_workflow(), "", _build_user(), {})

    with pytest.raises(ValueError):
        generator.single_iteration_generate(
            _build_pipeline(), _build_workflow(), "node", _build_user(), {"inputs": None}
        )


def test_single_iteration_generate_dataset_required(generator, mocker: _Mocker, pipeline_db: scoped_session[Session]):
    pipeline, workflow, _ = _persist_pipeline_records(pipeline_db(), include_dataset=False)

    with pytest.raises(ValueError):
        generator.single_iteration_generate(
            pipeline,
            workflow,
            "node",
            _build_user(),
            {"inputs": {"a": 1}},
        )


def test_single_iteration_generate_success(generator, mocker: _Mocker, pipeline_db: scoped_session[Session]):
    pipeline, workflow, _ = _persist_pipeline_records(pipeline_db())

    mocker.patch.object(
        module.PipelineConfigManager,
        "get_pipeline_config",
        return_value=SimpleNamespace(app_id="pipe", tenant_id="tenant"),
    )
    mocker.patch.object(
        module.DifyCoreRepositoryFactory,
        "create_workflow_execution_repository",
        return_value=MagicMock(),
    )
    mocker.patch.object(
        module.DifyCoreRepositoryFactory,
        "create_workflow_node_execution_repository",
        return_value=MagicMock(),
    )
    draft_service = MagicMock()
    draft_service_factory = mocker.patch.object(module, "WorkflowDraftVariableService", return_value=draft_service)
    mocker.patch.object(module, "DraftVarLoader", return_value=MagicMock())

    mocker.patch.object(generator, "_generate", return_value={"ok": True})

    result = generator.single_iteration_generate(
        pipeline,
        workflow,
        "node",
        _build_user(),
        {"inputs": {"a": 1}},
        streaming=False,
    )

    assert result == {"ok": True}
    assert isinstance(draft_service_factory.call_args.args[0], Session)
    draft_service.prefill_conversation_variable_default_values.assert_called_once_with(workflow, user_id="user")


def test_single_loop_generate_success(generator, mocker: _Mocker, pipeline_db: scoped_session[Session]):
    pipeline, workflow, _ = _persist_pipeline_records(pipeline_db())

    mocker.patch.object(
        module.PipelineConfigManager,
        "get_pipeline_config",
        return_value=SimpleNamespace(app_id="pipe", tenant_id="tenant"),
    )
    mocker.patch.object(
        module.DifyCoreRepositoryFactory,
        "create_workflow_execution_repository",
        return_value=MagicMock(),
    )
    mocker.patch.object(
        module.DifyCoreRepositoryFactory,
        "create_workflow_node_execution_repository",
        return_value=MagicMock(),
    )
    draft_service = MagicMock()
    draft_service_factory = mocker.patch.object(module, "WorkflowDraftVariableService", return_value=draft_service)
    mocker.patch.object(module, "DraftVarLoader", return_value=MagicMock())

    mocker.patch.object(generator, "_generate", return_value={"ok": True})

    result = generator.single_loop_generate(
        pipeline,
        workflow,
        "node",
        _build_user(),
        {"inputs": {"a": 1}},
        streaming=False,
    )

    assert result == {"ok": True}
    assert isinstance(draft_service_factory.call_args.args[0], Session)
    draft_service.prefill_conversation_variable_default_values.assert_called_once_with(workflow, user_id="user")


def test_handle_response_value_error_triggers_generate_task_stopped(generator, mocker: _Mocker):
    pipeline = _build_pipeline()
    workflow = _build_workflow()
    app_entity = FakeRagPipelineGenerateEntity(task_id="t")

    task_pipeline = MagicMock()
    task_pipeline.process.side_effect = ValueError("I/O operation on closed file.")
    mocker.patch.object(module, "WorkflowAppGenerateTaskPipeline", return_value=task_pipeline)

    with pytest.raises(GenerateTaskStoppedError):
        generator._handle_response(
            application_generate_entity=app_entity,
            workflow=workflow,
            queue_manager=MagicMock(),
            user=_build_user(),
            draft_var_saver_factory=MagicMock(),
            stream=False,
        )


def test_build_document_sets_metadata_for_builtin_fields(generator, mocker: _Mocker):
    class DummyDocument(SimpleNamespace):
        pass

    mocker.patch.object(module, "Document", side_effect=lambda **kwargs: DummyDocument(**kwargs))

    document = generator._build_document(
        tenant_id="tenant",
        dataset_id="ds",
        built_in_field_enabled=True,
        datasource_type=DatasourceProviderType.LOCAL_FILE,
        datasource_info={"name": "file"},
        created_from="rag-pipeline",
        position=1,
        account=_build_user(),
        batch="batch",
        document_form="text",
    )

    assert document.name == "file"
    assert document.doc_metadata


def test_build_document_supports_online_drive_datasource_type(generator):
    document = generator._build_document(
        tenant_id="tenant",
        dataset_id="ds",
        built_in_field_enabled=True,
        datasource_type=DatasourceProviderType.ONLINE_DRIVE,
        datasource_info={"id": "file-1", "bucket": "bucket-1", "name": "drive.pdf", "type": "file"},
        created_from="rag-pipeline",
        position=1,
        account=_build_user(),
        batch="batch",
        document_form="text",
    )

    assert DataSourceType(document.data_source_type) == DataSourceType.ONLINE_DRIVE
    assert document.name == "drive.pdf"


def test_build_document_invalid_datasource_type(generator):
    with pytest.raises(ValueError):
        generator._build_document(
            tenant_id="tenant",
            dataset_id="ds",
            built_in_field_enabled=False,
            datasource_type="invalid",
            datasource_info={},
            created_from="rag-pipeline",
            position=1,
            account=_build_user(),
            batch="batch",
            document_form="text",
        )


def test_format_datasource_info_list_non_online_drive(generator):
    result = generator._format_datasource_info_list(
        DatasourceProviderType.LOCAL_FILE,
        [{"name": "file"}],
        _build_pipeline(),
        _build_workflow(),
        "start",
        _build_user(),
    )

    assert result == [{"name": "file"}]


def test_format_datasource_info_list_missing_node_data(generator):
    workflow = MagicMock(graph_dict={"nodes": []})

    with pytest.raises(ValueError):
        generator._format_datasource_info_list(
            DatasourceProviderType.ONLINE_DRIVE,
            [],
            _build_pipeline(),
            workflow,
            "start",
            _build_user(),
        )


def test_format_datasource_info_list_online_drive_folder(generator, mocker: _Mocker):
    workflow = MagicMock(
        graph_dict={
            "nodes": [
                {
                    "id": "start",
                    "data": {
                        "plugin_id": "p",
                        "provider_name": "provider",
                        "datasource_name": "drive",
                        "credential_id": "cred",
                    },
                }
            ]
        }
    )

    runtime = MagicMock()
    runtime.runtime = SimpleNamespace(credentials=None)
    runtime.datasource_provider_type.return_value = DatasourceProviderType.ONLINE_DRIVE

    mocker.patch(
        "core.datasource.datasource_manager.DatasourceManager.get_datasource_runtime",
        return_value=runtime,
    )
    mocker.patch.object(module.DatasourceProviderService, "get_datasource_credentials", return_value={"k": "v"})

    mocker.patch.object(
        generator,
        "_get_files_in_folder",
        side_effect=lambda *args, **kwargs: args[4].append({"id": "f"}),
    )

    result = generator._format_datasource_info_list(
        DatasourceProviderType.ONLINE_DRIVE,
        [{"id": "folder", "type": "folder", "name": "Folder", "bucket": "b"}],
        _build_pipeline(),
        workflow,
        "start",
        _build_user(),
    )

    assert result == [{"id": "f"}]


def test_get_files_in_folder_recurses_and_collects(generator):
    class File:
        def __init__(self, id, name, type):
            self.id = id
            self.name = name
            self.type = type

    class FilesPage:
        def __init__(self, files, is_truncated=False, next_page_parameters=None):
            self.files = files
            self.is_truncated = is_truncated
            self.next_page_parameters = next_page_parameters

    class Result:
        def __init__(self, result):
            self.result = result

    class Runtime:
        def __init__(self):
            self.calls = []

        def datasource_provider_type(self):
            return DatasourceProviderType.ONLINE_DRIVE

        def online_drive_browse_files(self, user_id, request, provider_type):
            self.calls.append(request.next_page_parameters)
            if request.prefix == "fd":
                return iter([Result([FilesPage([File("f2", "file2", "file")], False, None)])])
            if request.next_page_parameters is None:
                return iter(
                    [
                        Result(
                            [FilesPage([File("f1", "file", "file"), File("fd", "folder", "folder")], True, {"page": 2})]
                        )
                    ]
                )
            return iter([Result([FilesPage([File("f2", "file2", "file")], False, None)])])

    runtime = Runtime()
    all_files = []

    generator._get_files_in_folder(
        datasource_runtime=runtime,
        prefix="root",
        bucket="b",
        user_id="user",
        all_files=all_files,
        datasource_info={},
    )

    assert {f["id"] for f in all_files} == {"f1", "f2"}


def test_get_files_in_folder_handles_empty_folder(generator):
    """An empty folder must return an empty file list without recursion errors."""

    class FilesPage:
        def __init__(self, files, is_truncated=False, next_page_parameters=None):
            self.files = files
            self.is_truncated = is_truncated
            self.next_page_parameters = next_page_parameters

    class Result:
        def __init__(self, result):
            self.result = result

    class Runtime:
        def datasource_provider_type(self):
            return DatasourceProviderType.ONLINE_DRIVE

        def online_drive_browse_files(self, user_id, request, provider_type):
            # Empty folder: returns a page with no files, not truncated
            return iter([Result([FilesPage([], False, None)])])

    runtime = Runtime()
    all_files: list = []

    generator._get_files_in_folder(
        datasource_runtime=runtime,
        prefix="empty-folder",
        bucket="b",
        user_id="user",
        all_files=all_files,
        datasource_info={},
    )

    assert all_files == []


def test_get_files_in_folder_handles_empty_folder_with_false_truncation(generator):
    """An empty folder that incorrectly reports is_truncated=True must not recurse forever."""

    call_count = 0

    class FilesPage:
        def __init__(self, files, is_truncated=False, next_page_parameters=None):
            self.files = files
            self.is_truncated = is_truncated
            self.next_page_parameters = next_page_parameters

    class Result:
        def __init__(self, result):
            self.result = result

    class Runtime:
        def datasource_provider_type(self):
            return DatasourceProviderType.ONLINE_DRIVE

        def online_drive_browse_files(self, user_id, request, provider_type):
            nonlocal call_count
            call_count += 1
            # Empty folder that incorrectly claims truncation
            return iter([Result([FilesPage([], True, {"page": 2})])])

    runtime = Runtime()
    all_files: list = []

    generator._get_files_in_folder(
        datasource_runtime=runtime,
        prefix="buggy-folder",
        bucket="b",
        user_id="user",
        all_files=all_files,
        datasource_info={},
    )

    assert all_files == []
    # Should only be called once -- the empty-page guard prevents further recursion
    assert call_count == 1


def test_get_files_in_folder_handles_self_referencing_folder(generator):
    """A folder that lists itself as a child must not recurse infinitely."""

    class File:
        def __init__(self, id, name, type):
            self.id = id
            self.name = name
            self.type = type

    class FilesPage:
        def __init__(self, files, is_truncated=False, next_page_parameters=None):
            self.files = files
            self.is_truncated = is_truncated
            self.next_page_parameters = next_page_parameters

    class Result:
        def __init__(self, result):
            self.result = result

    call_count = 0

    class Runtime:
        def datasource_provider_type(self):
            return DatasourceProviderType.ONLINE_DRIVE

        def online_drive_browse_files(self, user_id, request, provider_type):
            nonlocal call_count
            call_count += 1
            # The folder returns itself as a child (self-reference)
            return iter([Result([FilesPage([File("self-ref", "myfolder", "folder")], False, None)])])

    runtime = Runtime()
    all_files: list = []

    generator._get_files_in_folder(
        datasource_runtime=runtime,
        prefix="self-ref",
        bucket="b",
        user_id="user",
        all_files=all_files,
        datasource_info={},
    )

    assert all_files == []
    # Should only be called once -- the visited-set guard prevents re-entry
    assert call_count == 1

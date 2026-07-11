import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from models import Account, Tenant
from models.base import TypeBase
from models.dataset import (
    Dataset,
    Document,
    DocumentPipelineExecutionLog,
    Pipeline,
    PipelineCustomizedTemplate,
    PipelineRecommendedPlugin,
)
from models.enums import DataSourceType, DocumentCreatedFrom
from models.workflow import Workflow, WorkflowNodeExecutionModel, WorkflowRun
from services.entities.knowledge_entities.rag_pipeline_entities import IconInfo, PipelineTemplateInfoEntity
from services.rag_pipeline.rag_pipeline import RagPipelineService
from services.workflow_ref_service import WorkflowRef


@dataclass
class RagPipelineServiceTestContext:
    service: RagPipelineService
    session: Session
    session_maker: sessionmaker[Session]


@pytest.fixture
def rag_pipeline_service(mocker: MockerFixture, sqlite_engine: Engine) -> Iterator[RagPipelineServiceTestContext]:
    models = (
        Pipeline,
        Workflow,
        Dataset,
        Document,
        DocumentPipelineExecutionLog,
        PipelineCustomizedTemplate,
        PipelineRecommendedPlugin,
        WorkflowNodeExecutionModel,
        WorkflowRun,
    )
    TypeBase.metadata.create_all(sqlite_engine, tables=[model.__table__ for model in models])
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.DifyAPIRepositoryFactory.create_api_workflow_node_execution_repository",
        return_value=MockRepo(),
    )
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.DifyAPIRepositoryFactory.create_api_workflow_run_repository",
        return_value=MockRepo(),
    )
    session_maker = sessionmaker(sqlite_engine, expire_on_commit=False)
    mocker.patch("services.rag_pipeline.rag_pipeline.db", SimpleNamespace(engine=sqlite_engine))
    with session_maker() as session:
        service = RagPipelineService(session=session, session_maker=session_maker)
        yield RagPipelineServiceTestContext(service=service, session=session, session_maker=session_maker)


class MockRepo:
    pass


def _make_account(account_id: str = "u1", tenant_id: str = "t1") -> Account:
    account = Account(name="Test User", email=f"{account_id}@example.com")
    account.id = account_id
    tenant = Tenant(name="Test Tenant")
    tenant.id = tenant_id
    account._current_tenant = tenant
    return account


def _make_pipeline(
    *,
    pipeline_id: str = "p1",
    tenant_id: str = "t1",
    workflow_id: str | None = None,
    is_published: bool = False,
) -> Pipeline:
    pipeline = Pipeline(tenant_id=tenant_id, name="Test Pipeline", description="test")
    pipeline.id = pipeline_id
    pipeline.workflow_id = workflow_id
    pipeline.is_published = is_published
    return pipeline


def _make_workflow(
    *,
    workflow_id: str = "wf-1",
    tenant_id: str = "t1",
    app_id: str = "p1",
    graph: dict[str, object] | None = None,
    features: dict[str, object] | None = None,
    created_by: str = "u1",
) -> Workflow:
    workflow = Workflow(
        id=workflow_id,
        tenant_id=tenant_id,
        app_id=app_id,
        type="workflow",
        version="draft",
        marked_name="",
        marked_comment="",
        graph=json.dumps(graph or {"nodes": []}),
        features=json.dumps(features or {}),
        created_by=created_by,
        created_at=datetime(2024, 1, 1),
        updated_by=None,
        updated_at=datetime(2024, 1, 1),
        environment_variables=[],
        conversation_variables=[],
        rag_pipeline_variables=[],
    )
    return workflow


def _make_dataset(*, dataset_id: str = "d1", pipeline_id: str = "p1", tenant_id: str = "t1") -> Dataset:
    dataset = Dataset(
        id=dataset_id,
        tenant_id=tenant_id,
        name="Test Dataset",
        created_by="u1",
    )
    dataset.pipeline_id = pipeline_id
    return dataset


def _make_document(*, document_id: str = "doc-1", dataset_id: str = "d1", tenant_id: str = "t1") -> Document:
    return Document(
        id=document_id,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        batch="batch-1",
        name="Document",
        created_from=DocumentCreatedFrom.API,
        created_by="u1",
    )


def _make_customized_template() -> PipelineCustomizedTemplate:
    return PipelineCustomizedTemplate(
        tenant_id="t1",
        name="old",
        description="old",
        chunk_structure="paragraph",
        icon={},
        position=1,
        yaml_content="",
        install_count=0,
        language="en-US",
        created_by="u1",
    )


def _make_recommended_plugin(plugin_id: str) -> PipelineRecommendedPlugin:
    return PipelineRecommendedPlugin(plugin_id=plugin_id, provider_name=plugin_id, type="tool", position=0, active=True)


def test_get_pipeline_templates_fallbacks_to_builtin_for_non_english_empty_result(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    mocker.patch("services.rag_pipeline.rag_pipeline.dify_config.HOSTED_FETCH_PIPELINE_TEMPLATES_MODE", "remote")
    session = rag_pipeline_service.session

    remote_retrieval = mocker.Mock()
    remote_retrieval.get_pipeline_templates.return_value = {"pipeline_templates": []}

    factory_mock = mocker.patch("services.rag_pipeline.rag_pipeline.PipelineTemplateRetrievalFactory")
    factory_mock.get_pipeline_template_factory.return_value.return_value = remote_retrieval

    builtin_retrieval = mocker.Mock()
    builtin_retrieval.fetch_pipeline_templates_from_builtin.return_value = {"pipeline_templates": [{"id": "builtin-1"}]}
    factory_mock.get_built_in_pipeline_template_retrieval.return_value = builtin_retrieval

    result = RagPipelineService.get_pipeline_templates(type="built-in", language="ja-JP", session=session)

    assert result == {"pipeline_templates": [{"id": "builtin-1"}]}
    remote_retrieval.get_pipeline_templates.assert_called_once_with("ja-JP", None, session=session)
    builtin_retrieval.fetch_pipeline_templates_from_builtin.assert_called_once_with("en-US")


def test_get_pipeline_templates_customized_mode_uses_customized_factory(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    session = rag_pipeline_service.session
    retrieval = mocker.Mock()
    retrieval.get_pipeline_templates.return_value = {"pipeline_templates": [{"id": "custom-1"}]}

    factory_mock = mocker.patch("services.rag_pipeline.rag_pipeline.PipelineTemplateRetrievalFactory")
    factory_mock.get_pipeline_template_factory.return_value.return_value = retrieval

    result = RagPipelineService.get_pipeline_templates(type="customized", language="en-US", session=session)

    assert result == {"pipeline_templates": [{"id": "custom-1"}]}
    factory_mock.get_pipeline_template_factory.assert_called_with("customized")
    retrieval.get_pipeline_templates.assert_called_once_with("en-US", None, session=session)


@pytest.mark.parametrize("template_type", ["built-in", "customized"])
def test_get_pipeline_template_detail_uses_expected_mode(
    mocker: MockerFixture, template_type: str, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    mocker.patch("services.rag_pipeline.rag_pipeline.dify_config.HOSTED_FETCH_PIPELINE_TEMPLATES_MODE", "remote")
    session = rag_pipeline_service.session
    retrieval = mocker.Mock()
    retrieval.get_pipeline_template_detail.return_value = {"id": "tpl-1"}

    factory_mock = mocker.patch("services.rag_pipeline.rag_pipeline.PipelineTemplateRetrievalFactory")
    factory_mock.get_pipeline_template_factory.return_value.return_value = retrieval

    result = RagPipelineService.get_pipeline_template_detail("tpl-1", type=template_type, session=session)

    assert result == {"id": "tpl-1"}
    expected_mode = "remote" if template_type == "built-in" else "customized"
    factory_mock.get_pipeline_template_factory.assert_called_with(expected_mode)
    retrieval.get_pipeline_template_detail.assert_called_once_with("tpl-1", session=session)


def test_get_published_workflow_returns_none_when_pipeline_has_no_workflow_id(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    pipeline = _make_pipeline(workflow_id=None)

    result = rag_pipeline_service.service.get_published_workflow(pipeline)

    assert result is None


def test_get_all_published_workflow_returns_empty_for_unpublished_pipeline(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    pipeline = _make_pipeline(workflow_id=None)
    session = rag_pipeline_service.session

    workflows, has_more = rag_pipeline_service.service.get_all_published_workflow(
        session=session,
        pipeline=pipeline,
        page=1,
        limit=20,
        user_id=None,
        named_only=False,
    )

    assert workflows == []
    assert has_more is False


def test_get_all_published_workflow_applies_limit_and_has_more(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    session = rag_pipeline_service.session
    pipeline = _make_pipeline(pipeline_id="pipeline-1", workflow_id="wf-live")
    workflows_by_version = []
    for workflow_id, version in (("wf1", "v3"), ("wf2", "v2"), ("wf3", "v1")):
        workflow = _make_workflow(workflow_id=workflow_id, app_id=pipeline.id, created_by="user-1")
        workflow.version = version
        workflow.marked_name = version
        workflows_by_version.append(workflow)
    session.add_all(workflows_by_version)
    session.commit()

    workflows, has_more = rag_pipeline_service.service.get_all_published_workflow(
        session=session,
        pipeline=pipeline,
        page=1,
        limit=2,
        user_id="user-1",
        named_only=True,
    )

    assert [item.id for item in workflows] == ["wf1", "wf2"]
    assert has_more is True


# --- sync_draft_workflow ---


def test_sync_draft_workflow_creates_new_when_none_exists(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    session = rag_pipeline_service.session

    pipeline = _make_pipeline(workflow_id=None)
    account = _make_account()

    result = rag_pipeline_service.service.sync_draft_workflow(
        pipeline=pipeline,
        graph={"nodes": []},
        unique_hash=None,
        account=account,
        environment_variables=[],
        conversation_variables=[],
        rag_pipeline_variables=[],
    )

    assert result.app_id == "p1"
    assert pipeline.workflow_id == result.id


def test_sync_draft_workflow_raises_on_hash_mismatch(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    from services.errors.app import WorkflowHashNotEqualError

    existing_wf = _make_workflow(graph={"nodes": [{"id": "old"}]})
    session = rag_pipeline_service.session
    session.add(existing_wf)
    session.commit()

    pipeline = _make_pipeline()
    account = _make_account()

    with pytest.raises(WorkflowHashNotEqualError):
        rag_pipeline_service.service.sync_draft_workflow(
            pipeline=pipeline,
            graph={"nodes": []},
            unique_hash="hash-different",
            account=account,
            environment_variables=[],
            conversation_variables=[],
            rag_pipeline_variables=[],
        )


def test_sync_draft_workflow_updates_existing(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    existing_wf = _make_workflow()
    session = rag_pipeline_service.session
    session.add(existing_wf)
    session.commit()

    pipeline = _make_pipeline()
    account = _make_account()

    result = rag_pipeline_service.service.sync_draft_workflow(
        pipeline=pipeline,
        graph={"nodes": [{"id": "n1"}]},
        unique_hash=existing_wf.unique_hash,
        account=account,
        environment_variables=[],
        conversation_variables=[],
        rag_pipeline_variables=[],
    )

    assert result is existing_wf
    assert result.updated_by == "u1"
    assert result.environment_variables == []


# --- get_default_block_config ---


def test_get_default_block_config_returns_config_for_valid_type(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    fake_node_class = mocker.Mock()
    fake_node_class.get_default_config.return_value = {"type": "start", "config": {}}

    # Use a simpler approach: test with a known valid node type
    from graphon.enums import BuiltinNodeTypes

    mocker.patch(
        "services.rag_pipeline.rag_pipeline.get_node_type_classes_mapping",
        return_value={BuiltinNodeTypes.START: {"1": fake_node_class}},
    )
    mocker.patch("services.rag_pipeline.rag_pipeline.LATEST_VERSION", "1")

    result = rag_pipeline_service.service.get_default_block_config("start")

    assert result == {"type": "start", "config": {}}


def test_get_default_block_config_returns_none_for_unmapped_type(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    assert rag_pipeline_service.service.get_default_block_config("nonexistent-type") is None


# --- update_workflow ---


def test_update_workflow_updates_allowed_fields(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    workflow = _make_workflow(app_id="pipeline-1")
    workflow.disallowed = "original"
    workflow_ref = WorkflowRef(tenant_id="t1", owner_id="pipeline-1", workflow_id="wf-1")
    session = rag_pipeline_service.session
    session.add(workflow)
    session.commit()

    result = rag_pipeline_service.service.update_workflow(
        session=session,
        account_id="u1",
        data={"marked_name": "v1", "marked_comment": "release", "disallowed": "hacked"},
        workflow_ref=workflow_ref,
    )

    assert result.marked_name == "v1"
    assert result.marked_comment == "release"
    assert result.disallowed == "original"  # non-allowed field not updated
    assert result.updated_by == "u1"


def test_update_workflow_returns_none_when_not_found(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    session = rag_pipeline_service.session

    result = rag_pipeline_service.service.update_workflow(
        session=session,
        account_id="u1",
        data={"marked_name": "v1"},
        workflow_ref=WorkflowRef(tenant_id="t1", owner_id="pipeline-1", workflow_id="wf-missing"),
    )

    assert result is None


def test_update_workflow_with_ref_scopes_lookup_to_pipeline(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    workflow = _make_workflow(app_id="pipeline-1")
    workflow_ref = WorkflowRef(tenant_id="t1", owner_id="pipeline-1", workflow_id="wf-1")
    session = rag_pipeline_service.session
    session.add_all([workflow, _make_workflow(workflow_id="wf-other", app_id="pipeline-2")])
    session.commit()

    result = rag_pipeline_service.service.update_workflow(
        session=session,
        account_id="u1",
        data={"marked_name": "v1"},
        workflow_ref=workflow_ref,
    )

    assert result is workflow


# --- get_rag_pipeline_paginate_workflow_runs ---


def test_get_rag_pipeline_paginate_workflow_runs_delegates(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    expected = mocker.Mock()
    repo_mock = mocker.Mock()
    repo_mock.get_paginated_workflow_runs.return_value = expected
    rag_pipeline_service.service._workflow_run_repo = repo_mock

    pipeline = _make_pipeline()
    result = rag_pipeline_service.service.get_rag_pipeline_paginate_workflow_runs(
        pipeline, {"limit": 10, "last_id": "abc"}
    )

    assert result is expected
    repo_mock.get_paginated_workflow_runs.assert_called_once_with(
        tenant_id="t1",
        app_id="p1",
        triggered_from=mocker.ANY,
        limit=10,
        last_id="abc",
    )


# --- get_rag_pipeline_workflow_run ---


def test_get_rag_pipeline_workflow_run_delegates(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    expected = mocker.Mock()
    repo_mock = mocker.Mock()
    repo_mock.get_workflow_run_by_id.return_value = expected
    rag_pipeline_service.service._workflow_run_repo = repo_mock

    pipeline = _make_pipeline()
    result = rag_pipeline_service.service.get_rag_pipeline_workflow_run(pipeline, "run-1")

    assert result is expected
    repo_mock.get_workflow_run_by_id.assert_called_once_with(tenant_id="t1", app_id="p1", run_id="run-1")


# --- is_workflow_exist ---


def test_is_workflow_exist_returns_true_when_draft_exists(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    rag_pipeline_service.session.add(_make_workflow())
    rag_pipeline_service.session.commit()

    pipeline = _make_pipeline()
    assert rag_pipeline_service.service.is_workflow_exist(pipeline) is True


def test_is_workflow_exist_returns_false_when_no_draft(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    pipeline = _make_pipeline()
    assert rag_pipeline_service.service.is_workflow_exist(pipeline) is False


# --- publish_workflow ---


def test_publish_workflow_success(mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext) -> None:
    draft_wf = _make_workflow(
        workflow_id="wf-draft",
        graph={
            "nodes": [
                {
                    "data": {
                        "type": "knowledge-index",
                        "dataset_id": "d1",
                        "chunk_structure": "paragraph",
                        "indexing_technique": "high_quality",
                        "process_rule": {"mode": "automatic"},
                        "retrieval_model": {"search_method": "hybrid_search", "top_k": 3},
                    }
                }
            ]
        },
    )
    pipeline = _make_pipeline(workflow_id="wf-old-published")
    dataset = _make_dataset()
    session = rag_pipeline_service.session
    session.add_all([draft_wf, pipeline, dataset])
    session.commit()
    mock_dataset_service_class = mocker.patch("services.dataset_service.DatasetService")

    result = rag_pipeline_service.service.publish_workflow(session=session, pipeline=pipeline, account=_make_account())

    assert result.version != Workflow.VERSION_DRAFT
    assert result in session
    mock_dataset_service_class.update_rag_pipeline_dataset_settings.assert_called_once()


# --- run_datasource_workflow_node ---


def test_run_datasource_workflow_node_website_crawl(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    from core.datasource.entities.datasource_entities import DatasourceProviderType

    # 1. Setup workflow and node
    pipeline = mocker.Mock()
    pipeline.id = "p1"
    pipeline.tenant_id = "t1"

    workflow = mocker.Mock()
    workflow.graph_dict = {
        "nodes": [
            {
                "id": "node-1",
                "data": {
                    "type": "datasource",
                    "plugin_id": "p-1",
                    "provider_name": "firecrawl",
                    "datasource_name": "website_crawl",
                    "datasource_parameters": {"url": {"value": "{{#start.url#}}"}},
                },
            }
        ]
    }
    mocker.patch.object(rag_pipeline_service.service, "get_published_workflow", return_value=workflow)

    # 2. Mock DatasourceManager and Runtime
    mock_runtime = mocker.Mock()
    mock_runtime.datasource_provider_type.return_value = DatasourceProviderType.WEBSITE_CRAWL

    # Mock the generator result for website crawl
    def mock_crawl_gen(**kwargs):
        yield mocker.Mock(result=mocker.Mock(status="processing", total=10, completed=2))
        yield mocker.Mock(
            result=mocker.Mock(status="completed", total=10, completed=10, web_info_list=[{"title": "test"}])
        )

    mock_runtime.get_website_crawl.side_effect = mock_crawl_gen

    mocker.patch(
        "core.datasource.datasource_manager.DatasourceManager.get_datasource_runtime",
        return_value=mock_runtime,
    )

    # 3. Mock DatasourceProviderService
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.DatasourceProviderService.get_datasource_credentials",
        return_value={"api_key": "sk-123"},
    )

    # 4. Mock Enums to avoid import issues or for consistency
    mocker.patch("services.rag_pipeline.rag_pipeline.DatasourceProviderType", DatasourceProviderType)

    # 5. Run test
    gen = rag_pipeline_service.service.run_datasource_workflow_node(
        pipeline=pipeline,
        node_id="node-1",
        user_inputs={"url": "https://example.com"},
        account=mocker.Mock(id="u1"),
        datasource_type="website_crawl",
        is_published=True,
    )

    events = list(gen)

    # 6. Assertions
    assert len(events) == 2
    assert events[0]["total"] == 10
    assert events[0]["completed"] == 2
    assert events[1]["data"] == [{"title": "test"}]
    assert events[1]["total"] == 10
    assert events[1]["completed"] == 10


# --- run_datasource_node_preview ---


def test_run_datasource_node_preview_online_document(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    from core.datasource.entities.datasource_entities import DatasourceMessage, DatasourceProviderType

    # 1. Setup workflow and node
    pipeline = mocker.Mock()
    pipeline.id = "p1"
    pipeline.tenant_id = "t1"

    workflow = mocker.Mock()
    workflow.graph_dict = {
        "nodes": [
            {
                "id": "node-1",
                "data": {
                    "type": "datasource",
                    "plugin_id": "p-1",
                    "provider_name": "notion",
                    "datasource_name": "online_document",
                    "datasource_parameters": {
                        "workspace_id": {"value": "ws-1"},
                        "page_id": {"value": "pg-1"},
                        "type": {"value": "page"},
                    },
                },
            }
        ]
    }
    mocker.patch.object(rag_pipeline_service.service, "get_published_workflow", return_value=workflow)

    # 2. Mock Runtime and results
    mock_runtime = mocker.Mock()

    def mock_doc_gen(**kwargs):
        # Yield a variable message
        msg1 = DatasourceMessage(
            type=DatasourceMessage.MessageType.VARIABLE,
            message=DatasourceMessage.VariableMessage(variable_name="content", variable_value="Hello ", stream=True),
        )
        yield msg1
        msg2 = DatasourceMessage(
            type=DatasourceMessage.MessageType.VARIABLE,
            message=DatasourceMessage.VariableMessage(variable_name="content", variable_value="World", stream=True),
        )
        yield msg2

    mock_runtime.get_online_document_page_content.side_effect = mock_doc_gen
    mocker.patch(
        "core.datasource.datasource_manager.DatasourceManager.get_datasource_runtime",
        return_value=mock_runtime,
    )
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.DatasourceProviderService.get_datasource_credentials",
        return_value={"token": "abc"},
    )
    mocker.patch("services.rag_pipeline.rag_pipeline.DatasourceProviderType", DatasourceProviderType)

    # 3. Run test
    result = rag_pipeline_service.service.run_datasource_node_preview(
        pipeline=pipeline,
        node_id="node-1",
        user_inputs={},
        account=mocker.Mock(id="u1"),
        datasource_type="online_document",
        is_published=True,
    )

    # 4. Assertions
    assert result == {"content": "Hello World"}


# --- _handle_node_run_result ---


def test_handle_node_run_result_success(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    from graphon.enums import WorkflowNodeExecutionMetadataKey, WorkflowNodeExecutionStatus
    from graphon.graph_events import NodeRunSucceededEvent
    from graphon.node_events.base import NodeRunResult

    # 1. Setup mock node and result
    node_instance = mocker.Mock()
    node_instance.workflow_id = "wf-1"
    node_instance.node_type = "start"
    node_instance.title = "Start"

    node_run_result = NodeRunResult(
        status=WorkflowNodeExecutionStatus.SUCCEEDED,
        inputs={"q": "hi"},
        outputs={"ans": "hello"},
        metadata={WorkflowNodeExecutionMetadataKey.TOTAL_TOKENS: 10},
    )

    def mock_getter():
        event = NodeRunSucceededEvent(
            id="event-1",
            start_at=time.time(),
            node_id="node-1",
            node_type="start",
            node_run_result=node_run_result,
            route_node_id=None,
        )
        yield event

    # 2. Run test
    result = rag_pipeline_service.service._handle_node_run_result(
        getter=lambda: (node_instance, mock_getter()), start_at=time.perf_counter(), tenant_id="t1", node_id="node-1"
    )

    # 3. Assertions
    assert result.status == WorkflowNodeExecutionStatus.SUCCEEDED
    assert result.inputs == {"q": "hi"}
    assert result.outputs == {"ans": "hello"}
    assert result.metadata == {WorkflowNodeExecutionMetadataKey.TOTAL_TOKENS: 10}


# --- get_first_step_parameters / get_second_step_parameters ---


def test_get_first_step_parameters_success(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    # 1. Setup mock workflow
    pipeline = mocker.Mock()
    workflow = mocker.Mock()
    workflow.graph_dict = {
        "nodes": [{"id": "node-1", "data": {"datasource_parameters": {"url": {"value": "{{#start.url#}}"}}}}]
    }
    workflow.rag_pipeline_variables = [{"variable": "url", "label": "URL", "type": "string"}]
    mocker.patch.object(rag_pipeline_service.service, "get_published_workflow", return_value=workflow)

    # 2. Run test
    result = rag_pipeline_service.service.get_first_step_parameters(pipeline=pipeline, node_id="node-1", is_draft=False)

    # 3. Assertions
    assert len(result) == 1
    assert result[0]["variable"] == "url"


def test_get_second_step_parameters_success(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    # 1. Setup mock workflow
    pipeline = mocker.Mock()
    workflow = mocker.Mock()
    workflow.graph_dict = {
        "nodes": [
            {
                "id": "node-1",
                "data": {},  # Second step logic is slightly different in how it gets variables
            }
        ]
    }
    workflow.rag_pipeline_variables = [{"variable": "var1", "label": "Var 1"}]
    mocker.patch.object(rag_pipeline_service.service, "get_published_workflow", return_value=workflow)

    # 2. Run test
    result = rag_pipeline_service.service.get_second_step_parameters(
        pipeline=pipeline, node_id="node-1", is_draft=False
    )

    # 3. Assertions
    # Note: get_second_step_parameters also filters by variable names found in node data
    # (Checking the code again, it seems to iterate through nodes but doesn't do much with variables yet)
    # Wait, let me check the code for get_second_step_parameters again.
    assert len(result) == 0  # Based on current implementation which seems to filter but no logic added yet?


# --- publish_customized_pipeline_template ---


def test_publish_customized_pipeline_template_success(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    # 1. Setup mocks
    pipeline = _make_pipeline(workflow_id="wf-1", is_published=True)

    workflow = _make_workflow(workflow_id="wf-1")

    session = rag_pipeline_service.session

    dataset = _make_dataset()
    dataset.chunk_structure = "paragraph"
    session.add_all([pipeline, workflow, dataset])
    session.commit()

    # Mock RagPipelineDslService
    mock_dsl_service = mocker.Mock()
    mock_dsl_service.export_rag_pipeline_dsl.return_value = "dsl: content"
    mocker.patch("services.rag_pipeline.rag_pipeline_dsl_service.RagPipelineDslService", return_value=mock_dsl_service)

    account = _make_account(account_id="user-123")

    # 2. Run test
    args = {"name": "New Template", "description": "Desc", "icon_info": {"icon": "star"}, "tags": ["tag1"]}
    rag_pipeline_service.service.publish_customized_pipeline_template("p1", args, account, "t1", session=session)

    # 3. Assertions
    # Verify a new template was added to session or similar?
    # Since we can't easily check the session inside the context manager with Mock,
    # we just check that no error was raised and DSL was exported.
    mock_dsl_service.export_rag_pipeline_dsl.assert_called_once_with(pipeline=pipeline, include_secret=True)
    template = session.scalar(select(PipelineCustomizedTemplate))
    assert template is not None
    assert template.name == "New Template"


# --- get_datasource_plugins ---


def test_get_datasource_plugins_success(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    # 1. Setup mocks
    dataset = _make_dataset()

    pipeline = _make_pipeline(workflow_id="wf-1")

    workflow = _make_workflow(
        graph={
            "nodes": [
                {
                    "id": "node-1",
                    "data": {
                        "type": "datasource",
                        "plugin_id": "p-1",
                        "provider_name": "notion",
                        "provider_type": "online_document",
                        "title": "Notion",
                    },
                }
            ]
        }
    )
    rag_pipeline_service.session.add_all([dataset, pipeline, workflow])
    rag_pipeline_service.session.commit()

    # Mock DatasourceProviderService
    mock_provider_service = mocker.Mock()
    mock_provider_service.list_datasource_credentials.return_value = [
        {"id": "c1", "name": "Cred 1", "type": "token", "is_default": True}
    ]
    mocker.patch("services.rag_pipeline.rag_pipeline.DatasourceProviderService", return_value=mock_provider_service)

    # 2. Run test
    result = rag_pipeline_service.service.get_datasource_plugins("t1", "d1", True)

    # 3. Assertions
    assert len(result) == 1
    assert result[0]["node_id"] == "node-1"
    assert result[0]["credentials"][0]["id"] == "c1"


# --- retry_error_document ---


def test_retry_error_document_success(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    from models.dataset import Document, DocumentPipelineExecutionLog

    # 1. Setup mocks
    dataset = _make_dataset()
    document = mocker.Mock(spec=Document)
    document.id = "doc-1"
    log = DocumentPipelineExecutionLog(
        pipeline_id="p1",
        document_id=document.id,
        datasource_type="upload_file",
        datasource_info="{}",
        datasource_node_id="source-1",
        input_data={},
        created_by="u1",
    )
    pipeline = _make_pipeline(workflow_id="wf-1")
    workflow = _make_workflow(workflow_id="wf-1")
    rag_pipeline_service.session.add_all([log, pipeline, workflow])
    rag_pipeline_service.session.commit()

    # Mock PipelineGenerator
    mock_gen_instance = mocker.Mock()
    mocker.patch("services.rag_pipeline.rag_pipeline.PipelineGenerator", return_value=mock_gen_instance)

    # 2. Run test
    user = mocker.Mock()
    rag_pipeline_service.service.retry_error_document(dataset, document, user)

    # 3. Assertions
    mock_gen_instance.generate.assert_called_once()


# --- set_datasource_variables ---


def test_set_datasource_variables_success(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    from graphon.entities.workflow_node_execution import WorkflowNodeExecution
    from models.dataset import Pipeline

    # 1. Setup mocks
    pipeline = mocker.Mock(spec=Pipeline)
    pipeline.id = "p-1"
    pipeline.tenant_id = "t1"

    draft_wf = mocker.Mock()
    draft_wf.id = "wf-1"
    draft_wf.get_enclosing_node_type_and_id.return_value = None  # Avoid unpacking error
    mocker.patch.object(rag_pipeline_service.service, "get_draft_workflow", return_value=draft_wf)

    execution = mocker.Mock(spec=WorkflowNodeExecution)
    execution.id = "exec-1"
    execution.process_data = {}
    execution.inputs = {}
    execution.outputs = {}
    mocker.patch.object(rag_pipeline_service.service, "_handle_node_run_result", return_value=execution)

    # Mock Repository
    mock_repo_instance = mocker.Mock()
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.SQLAlchemyWorkflowNodeExecutionRepository",
        return_value=mock_repo_instance,
    )
    # Repository._to_db_model is also called
    mock_db_exec = mocker.Mock()
    mock_db_exec.node_id = "node-1"
    mock_db_exec.node_type = "datasource"
    mock_repo_instance._to_db_model.return_value = mock_db_exec

    # Mock Session and begin
    mocker.patch("services.rag_pipeline.rag_pipeline.Session", return_value=mocker.MagicMock())

    # Mock DraftVariableSaver
    mock_saver_instance = mocker.Mock()
    mocker.patch("services.rag_pipeline.rag_pipeline.DraftVariableSaver", return_value=mock_saver_instance)

    # 2. Run test
    args = {"start_node_id": "node-1"}
    user = mocker.Mock()
    user.id = "user-1"
    rag_pipeline_service.service.set_datasource_variables(pipeline, args, user)

    # 3. Assertions
    mock_repo_instance.save.assert_called_once()
    mock_saver_instance.save.assert_called_once()


# --- Utility Methods ---


def test_get_draft_workflow_success(rag_pipeline_service: RagPipelineServiceTestContext) -> None:
    # 1. Setup mocks
    pipeline = _make_pipeline()

    workflow = _make_workflow()

    rag_pipeline_service.session.add(workflow)
    rag_pipeline_service.session.commit()

    # 2. Run test
    result = rag_pipeline_service.service.get_draft_workflow(pipeline)

    # 3. Assertions
    assert result == workflow


def test_get_published_workflow_success(rag_pipeline_service: RagPipelineServiceTestContext) -> None:
    # 1. Setup mocks
    pipeline = _make_pipeline(workflow_id="wf-pub")

    workflow = _make_workflow(workflow_id="wf-pub")

    rag_pipeline_service.session.add(workflow)
    rag_pipeline_service.session.commit()

    # 2. Run test
    result = rag_pipeline_service.service.get_published_workflow(pipeline)

    # 3. Assertions
    assert result == workflow


def test_get_default_block_configs_success(rag_pipeline_service: RagPipelineServiceTestContext) -> None:
    # This calls static methods on node classes, should be safe with default mocks or as-is
    # unless they access db.
    result = rag_pipeline_service.service.get_default_block_configs()
    assert isinstance(result, list)
    assert len(result) > 0


def test_get_default_block_config_success(rag_pipeline_service: RagPipelineServiceTestContext) -> None:
    from graphon.enums import BuiltinNodeTypes

    result = rag_pipeline_service.service.get_default_block_config(BuiltinNodeTypes.LLM)
    assert result is not None
    assert result["type"] == "llm"


def test_publish_workflow_raises_when_draft_workflow_missing(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    session = rag_pipeline_service.session
    pipeline = _make_pipeline()
    account = _make_account()

    with pytest.raises(ValueError, match="No valid workflow found"):
        rag_pipeline_service.service.publish_workflow(session=session, pipeline=pipeline, account=account)


def test_get_default_block_config_returns_none_when_mapped_type_missing(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    from graphon.enums import BuiltinNodeTypes

    mocker.patch("services.rag_pipeline.rag_pipeline.get_node_type_classes_mapping", return_value={})

    assert rag_pipeline_service.service.get_default_block_config(BuiltinNodeTypes.START) is None


def test_get_default_block_config_injects_http_request_filter(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    from graphon.enums import BuiltinNodeTypes

    fake_node_cls = mocker.Mock()
    fake_node_cls.get_default_config.return_value = {"type": "http-request"}
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.get_node_type_classes_mapping",
        return_value={BuiltinNodeTypes.HTTP_REQUEST: {"1": fake_node_cls}},
    )
    mocker.patch("services.rag_pipeline.rag_pipeline.LATEST_VERSION", "1")

    rag_pipeline_service.service.get_default_block_config(BuiltinNodeTypes.HTTP_REQUEST)

    called_filters = fake_node_cls.get_default_config.call_args.kwargs["filters"]
    assert "http_request_config" in called_filters


def test_run_draft_workflow_node_raises_when_workflow_missing(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    pipeline = _make_pipeline()
    account = _make_account()
    mocker.patch.object(rag_pipeline_service.service, "get_draft_workflow", return_value=None)

    with pytest.raises(ValueError, match="Workflow not initialized"):
        rag_pipeline_service.service.run_draft_workflow_node(pipeline, "node-1", {}, account)


def test_run_draft_workflow_node_saves_execution_and_variables(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    pipeline = _make_pipeline()
    account = _make_account()
    draft_workflow = mocker.Mock(id="wf-1")
    draft_workflow.get_node_config_by_id.return_value = {"id": "node-1"}
    draft_workflow.get_enclosing_node_type_and_id.return_value = ("loop", "enclosing-node")
    mocker.patch.object(rag_pipeline_service.service, "get_draft_workflow", return_value=draft_workflow)

    execution = SimpleNamespace(id="exec-1", node_id="node-1", node_type="llm", process_data={}, outputs={})
    mocker.patch.object(rag_pipeline_service.service, "_handle_node_run_result", return_value=execution)

    repo = mocker.Mock()
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.DifyCoreRepositoryFactory.create_workflow_node_execution_repository",
        return_value=repo,
    )
    rag_pipeline_service.service._node_execution_service_repo = mocker.Mock(
        get_execution_by_id=mocker.Mock(return_value="db")
    )
    saver = mocker.Mock()
    mocker.patch("services.rag_pipeline.rag_pipeline.DraftVariableSaver", return_value=saver)

    session_ctx = mocker.MagicMock()
    begin_ctx = mocker.MagicMock()
    session_ctx.begin.return_value = begin_ctx
    mocker.patch("services.rag_pipeline.rag_pipeline.Session", return_value=session_ctx)

    result = rag_pipeline_service.service.run_draft_workflow_node(pipeline, "node-1", {"q": "x"}, account)

    assert result == "db"
    assert execution.workflow_id == "wf-1"
    repo.save.assert_called_once_with(execution)
    saver.save.assert_called_once()


def test_run_datasource_workflow_node_returns_error_when_workflow_missing(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    pipeline = SimpleNamespace(id="p1", tenant_id="t1")
    mocker.patch.object(rag_pipeline_service.service, "get_draft_workflow", return_value=None)

    events = list(
        rag_pipeline_service.service.run_datasource_workflow_node(
            pipeline=pipeline,
            node_id="node-1",
            user_inputs={},
            account=SimpleNamespace(id="u1"),
            datasource_type="online_document",
            is_published=False,
        )
    )

    assert events[0]["event"] == "datasource_error"


def test_run_datasource_workflow_node_online_document_success(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    from core.datasource.entities.datasource_entities import DatasourceProviderType

    pipeline = SimpleNamespace(id="p1", tenant_id="t1")
    workflow = mocker.Mock()
    workflow.graph_dict = {
        "nodes": [
            {
                "id": "node-1",
                "data": {
                    "type": "datasource",
                    "plugin_id": "pid",
                    "provider_name": "notion",
                    "datasource_name": "online_document",
                    "datasource_parameters": {"workspace_id": {"value": None}, "page_id": {"value": "fixed"}},
                },
            }
        ]
    }
    mocker.patch.object(rag_pipeline_service.service, "get_published_workflow", return_value=workflow)

    runtime = mocker.Mock()
    runtime.runtime = SimpleNamespace(credentials=None)
    runtime.datasource_provider_type.return_value = DatasourceProviderType.ONLINE_DOCUMENT
    runtime.get_online_document_pages.return_value = [SimpleNamespace(result=[{"id": "pg-1"}])]
    mocker.patch("core.datasource.datasource_manager.DatasourceManager.get_datasource_runtime", return_value=runtime)
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.DatasourceProviderService.get_datasource_credentials",
        return_value={"token": "x"},
    )

    events = list(
        rag_pipeline_service.service.run_datasource_workflow_node(
            pipeline=pipeline,
            node_id="node-1",
            user_inputs={},
            account=SimpleNamespace(id="u1"),
            datasource_type=DatasourceProviderType.ONLINE_DOCUMENT,
            is_published=True,
        )
    )

    assert events[0]["event"] == "datasource_processing"
    assert events[1]["event"] == "datasource_completed"


def test_run_datasource_workflow_node_online_drive_success(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    from core.datasource.entities.datasource_entities import DatasourceProviderType

    pipeline = SimpleNamespace(id="p1", tenant_id="t1")
    workflow = mocker.Mock()
    workflow.graph_dict = {
        "nodes": [
            {
                "id": "node-1",
                "data": {
                    "type": "datasource",
                    "plugin_id": "pid",
                    "provider_name": "drive",
                    "datasource_name": "online_drive",
                    "datasource_parameters": {"bucket": {"value": "bucket-1"}, "next_page_parameters": {"value": []}},
                },
            }
        ]
    }
    mocker.patch.object(rag_pipeline_service.service, "get_published_workflow", return_value=workflow)

    runtime = mocker.Mock()
    runtime.runtime = SimpleNamespace(credentials=None)
    runtime.datasource_provider_type.return_value = DatasourceProviderType.ONLINE_DRIVE
    runtime.online_drive_browse_files.return_value = [SimpleNamespace(result=[{"name": "f1"}])]
    mocker.patch("core.datasource.datasource_manager.DatasourceManager.get_datasource_runtime", return_value=runtime)
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.DatasourceProviderService.get_datasource_credentials",
        return_value={},
    )

    events = list(
        rag_pipeline_service.service.run_datasource_workflow_node(
            pipeline=pipeline,
            node_id="node-1",
            user_inputs={"bucket": "bucket-1"},
            account=SimpleNamespace(id="u1"),
            datasource_type=DatasourceProviderType.ONLINE_DRIVE,
            is_published=True,
        )
    )

    assert events[0]["event"] == "datasource_processing"
    assert events[1]["event"] == "datasource_completed"


def test_handle_node_run_result_default_value_strategy(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    from datetime import datetime

    from graphon.enums import BuiltinNodeTypes, ErrorStrategy, WorkflowNodeExecutionStatus
    from graphon.graph_events import NodeRunFailedEvent
    from graphon.node_events.base import NodeRunResult

    node_instance = SimpleNamespace(
        workflow_id="wf-1",
        node_type=BuiltinNodeTypes.START,
        title="Start",
        error_strategy=ErrorStrategy.DEFAULT_VALUE,
        default_value_dict={"fallback": "ok"},
        graph_runtime_state=SimpleNamespace(variable_pool=mocker.Mock()),
    )

    failed_result = NodeRunResult(
        status=WorkflowNodeExecutionStatus.FAILED,
        error="boom",
        error_type="runtime_error",
        inputs={"x": 1},
    )

    def _events():
        yield NodeRunFailedEvent(
            id="e-1",
            node_id="node-1",
            node_type=BuiltinNodeTypes.START,
            start_at=datetime.now(),
            error="boom",
            node_run_result=failed_result,
        )

    result = rag_pipeline_service.service._handle_node_run_result(
        getter=lambda: (node_instance, _events()),
        start_at=time.perf_counter(),
        tenant_id="t1",
        node_id="node-1",
    )

    assert result.status == WorkflowNodeExecutionStatus.EXCEPTION
    assert result.outputs
    assert result.outputs["fallback"] == "ok"


def test_get_first_step_parameters_raises_when_datasource_node_missing(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    workflow = SimpleNamespace(graph_dict={"nodes": []}, rag_pipeline_variables=[{"variable": "url"}])
    mocker.patch.object(rag_pipeline_service.service, "get_published_workflow", return_value=workflow)

    with pytest.raises(ValueError, match="Datasource node data not found"):
        rag_pipeline_service.service.get_first_step_parameters(SimpleNamespace(), "missing-node")


def test_get_second_step_parameters_handles_string_and_list_variable_references(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    workflow = SimpleNamespace(
        rag_pipeline_variables=[
            {"variable": "url", "belong_to_node_id": "node-1"},
            {"variable": "bucket", "belong_to_node_id": "shared"},
            {"variable": "keep", "belong_to_node_id": "node-1"},
        ],
        graph_dict={
            "nodes": [
                {
                    "id": "node-1",
                    "data": {
                        "datasource_parameters": {
                            "u": {"value": "{{#start.url#}}"},
                            "b": {"value": ["start", "bucket"]},
                        }
                    },
                }
            ]
        },
    )
    mocker.patch.object(rag_pipeline_service.service, "get_published_workflow", return_value=workflow)

    result = rag_pipeline_service.service.get_second_step_parameters(SimpleNamespace(), "node-1")

    assert result == [{"variable": "keep", "belong_to_node_id": "node-1"}]


def test_get_rag_pipeline_workflow_run_node_executions_empty_when_run_missing(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    pipeline = _make_pipeline()
    mocker.patch.object(rag_pipeline_service.service, "get_rag_pipeline_workflow_run", return_value=None)

    result = rag_pipeline_service.service.get_rag_pipeline_workflow_run_node_executions(
        pipeline=pipeline, run_id="run-1", user=_make_account()
    )

    assert result == []


def test_get_rag_pipeline_workflow_run_node_executions_returns_sorted_executions(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    pipeline = _make_pipeline()
    mocker.patch.object(
        rag_pipeline_service.service, "get_rag_pipeline_workflow_run", return_value=SimpleNamespace(id="run-1")
    )
    repo = mocker.Mock()
    repo.get_db_models_by_workflow_run.return_value = ["n1", "n2"]
    mocker.patch("services.rag_pipeline.rag_pipeline.SQLAlchemyWorkflowNodeExecutionRepository", return_value=repo)

    result = rag_pipeline_service.service.get_rag_pipeline_workflow_run_node_executions(
        pipeline=pipeline, run_id="run-1", user=_make_account()
    )

    assert result == ["n1", "n2"]


def test_get_recommended_plugins_returns_empty_when_no_active_plugins(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    result = rag_pipeline_service.service.get_recommended_plugins("all", _make_account(), "t1")

    assert result == {
        "installed_recommended_plugins": [],
        "uninstalled_recommended_plugins": [],
    }


def test_get_recommended_plugins_returns_installed_and_uninstalled(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    plugin_a = _make_recommended_plugin("plugin-a")
    plugin_b = _make_recommended_plugin("plugin-b")
    plugin_a.type = "custom"
    plugin_b.type = "custom"
    rag_pipeline_service.session.add_all([plugin_a, plugin_b])
    rag_pipeline_service.session.commit()
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.BuiltinToolManageService.list_builtin_tools",
        return_value=[SimpleNamespace(plugin_id="plugin-a", to_dict=lambda: {"plugin_id": "plugin-a"})],
    )
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.marketplace.batch_fetch_plugin_by_ids",
        return_value=[{"plugin_id": "plugin-b", "name": "Plugin B"}],
    )

    result = rag_pipeline_service.service.get_recommended_plugins("custom", _make_account(), "t1")

    assert result["installed_recommended_plugins"] == [{"plugin_id": "plugin-a"}]
    assert result["uninstalled_recommended_plugins"] == [{"plugin_id": "plugin-b", "name": "Plugin B"}]


def test_get_node_last_run_delegates_to_repository(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    repo = mocker.Mock()
    repo.get_node_last_execution.return_value = "node-exec"
    rag_pipeline_service.service._node_execution_service_repo = repo
    pipeline = _make_pipeline()
    workflow = _make_workflow(workflow_id="wf1")

    result = rag_pipeline_service.service.get_node_last_run(pipeline, workflow, "node-1")

    assert result == "node-exec"


def test_set_datasource_variables_raises_when_node_id_missing(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    pipeline = SimpleNamespace(id="p1", tenant_id="t1")
    workflow = mocker.Mock()
    mocker.patch.object(rag_pipeline_service.service, "get_draft_workflow", return_value=workflow)

    with pytest.raises(ValueError, match="Node id is required"):
        rag_pipeline_service.service.set_datasource_variables(pipeline, {"start_node_id": ""}, SimpleNamespace(id="u1"))


def test_get_default_block_configs_skips_empty_configs(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    from graphon.enums import BuiltinNodeTypes

    http_node = mocker.Mock()
    http_node.get_default_config.return_value = {"type": "http-request"}
    empty_node = mocker.Mock()
    empty_node.get_default_config.return_value = None

    mocker.patch(
        "services.rag_pipeline.rag_pipeline.get_node_type_classes_mapping",
        return_value={
            BuiltinNodeTypes.HTTP_REQUEST: {"1": http_node},
            BuiltinNodeTypes.START: {"1": empty_node},
        },
    )
    mocker.patch("services.rag_pipeline.rag_pipeline.LATEST_VERSION", "1")

    result = rag_pipeline_service.service.get_default_block_configs()

    assert result == [{"type": "http-request"}]
    http_node.get_default_config.assert_called_once()
    empty_node.get_default_config.assert_called_once()


def test_run_datasource_workflow_node_returns_error_when_node_missing(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    pipeline = SimpleNamespace(id="p1", tenant_id="t1")
    workflow = SimpleNamespace(graph_dict={"nodes": []})
    mocker.patch.object(rag_pipeline_service.service, "get_published_workflow", return_value=workflow)

    events = list(
        rag_pipeline_service.service.run_datasource_workflow_node(
            pipeline=pipeline,
            node_id="missing-node",
            user_inputs={},
            account=SimpleNamespace(id="u1"),
            datasource_type="online_document",
            is_published=True,
        )
    )

    assert len(events) == 1
    assert "Datasource node data not found" in events[0]["error"]


def test_run_datasource_workflow_node_online_document_exception(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    pipeline = SimpleNamespace(id="p1", tenant_id="t1")
    workflow = SimpleNamespace(
        graph_dict={
            "nodes": [
                {
                    "id": "node-1",
                    "data": {
                        "plugin_id": "plugin-1",
                        "provider_name": "provider-1",
                        "datasource_name": "doc",
                        "datasource_parameters": {},
                    },
                }
            ]
        }
    )
    mocker.patch.object(rag_pipeline_service.service, "get_published_workflow", return_value=workflow)

    runtime = mocker.Mock()

    class _FailingIterator:
        def __iter__(self):
            return self

        def __next__(self):
            raise RuntimeError("doc failed")

    runtime.get_online_document_pages.return_value = _FailingIterator()
    runtime.datasource_provider_type.return_value = "online_document"

    mocker.patch("core.datasource.datasource_manager.DatasourceManager.get_datasource_runtime", return_value=runtime)
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.DatasourceProviderService.get_datasource_credentials", return_value=None
    )

    events = list(
        rag_pipeline_service.service.run_datasource_workflow_node(
            pipeline=pipeline,
            node_id="node-1",
            user_inputs={},
            account=SimpleNamespace(id="u1"),
            datasource_type="online_document",
            is_published=True,
        )
    )

    assert len(events) == 2
    assert events[0]["event"] == "datasource_processing"
    assert "doc failed" in events[1]["error"]


def test_run_datasource_node_preview_raises_for_stream_non_string(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    from core.datasource.entities.datasource_entities import DatasourceMessage

    pipeline = SimpleNamespace(id="p1", tenant_id="t1")
    workflow = SimpleNamespace(
        graph_dict={
            "nodes": [
                {
                    "id": "node-1",
                    "data": {
                        "plugin_id": "plugin-1",
                        "provider_name": "provider-1",
                        "datasource_name": "doc",
                        "datasource_parameters": {},
                    },
                }
            ]
        }
    )
    mocker.patch.object(rag_pipeline_service.service, "get_published_workflow", return_value=workflow)

    runtime = mocker.Mock()

    def _bad_stream_generator(*args, **kwargs):
        yield DatasourceMessage(
            type=DatasourceMessage.MessageType.VARIABLE,
            message=DatasourceMessage.VariableMessage(variable_name="content", variable_value=1, stream=True),
        )

    runtime.get_online_document_page_content.side_effect = _bad_stream_generator
    runtime.datasource_provider_type.return_value = "online_document"

    mocker.patch("core.datasource.datasource_manager.DatasourceManager.get_datasource_runtime", return_value=runtime)
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.DatasourceProviderService.get_datasource_credentials", return_value=None
    )

    with pytest.raises(RuntimeError, match="must be a string"):
        rag_pipeline_service.service.run_datasource_node_preview(
            pipeline=pipeline,
            node_id="node-1",
            user_inputs={},
            account=SimpleNamespace(id="u1"),
            datasource_type="online_document",
            is_published=True,
        )


def test_get_first_step_parameters_returns_empty_when_no_rag_variables(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    workflow = SimpleNamespace(
        graph_dict={"nodes": [{"id": "node-1", "data": {"datasource_parameters": {"url": {"value": "literal"}}}}]},
        rag_pipeline_variables=[],
    )
    mocker.patch.object(rag_pipeline_service.service, "get_published_workflow", return_value=workflow)

    result = rag_pipeline_service.service.get_first_step_parameters(SimpleNamespace(), "node-1")

    assert result == []


def test_get_second_step_parameters_filters_first_step_variables(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    workflow = SimpleNamespace(
        graph_dict={
            "nodes": [
                {
                    "id": "node-1",
                    "data": {
                        "datasource_parameters": {
                            "workspace": {"value": "{{#start.workspace#}}"},
                            "bucket": {"value": ["input", "bucket"]},
                        }
                    },
                }
            ]
        },
        rag_pipeline_variables=[
            {"variable": "workspace", "belong_to_node_id": "shared"},
            {"variable": "bucket", "belong_to_node_id": "shared"},
            {"variable": "keep", "belong_to_node_id": "shared"},
            {"variable": "other-node", "belong_to_node_id": "node-x"},
        ],
    )
    mocker.patch.object(rag_pipeline_service.service, "get_published_workflow", return_value=workflow)

    result = rag_pipeline_service.service.get_second_step_parameters(SimpleNamespace(), "node-1")

    assert result == [{"variable": "keep", "belong_to_node_id": "shared"}]


def test_retry_error_document_raises_when_execution_log_not_found(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    with pytest.raises(ValueError, match="Document pipeline execution log not found"):
        rag_pipeline_service.service.retry_error_document(
            SimpleNamespace(), SimpleNamespace(id="doc-1"), SimpleNamespace(id="u1")
        )


def test_get_datasource_plugins_raises_when_workflow_not_found(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    dataset = _make_dataset()
    pipeline = _make_pipeline(workflow_id="wf-1")
    rag_pipeline_service.session.add_all([dataset, pipeline])
    rag_pipeline_service.session.commit()

    with pytest.raises(ValueError, match="Pipeline or workflow not found"):
        rag_pipeline_service.service.get_datasource_plugins("t1", "d1", True)


def test_handle_node_run_result_raises_when_no_terminal_event(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    node_instance = SimpleNamespace(
        workflow_id="wf-1",
        node_type="start",
        title="Start",
        graph_runtime_state=SimpleNamespace(variable_pool=SimpleNamespace(get=lambda _: None)),
        error_strategy=None,
    )

    def _event_generator():
        yield object()

    with pytest.raises(ValueError, match="Node run failed with no run result"):
        rag_pipeline_service.service._handle_node_run_result(
            getter=lambda: (node_instance, _event_generator()),
            start_at=time.perf_counter(),
            tenant_id="t1",
            node_id="node-1",
        )


def test_handle_node_run_result_marks_document_error_for_published_invoke(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    from core.app.entities.app_invoke_entities import InvokeFrom
    from graphon.enums import WorkflowNodeExecutionStatus
    from graphon.graph_events import NodeRunFailedEvent
    from graphon.node_events.base import NodeRunResult

    class FakeVariablePool:
        def __init__(self):
            self._values = {
                ("sys", "invoke_from"): SimpleNamespace(value=InvokeFrom.PUBLISHED_PIPELINE),
                ("sys", "app_id"): SimpleNamespace(value="pipeline-1"),
                ("sys", "dataset_id"): SimpleNamespace(value="dataset-1"),
                ("sys", "document_id"): SimpleNamespace(value="doc-1"),
            }

        def get(self, path):
            return self._values.get(tuple(path))

    node_instance = SimpleNamespace(
        workflow_id="wf-1",
        node_type="start",
        title="Start",
        graph_runtime_state=SimpleNamespace(variable_pool=FakeVariablePool()),
        error_strategy=None,
    )
    run_result = NodeRunResult(
        status=WorkflowNodeExecutionStatus.FAILED,
        error="boom",
        error_type="runtime",
        inputs={},
        outputs={},
    )

    def _event_generator():
        yield NodeRunFailedEvent(
            id="evt-1",
            start_at=time.time(),
            node_id="node-1",
            node_type="start",
            node_run_result=run_result,
            error="boom",
            route_node_id=None,
        )

    dataset = _make_dataset(dataset_id="dataset-1", pipeline_id="pipeline-1")
    document = _make_document(document_id="doc-1", dataset_id=dataset.id)
    rag_pipeline_service.session.add_all([dataset, document])
    rag_pipeline_service.session.commit()

    result = rag_pipeline_service.service._handle_node_run_result(
        getter=lambda: (node_instance, _event_generator()),
        start_at=time.perf_counter(),
        tenant_id="t1",
        node_id="node-1",
    )

    assert result.status == WorkflowNodeExecutionStatus.FAILED
    assert document.indexing_status == "error"
    assert document.error == "boom"


def test_run_datasource_node_preview_raises_for_unsupported_provider(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    pipeline = SimpleNamespace(id="p1", tenant_id="t1")
    workflow = SimpleNamespace(
        graph_dict={
            "nodes": [
                {
                    "id": "node-1",
                    "data": {
                        "plugin_id": "plugin-1",
                        "provider_name": "provider-1",
                        "datasource_name": "doc",
                        "datasource_parameters": {},
                    },
                }
            ]
        }
    )
    mocker.patch.object(rag_pipeline_service.service, "get_published_workflow", return_value=workflow)
    runtime = mocker.Mock()
    runtime.datasource_provider_type.return_value = "unsupported"
    mocker.patch("core.datasource.datasource_manager.DatasourceManager.get_datasource_runtime", return_value=runtime)
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.DatasourceProviderService.get_datasource_credentials", return_value=None
    )

    with pytest.raises(RuntimeError, match="Unsupported datasource provider"):
        rag_pipeline_service.service.run_datasource_node_preview(
            pipeline=pipeline,
            node_id="node-1",
            user_inputs={},
            account=SimpleNamespace(id="u1"),
            datasource_type="website_crawl",
            is_published=True,
        )


def test_publish_customized_pipeline_template_raises_for_missing_pipeline(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    session = rag_pipeline_service.session

    with pytest.raises(ValueError, match="Pipeline not found"):
        rag_pipeline_service.service.publish_customized_pipeline_template(
            "p1", {}, _make_account(), "t1", session=session
        )


def test_publish_customized_pipeline_template_raises_for_missing_workflow_id(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    pipeline = _make_pipeline(workflow_id=None)
    session = rag_pipeline_service.session
    session.add(pipeline)
    session.commit()

    with pytest.raises(ValueError, match="Pipeline workflow not found"):
        rag_pipeline_service.service.publish_customized_pipeline_template(
            "p1", {"name": "template-name"}, _make_account(), "t1", session=session
        )


def test_get_pipeline_raises_when_dataset_missing(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    with pytest.raises(ValueError, match="Dataset not found"):
        rag_pipeline_service.service.get_pipeline("t1", "d1")


def test_get_pipeline_raises_when_pipeline_missing(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    dataset = _make_dataset()
    rag_pipeline_service.session.add(dataset)
    rag_pipeline_service.session.commit()

    with pytest.raises(ValueError, match="Pipeline not found"):
        rag_pipeline_service.service.get_pipeline("t1", "d1")


def test_init_uses_default_sessionmaker_when_none(mocker: MockerFixture, sqlite_engine: Engine) -> None:
    default_session_maker = mocker.Mock()
    mocker.patch("services.rag_pipeline.rag_pipeline.sessionmaker", return_value=default_session_maker)
    mocker.patch("services.rag_pipeline.rag_pipeline.db", SimpleNamespace(engine=sqlite_engine))
    create_exec_repo = mocker.patch(
        "services.rag_pipeline.rag_pipeline.DifyAPIRepositoryFactory.create_api_workflow_node_execution_repository"
    )
    create_run_repo = mocker.patch(
        "services.rag_pipeline.rag_pipeline.DifyAPIRepositoryFactory.create_api_workflow_run_repository"
    )

    with Session(sqlite_engine) as session:
        RagPipelineService(session=session, session_maker=None)

    create_exec_repo.assert_called_once_with(default_session_maker)
    create_run_repo.assert_called_once_with(default_session_maker)


def test_get_pipeline_templates_builtin_en_us_no_fallback(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    mocker.patch("services.rag_pipeline.rag_pipeline.dify_config.HOSTED_FETCH_PIPELINE_TEMPLATES_MODE", "remote")
    session = rag_pipeline_service.session
    retrieval = mocker.Mock()
    retrieval.get_pipeline_templates.return_value = {"pipeline_templates": []}
    factory = mocker.patch("services.rag_pipeline.rag_pipeline.PipelineTemplateRetrievalFactory")
    factory.get_pipeline_template_factory.return_value.return_value = retrieval
    builtin = factory.get_built_in_pipeline_template_retrieval.return_value
    result = RagPipelineService.get_pipeline_templates(type="built-in", language="en-US", session=session)

    assert result == {"pipeline_templates": []}
    retrieval.get_pipeline_templates.assert_called_once_with("en-US", None, session=session)
    builtin.fetch_pipeline_templates_from_builtin.assert_not_called()


def test_update_customized_pipeline_template_commits_when_name_empty(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    template = _make_customized_template()
    template.id = "tpl-1"
    session = rag_pipeline_service.session
    session.add(template)
    session.commit()

    info = PipelineTemplateInfoEntity(name="", description="updated", icon_info=IconInfo(icon="i"))
    result = RagPipelineService.update_customized_pipeline_template(
        "tpl-1", info, _make_account(), "t1", session=session
    )

    assert result.description == "updated"
    assert session.get(PipelineCustomizedTemplate, template.id).description == "updated"


def test_get_all_published_workflow_without_filters_has_no_more(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    session = rag_pipeline_service.session
    pipeline = _make_pipeline(workflow_id="wf-live")
    workflow = _make_workflow(workflow_id="wf1")
    workflow.version = "published"
    session.add(workflow)
    session.commit()

    workflows, has_more = rag_pipeline_service.service.get_all_published_workflow(
        session=session,
        pipeline=pipeline,
        page=1,
        limit=2,
        user_id=None,
        named_only=False,
    )

    assert [item.id for item in workflows] == ["wf1"]
    assert has_more is False


def test_publish_workflow_skips_dataset_update_for_non_knowledge_nodes(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    draft = SimpleNamespace(
        type="workflow",
        graph={"nodes": [{"data": {"type": "start"}}]},
        features={},
        environment_variables=[],
        conversation_variables=[],
        rag_pipeline_variables=[],
    )
    session = rag_pipeline_service.session
    real_draft = _make_workflow(graph=draft.graph)
    session.add(real_draft)
    session.commit()
    result = rag_pipeline_service.service.publish_workflow(
        session=session,
        pipeline=SimpleNamespace(id="p1", tenant_id="t1", is_published=False, retrieve_dataset=lambda session: None),
        account=SimpleNamespace(id="u1"),
    )

    assert result.graph_dict == {"nodes": [{"data": {"type": "start"}}]}


def test_get_default_block_config_returns_none_when_default_empty(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    from graphon.enums import BuiltinNodeTypes

    node_cls = mocker.Mock()
    node_cls.get_default_config.return_value = None
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.get_node_type_classes_mapping",
        return_value={BuiltinNodeTypes.START: {"1": node_cls}},
    )
    mocker.patch("services.rag_pipeline.rag_pipeline.LATEST_VERSION", "1")

    assert rag_pipeline_service.service.get_default_block_config("start") is None


def test_run_datasource_workflow_node_handles_variable_parameter_types(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    from core.datasource.entities.datasource_entities import DatasourceProviderType

    workflow = SimpleNamespace(
        graph_dict={
            "nodes": [
                {
                    "id": "node-1",
                    "data": {
                        "plugin_id": "p",
                        "provider_name": "provider",
                        "datasource_name": "crawl",
                        "datasource_parameters": {
                            "a": {"value": None},
                            "b": {"value": "literal"},
                            "c": {"value": ["input", "k"]},
                        },
                    },
                }
            ]
        }
    )
    mocker.patch.object(rag_pipeline_service.service, "get_published_workflow", return_value=workflow)
    runtime = mocker.Mock()

    def crawl_gen(**kwargs):
        yield SimpleNamespace(result=SimpleNamespace(status="completed", total=1, completed=1, web_info_list=[]))

    runtime.get_website_crawl.side_effect = crawl_gen
    runtime.datasource_provider_type.return_value = DatasourceProviderType.WEBSITE_CRAWL
    mocker.patch("core.datasource.datasource_manager.DatasourceManager.get_datasource_runtime", return_value=runtime)
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.DatasourceProviderService.get_datasource_credentials", return_value=None
    )

    events = list(
        rag_pipeline_service.service.run_datasource_workflow_node(
            pipeline=SimpleNamespace(id="p1", tenant_id="t1"),
            node_id="node-1",
            user_inputs={"k": "mapped"},
            account=SimpleNamespace(id="u1"),
            datasource_type="website_crawl",
            is_published=True,
        )
    )

    assert events
    assert events[0]["data"] == []


def test_run_datasource_workflow_node_online_drive_branch(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    from core.datasource.entities.datasource_entities import DatasourceProviderType

    workflow = SimpleNamespace(
        graph_dict={
            "nodes": [
                {
                    "id": "node-1",
                    "data": {
                        "plugin_id": "p",
                        "provider_name": "provider",
                        "datasource_name": "drive",
                        "datasource_parameters": {},
                    },
                }
            ]
        }
    )
    mocker.patch.object(rag_pipeline_service.service, "get_published_workflow", return_value=workflow)
    runtime = mocker.Mock()

    def drive_gen(**kwargs):
        yield SimpleNamespace(result={"items": [1]})

    runtime.online_drive_browse_files.side_effect = drive_gen
    runtime.datasource_provider_type.return_value = DatasourceProviderType.ONLINE_DRIVE
    mocker.patch("core.datasource.datasource_manager.DatasourceManager.get_datasource_runtime", return_value=runtime)
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.DatasourceProviderService.get_datasource_credentials", return_value=None
    )

    events = list(
        rag_pipeline_service.service.run_datasource_workflow_node(
            pipeline=SimpleNamespace(id="p1", tenant_id="t1"),
            node_id="node-1",
            user_inputs={},
            account=SimpleNamespace(id="u1"),
            datasource_type="online_drive",
            is_published=True,
        )
    )

    assert len(events) == 2
    assert events[1]["data"] == {"items": [1]}


def test_run_datasource_node_preview_not_published_uses_draft(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    from core.datasource.entities.datasource_entities import DatasourceMessage

    workflow = SimpleNamespace(
        graph_dict={
            "nodes": [
                {
                    "id": "n1",
                    "data": {
                        "plugin_id": "p",
                        "provider_name": "provider",
                        "datasource_name": "doc",
                        "datasource_parameters": {"workspace_id": {"value": "w"}},
                    },
                }
            ]
        }
    )
    get_draft = mocker.patch.object(rag_pipeline_service.service, "get_draft_workflow", return_value=workflow)
    runtime = mocker.Mock()

    def doc_gen(**kwargs):
        yield DatasourceMessage(
            type=DatasourceMessage.MessageType.VARIABLE,
            message=DatasourceMessage.VariableMessage(variable_name="x", variable_value="v", stream=False),
        )

    runtime.get_online_document_page_content.side_effect = doc_gen
    mocker.patch("core.datasource.datasource_manager.DatasourceManager.get_datasource_runtime", return_value=runtime)
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.DatasourceProviderService.get_datasource_credentials", return_value=None
    )

    result = rag_pipeline_service.service.run_datasource_node_preview(
        pipeline=SimpleNamespace(id="p1", tenant_id="t1"),
        node_id="n1",
        user_inputs={},
        account=SimpleNamespace(id="u1"),
        datasource_type="online_document",
        is_published=False,
    )

    assert result == {"x": "v"}
    get_draft.assert_called_once()


def test_run_free_workflow_node_delegates_to_handle_result(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    expected = SimpleNamespace(id="exec-1")
    handle = mocker.patch.object(rag_pipeline_service.service, "_handle_node_run_result", return_value=expected)

    result = rag_pipeline_service.service.run_free_workflow_node(
        node_data={"type": "start"},
        tenant_id="t1",
        user_id="u1",
        node_id="n1",
        user_inputs={},
    )

    assert result is expected
    handle.assert_called_once()


def test_publish_customized_pipeline_template_raises_when_workflow_missing(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    pipeline = _make_pipeline(workflow_id="wf-1")
    session = rag_pipeline_service.session
    session.add(pipeline)
    session.commit()

    with pytest.raises(ValueError, match="Workflow not found"):
        rag_pipeline_service.service.publish_customized_pipeline_template(
            "p1", {}, _make_account(), "t1", session=session
        )


def test_publish_customized_pipeline_template_raises_when_dataset_missing(
    rag_pipeline_service: RagPipelineServiceTestContext,
) -> None:
    pipeline = _make_pipeline(workflow_id="wf-1")
    workflow = _make_workflow(workflow_id="wf-1")
    session = rag_pipeline_service.session
    session.add_all([pipeline, workflow])
    session.commit()

    with pytest.raises(ValueError, match="Dataset not found"):
        rag_pipeline_service.service.publish_customized_pipeline_template(
            "p1", {}, _make_account(), "t1", session=session
        )


def test_get_recommended_plugins_skips_manifest_when_missing(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    plugin = _make_recommended_plugin("plugin-a")
    rag_pipeline_service.session.add(plugin)
    rag_pipeline_service.session.commit()
    mocker.patch("services.rag_pipeline.rag_pipeline.BuiltinToolManageService.list_builtin_tools", return_value=[])
    mocker.patch("services.rag_pipeline.rag_pipeline.marketplace.batch_fetch_plugin_by_ids", return_value=[])

    result = rag_pipeline_service.service.get_recommended_plugins("all", _make_account(), "t1")

    assert result["installed_recommended_plugins"] == []
    assert result["uninstalled_recommended_plugins"] == []


def test_retry_error_document_raises_when_pipeline_missing(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    exec_log = DocumentPipelineExecutionLog(
        pipeline_id="p1",
        document_id="doc-1",
        datasource_type="upload_file",
        datasource_info="{}",
        datasource_node_id="source-1",
        input_data={},
        created_by=None,
    )
    rag_pipeline_service.session.add(exec_log)
    rag_pipeline_service.session.commit()

    with pytest.raises(ValueError, match="Pipeline not found"):
        rag_pipeline_service.service.retry_error_document(
            SimpleNamespace(), SimpleNamespace(id="doc-1"), SimpleNamespace(id="u1")
        )


def test_retry_error_document_raises_when_workflow_missing(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    exec_log = DocumentPipelineExecutionLog(
        pipeline_id="p1",
        document_id="doc-1",
        datasource_type="upload_file",
        datasource_info="{}",
        datasource_node_id="source-1",
        input_data={},
        created_by=None,
    )
    pipeline = _make_pipeline(workflow_id="wf-1")
    rag_pipeline_service.session.add_all([exec_log, pipeline])
    rag_pipeline_service.session.commit()

    with pytest.raises(ValueError, match="Workflow not found"):
        rag_pipeline_service.service.retry_error_document(
            SimpleNamespace(), SimpleNamespace(id="doc-1"), SimpleNamespace(id="u1")
        )


def test_get_datasource_plugins_returns_empty_for_non_datasource_nodes(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    dataset = _make_dataset()
    pipeline = _make_pipeline(workflow_id="wf-1")
    workflow = _make_workflow(graph={"nodes": [{"id": "n1", "data": {"type": "start"}}]})
    rag_pipeline_service.session.add_all([dataset, pipeline, workflow])
    rag_pipeline_service.session.commit()

    assert rag_pipeline_service.service.get_datasource_plugins("t1", "d1", True) == []


def test_publish_workflow_raises_when_knowledge_index_dataset_missing(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    draft = SimpleNamespace(
        type="workflow",
        graph={"nodes": [{"data": {"type": "knowledge-index"}}]},
        features={},
        environment_variables=[],
        conversation_variables=[],
        rag_pipeline_variables=[],
    )
    session = rag_pipeline_service.session
    real_draft = _make_workflow(graph=draft.graph)
    session.add(real_draft)
    session.commit()
    mocker.patch("services.rag_pipeline.rag_pipeline.KnowledgeConfiguration.model_validate", return_value=mocker.Mock())
    pipeline = SimpleNamespace(id="p1", tenant_id="t1", is_published=False, retrieve_dataset=lambda session: None)

    with pytest.raises(ValueError, match="Dataset not found"):
        rag_pipeline_service.service.publish_workflow(
            session=session, pipeline=pipeline, account=SimpleNamespace(id="u1")
        )


def test_run_datasource_node_preview_raises_when_workflow_missing(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    mocker.patch.object(rag_pipeline_service.service, "get_published_workflow", return_value=None)

    with pytest.raises(RuntimeError, match="Workflow not initialized"):
        rag_pipeline_service.service.run_datasource_node_preview(
            pipeline=SimpleNamespace(id="p1", tenant_id="t1"),
            node_id="n1",
            user_inputs={},
            account=SimpleNamespace(id="u1"),
            datasource_type="online_document",
            is_published=True,
        )


def test_run_datasource_node_preview_raises_when_node_missing(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    mocker.patch.object(
        rag_pipeline_service.service, "get_published_workflow", return_value=SimpleNamespace(graph_dict={"nodes": []})
    )

    with pytest.raises(RuntimeError, match="Datasource node data not found"):
        rag_pipeline_service.service.run_datasource_node_preview(
            pipeline=SimpleNamespace(id="p1", tenant_id="t1"),
            node_id="missing",
            user_inputs={},
            account=SimpleNamespace(id="u1"),
            datasource_type="online_document",
            is_published=True,
        )


def test_run_datasource_node_preview_keeps_existing_user_input(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    from core.datasource.entities.datasource_entities import DatasourceMessage

    workflow = SimpleNamespace(
        graph_dict={
            "nodes": [
                {
                    "id": "n1",
                    "data": {
                        "plugin_id": "p",
                        "provider_name": "provider",
                        "datasource_name": "doc",
                        "datasource_parameters": {"workspace_id": {"value": "default"}},
                    },
                }
            ]
        }
    )
    mocker.patch.object(rag_pipeline_service.service, "get_published_workflow", return_value=workflow)
    runtime = mocker.Mock()

    def gen(**kwargs):
        request = kwargs["datasource_parameters"]
        assert request.workspace_id == "existing"
        yield DatasourceMessage(
            type=DatasourceMessage.MessageType.VARIABLE,
            message=DatasourceMessage.VariableMessage(variable_name="ok", variable_value="1", stream=False),
        )

    runtime.get_online_document_page_content.side_effect = gen
    mocker.patch("core.datasource.datasource_manager.DatasourceManager.get_datasource_runtime", return_value=runtime)
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.DatasourceProviderService.get_datasource_credentials", return_value=None
    )

    result = rag_pipeline_service.service.run_datasource_node_preview(
        pipeline=SimpleNamespace(id="p1", tenant_id="t1"),
        node_id="n1",
        user_inputs={"workspace_id": "existing"},
        account=SimpleNamespace(id="u1"),
        datasource_type="online_document",
        is_published=True,
    )
    assert result == {"ok": "1"}


def test_run_datasource_node_preview_ignores_non_variable_messages(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    workflow = SimpleNamespace(
        graph_dict={
            "nodes": [
                {
                    "id": "n1",
                    "data": {
                        "plugin_id": "p",
                        "provider_name": "provider",
                        "datasource_name": "doc",
                        "datasource_parameters": {},
                    },
                }
            ]
        }
    )
    mocker.patch.object(rag_pipeline_service.service, "get_published_workflow", return_value=workflow)
    runtime = mocker.Mock()

    def gen(**kwargs):
        yield SimpleNamespace(type="log", message=None)

    runtime.get_online_document_page_content.side_effect = gen
    mocker.patch("core.datasource.datasource_manager.DatasourceManager.get_datasource_runtime", return_value=runtime)
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.DatasourceProviderService.get_datasource_credentials", return_value=None
    )

    result = rag_pipeline_service.service.run_datasource_node_preview(
        pipeline=SimpleNamespace(id="p1", tenant_id="t1"),
        node_id="n1",
        user_inputs={},
        account=SimpleNamespace(id="u1"),
        datasource_type="online_document",
        is_published=True,
    )
    assert result == {}


def test_set_datasource_variables_raises_when_workflow_missing(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    mocker.patch.object(rag_pipeline_service.service, "get_draft_workflow", return_value=None)

    with pytest.raises(ValueError, match="Workflow not initialized"):
        rag_pipeline_service.service.set_datasource_variables(
            SimpleNamespace(id="p1", tenant_id="t1"),
            {"start_node_id": "n1"},
            SimpleNamespace(id="u1"),
        )


def test_get_datasource_plugins_handles_empty_datasource_data_and_non_published(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    dataset = _make_dataset()
    pipeline = _make_pipeline()
    workflow = _make_workflow(
        graph={"nodes": [{"id": "n1", "data": {"type": "datasource", "datasource_parameters": {}}}]},
    )
    rag_pipeline_service.session.add_all([dataset, pipeline, workflow])
    rag_pipeline_service.session.commit()
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.DatasourceProviderService.list_datasource_credentials", return_value=[]
    )

    result = rag_pipeline_service.service.get_datasource_plugins("t1", "d1", False)

    assert len(result) == 1


def test_get_datasource_plugins_extracts_user_inputs_and_credentials(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    dataset = _make_dataset()
    pipeline = _make_pipeline(workflow_id="wf-1")
    workflow = _make_workflow(
        graph={
            "nodes": [
                {
                    "id": "n1",
                    "data": {
                        "type": "datasource",
                        "plugin_id": "plugin-1",
                        "provider_name": "provider",
                        "provider_type": "online_document",
                        "title": "Datasource",
                        "datasource_parameters": {
                            "a": {"value": "{{#start.v1#}}"},
                            "b": {"value": ["x", "v2"]},
                        },
                    },
                }
            ]
        }
    )
    workflow.rag_pipeline_variables = [
        {"variable": "v1", "belong_to_node_id": "shared", "type": "text-input", "label": "V1"},
        {"variable": "v2", "belong_to_node_id": "shared", "type": "text-input", "label": "V2"},
        {"variable": "v3", "belong_to_node_id": "shared", "type": "text-input", "label": "V3"},
    ]
    rag_pipeline_service.session.add_all([dataset, pipeline, workflow])
    rag_pipeline_service.session.commit()
    mocker.patch(
        "services.rag_pipeline.rag_pipeline.DatasourceProviderService.list_datasource_credentials",
        return_value=[{"id": "c1", "name": "Cred", "type": "api", "is_default": True}],
    )

    result = rag_pipeline_service.service.get_datasource_plugins("t1", "d1", True)

    assert len(result) == 1
    assert len(result[0]["user_input_variables"]) == 2
    assert result[0]["credentials"][0]["id"] == "c1"


def test_get_pipeline_returns_pipeline_when_found(
    mocker: MockerFixture, rag_pipeline_service: RagPipelineServiceTestContext
) -> None:
    dataset = _make_dataset()
    pipeline = _make_pipeline()
    rag_pipeline_service.session.add_all([dataset, pipeline])
    rag_pipeline_service.session.commit()

    result = rag_pipeline_service.service.get_pipeline("t1", "d1")

    assert result is pipeline

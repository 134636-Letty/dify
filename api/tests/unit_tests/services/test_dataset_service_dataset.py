"""Unit tests for DatasetService and dataset-related collaborators."""

from collections.abc import Iterator
from datetime import datetime
from uuid import uuid4

from sqlalchemy import event, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from models.base import TypeBase
from models.dataset import (
    Dataset,
    DatasetPermission,
    ExternalKnowledgeBindings,
    Pipeline,
)
from models.workflow import Workflow

from .dataset_service_test_helpers import (
    DatasetNameDuplicateError,
    DatasetPermissionEnum,
    DatasetPermissionService,
    DatasetService,
    DatasetServiceUnitDataFactory,
    LLMBadRequestError,
    MagicMock,
    ModelFeature,
    ModelType,
    NoPermissionError,
    PipelineIconInfo,
    ProviderTokenNotInitError,
    RagPipelineDatasetCreateEntity,
    SimpleNamespace,
    TenantAccountRole,
    _make_knowledge_configuration,
    _make_retrieval_model,
    json,
    patch,
    pytest,
)


@pytest.fixture
def dataset_session(sqlite_engine: Engine) -> Iterator[Session]:
    """Yield an isolated SQLite session for dataset service transactions."""

    models = (Dataset, DatasetPermission, ExternalKnowledgeBindings, Pipeline, Workflow)
    tables = [TypeBase.metadata.tables[model.__tablename__] for model in models]
    TypeBase.metadata.create_all(sqlite_engine, tables=tables)
    with Session(sqlite_engine, expire_on_commit=False) as session:
        yield session


def _persist_dataset(
    session: Session,
    *,
    dataset_id: str,
    tenant_id: str,
    name: str = "Dataset",
    created_by: str | None = None,
    provider: str = "vendor",
) -> Dataset:
    account_id = created_by or str(uuid4())
    dataset = Dataset(
        id=dataset_id,
        tenant_id=tenant_id,
        name=name,
        provider=provider,
        created_by=account_id,
        maintainer=account_id,
        indexing_technique="economy",
    )
    session.add(dataset)
    session.commit()
    return dataset


class TestDatasetServiceValidation:
    """Unit tests for DatasetService validation helpers."""

    @pytest.mark.parametrize(
        ("dataset_doc_form", "incoming_doc_form"),
        [(None, "text_model"), ("text_model", "text_model")],
    )
    def test_check_doc_form_allows_matching_or_missing_dataset_doc_form(self, dataset_doc_form, incoming_doc_form):
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(doc_form=dataset_doc_form)

        DatasetService.check_doc_form(dataset, incoming_doc_form)

    def test_check_doc_form_rejects_mismatched_doc_form(self):
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(doc_form="qa_model")

        with pytest.raises(ValueError, match="doc_form is different"):
            DatasetService.check_doc_form(dataset, "text_model")

    def test_check_dataset_model_setting_skips_non_high_quality_datasets(self):
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(indexing_technique="economy")

        with patch("services.dataset_service.ModelManager") as model_manager_cls:
            DatasetService.check_dataset_model_setting(dataset)

        model_manager_cls.assert_not_called()

    def test_check_dataset_model_setting_validates_embedding_model_for_high_quality_dataset(self):
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(indexing_technique="high_quality")

        with patch("services.dataset_service.ModelManager") as model_manager_cls:
            DatasetService.check_dataset_model_setting(dataset)

        model_manager_cls.for_tenant.return_value.get_model_instance.assert_called_once_with(
            tenant_id=dataset.tenant_id,
            provider=dataset.embedding_model_provider,
            model_type=ModelType.TEXT_EMBEDDING,
            model=dataset.embedding_model,
        )

    def test_check_dataset_model_setting_wraps_llm_bad_request_error(self):
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(indexing_technique="high_quality")

        with patch("services.dataset_service.ModelManager") as model_manager_cls:
            model_manager_cls.for_tenant.return_value.get_model_instance.side_effect = LLMBadRequestError()

            with pytest.raises(ValueError, match="No Embedding Model available"):
                DatasetService.check_dataset_model_setting(dataset)

    def test_check_dataset_model_setting_wraps_provider_token_error(self):
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(indexing_technique="high_quality")

        with patch("services.dataset_service.ModelManager") as model_manager_cls:
            model_manager_cls.for_tenant.return_value.get_model_instance.side_effect = ProviderTokenNotInitError(
                "token missing"
            )

            with pytest.raises(ValueError, match="The dataset is unavailable, due to: token missing"):
                DatasetService.check_dataset_model_setting(dataset)

    def test_check_embedding_model_setting_wraps_provider_token_error_description(self):
        with patch("services.dataset_service.ModelManager") as model_manager_cls:
            model_manager_cls.for_tenant.return_value.get_model_instance.side_effect = ProviderTokenNotInitError(
                "provider setup"
            )

            with pytest.raises(ValueError, match="provider setup"):
                DatasetService.check_embedding_model_setting("tenant-1", "provider", "embedding-model")

    def test_check_reranking_model_setting_uses_rerank_model_type(self):
        with patch("services.dataset_service.ModelManager") as model_manager_cls:
            DatasetService.check_reranking_model_setting("tenant-1", "provider", "reranker")

        model_manager_cls.for_tenant.return_value.get_model_instance.assert_called_once_with(
            tenant_id="tenant-1",
            provider="provider",
            model_type=ModelType.RERANK,
            model="reranker",
        )

    def test_check_reranking_model_setting_wraps_bad_request(self):
        with patch("services.dataset_service.ModelManager") as model_manager_cls:
            model_manager_cls.for_tenant.return_value.get_model_instance.side_effect = LLMBadRequestError()

            with pytest.raises(ValueError, match="No Rerank Model available"):
                DatasetService.check_reranking_model_setting("tenant-1", "provider", "reranker")

    def test_check_is_multimodal_model_returns_true_when_model_supports_vision(self):
        model_schema = SimpleNamespace(features=[ModelFeature.VISION])
        model_type_instance = MagicMock()
        model_type_instance.get_model_schema.return_value = model_schema
        model_instance = SimpleNamespace(
            model_type_instance=model_type_instance,
            model_name="embedding-model",
            credentials={"api_key": "secret"},
        )

        with patch("services.dataset_service.ModelManager") as model_manager_cls:
            model_manager_cls.for_tenant.return_value.get_model_instance.return_value = model_instance

            result = DatasetService.check_is_multimodal_model("tenant-1", "provider", "embedding-model")

        assert result is True

    def test_check_is_multimodal_model_returns_false_when_vision_feature_is_absent(self):
        model_schema = SimpleNamespace(features=[])
        model_type_instance = MagicMock()
        model_type_instance.get_model_schema.return_value = model_schema
        model_instance = SimpleNamespace(
            model_type_instance=model_type_instance,
            model_name="embedding-model",
            credentials={"api_key": "secret"},
        )

        with patch("services.dataset_service.ModelManager") as model_manager_cls:
            model_manager_cls.for_tenant.return_value.get_model_instance.return_value = model_instance

            result = DatasetService.check_is_multimodal_model("tenant-1", "provider", "embedding-model")

        assert result is False

    def test_check_is_multimodal_model_raises_when_schema_is_missing(self):
        model_type_instance = MagicMock()
        model_type_instance.get_model_schema.return_value = None
        model_instance = SimpleNamespace(
            model_type_instance=model_type_instance,
            model_name="embedding-model",
            credentials={"api_key": "secret"},
        )

        with patch("services.dataset_service.ModelManager") as model_manager_cls:
            model_manager_cls.for_tenant.return_value.get_model_instance.return_value = model_instance

            with pytest.raises(ValueError, match="Model schema not found"):
                DatasetService.check_is_multimodal_model("tenant-1", "provider", "embedding-model")

    def test_check_is_multimodal_model_wraps_bad_request_error(self):
        with patch("services.dataset_service.ModelManager") as model_manager_cls:
            model_manager_cls.for_tenant.return_value.get_model_instance.side_effect = LLMBadRequestError()

            with pytest.raises(ValueError, match="No Model available"):
                DatasetService.check_is_multimodal_model("tenant-1", "provider", "embedding-model")


class TestDatasetServiceRetrievalPermissions:
    """Unit tests for dataset list permission branching."""

    def test_get_datasets_filters_by_maintainer_and_rbac_overrides(self, dataset_session: Session):
        tenant_id = str(uuid4())
        user = DatasetServiceUnitDataFactory.create_user_mock(
            user_id=str(uuid4()), tenant_id=tenant_id, role=TenantAccountRole.NORMAL
        )

        with (
            patch("services.dataset_service.paginate_query") as mock_paginate,
            patch("services.dataset_service.dify_config.RBAC_ENABLED", True),
            patch(
                "services.dataset_service.enterprise_rbac_service.RBACService.MyPermissions.get",
                return_value=SimpleNamespace(workspace=SimpleNamespace(permission_keys=[])),
            ),
        ):
            mock_paginate.return_value = SimpleNamespace(items=[], total=0)
            DatasetService.get_datasets(
                page=1,
                per_page=20,
                session=dataset_session,
                tenant_id=tenant_id,
                user=user,
                accessible_dataset_ids=["dataset-shared"],
                include_own_datasets=True,
            )

        select_stmt = mock_paginate.call_args.args[0]
        visibility_clause = str(select_stmt._where_criteria[1])
        assert "maintainer" in visibility_clause
        assert "IN" in visibility_clause

    def test_get_datasets_filters_only_by_rbac_overrides_without_manage_own_permission(self, dataset_session: Session):
        tenant_id = str(uuid4())
        user = DatasetServiceUnitDataFactory.create_user_mock(
            user_id=str(uuid4()), tenant_id=tenant_id, role=TenantAccountRole.NORMAL
        )

        with (
            patch("services.dataset_service.paginate_query") as mock_paginate,
            patch("services.dataset_service.dify_config.RBAC_ENABLED", True),
            patch(
                "services.dataset_service.enterprise_rbac_service.RBACService.MyPermissions.get",
                return_value=SimpleNamespace(workspace=SimpleNamespace(permission_keys=[])),
            ),
        ):
            mock_paginate.return_value = SimpleNamespace(items=[], total=0)
            DatasetService.get_datasets(
                page=1,
                per_page=20,
                session=dataset_session,
                tenant_id=tenant_id,
                user=user,
                accessible_dataset_ids=["dataset-shared"],
            )

        select_stmt = mock_paginate.call_args.args[0]
        visibility_clause = str(select_stmt._where_criteria[1])
        assert "maintainer" not in visibility_clause
        assert "IN" in visibility_clause

    def test_get_datasets_by_ids_applies_rbac_visibility(self):
        user = DatasetServiceUnitDataFactory.create_user_mock(role=TenantAccountRole.NORMAL)

        with (
            patch("services.dataset_service.paginate_query") as mock_paginate,
            patch("services.dataset_service.dify_config.RBAC_ENABLED", True),
        ):
            mock_paginate.return_value = SimpleNamespace(items=[], total=0)
            DatasetService.get_datasets_by_ids(
                ["dataset-requested", "dataset-shared"],
                "tenant-1",
                user=user,
                accessible_dataset_ids=["dataset-shared", "dataset-not-requested"],
                include_own_datasets=True,
            )

        select_stmt = mock_paginate.call_args.args[0]
        visibility_clause = str(select_stmt._where_criteria[-1])
        assert "maintainer" in visibility_clause
        assert "IN" in visibility_clause
        visibility_params = select_stmt._where_criteria[-1].compile().params
        assert ["dataset-shared"] in visibility_params.values()
        list_params = [value for value in visibility_params.values() if isinstance(value, list)]
        assert all("dataset-not-requested" not in value for value in list_params)

    def test_get_datasets_rbac_include_all_uses_workspace_permission(self, dataset_session: Session):
        tenant_id = str(uuid4())
        user = DatasetServiceUnitDataFactory.create_user_mock(
            user_id=str(uuid4()), tenant_id=tenant_id, role=TenantAccountRole.NORMAL
        )
        mock_permissions = SimpleNamespace(workspace=SimpleNamespace(permission_keys=["dataset.create_and_management"]))

        with (
            patch("services.dataset_service.paginate_query") as mock_paginate,
            patch("services.dataset_service.dify_config.RBAC_ENABLED", True),
            patch(
                "services.dataset_service.enterprise_rbac_service.RBACService.MyPermissions.get",
                return_value=mock_permissions,
            ),
        ):
            mock_paginate.return_value = SimpleNamespace(items=[], total=0)
            DatasetService.get_datasets(
                page=1,
                per_page=20,
                session=dataset_session,
                tenant_id=tenant_id,
                user=user,
                include_all=True,
            )

        mock_paginate.assert_called_once()
        select_stmt = mock_paginate.call_args.args[0]
        assert len(select_stmt._where_criteria) == 1

    def test_get_datasets_rbac_without_user_returns_empty_result(self, dataset_session: Session):
        with (
            patch("services.dataset_service.paginate_query") as mock_paginate,
            patch("services.dataset_service.dify_config.RBAC_ENABLED", True),
        ):
            mock_paginate.return_value = SimpleNamespace(items=[], total=0)
            DatasetService.get_datasets(page=1, per_page=20, session=dataset_session, tenant_id=str(uuid4()), user=None)

        mock_paginate.assert_called_once()
        select_stmt = mock_paginate.call_args.args[0]
        assert len(select_stmt._where_criteria) == 2

    def test_get_datasets_legacy_owner_include_all_keeps_full_access(self, dataset_session: Session):
        tenant_id = str(uuid4())
        user = DatasetServiceUnitDataFactory.create_user_mock(
            user_id=str(uuid4()), tenant_id=tenant_id, role=TenantAccountRole.OWNER
        )

        with (
            patch("services.dataset_service.paginate_query") as mock_paginate,
            patch("services.dataset_service.dify_config.RBAC_ENABLED", False),
        ):
            mock_paginate.return_value = SimpleNamespace(items=[], total=0)
            DatasetService.get_datasets(
                page=1,
                per_page=20,
                session=dataset_session,
                tenant_id=tenant_id,
                user=user,
                include_all=True,
            )

        mock_paginate.assert_called_once()
        select_stmt = mock_paginate.call_args.args[0]
        assert len(select_stmt._where_criteria) == 1


class TestDatasetServiceCreationAndUpdate:
    """Unit tests for dataset creation and update helpers."""

    def test_create_empty_dataset_raises_when_name_already_exists(self, dataset_session: Session):
        tenant_id = str(uuid4())
        account = SimpleNamespace(id=str(uuid4()))
        _persist_dataset(dataset_session, dataset_id=str(uuid4()), tenant_id=tenant_id, name="Dataset")
        _persist_dataset(dataset_session, dataset_id=str(uuid4()), tenant_id=str(uuid4()), name="Dataset")

        with pytest.raises(DatasetNameDuplicateError, match="Dataset with name Dataset already exists"):
            DatasetService.create_empty_dataset(tenant_id, "Dataset", None, "economy", account, session=dataset_session)

    def test_create_empty_dataset_uses_default_embedding_model_for_high_quality_dataset(self, dataset_session: Session):
        tenant_id = str(uuid4())
        account = SimpleNamespace(id=str(uuid4()))
        default_embedding_model = SimpleNamespace(provider="provider", model_name="default-embedding")

        with (
            patch("services.dataset_service.ModelManager") as model_manager_cls,
            patch.object(DatasetService, "check_embedding_model_setting") as check_embedding,
        ):
            model_manager_cls.for_tenant.return_value.get_default_model_instance.return_value = default_embedding_model

            dataset = DatasetService.create_empty_dataset(
                tenant_id=tenant_id,
                name="Dataset",
                description="Description",
                indexing_technique="high_quality",
                account=account,
                session=dataset_session,
            )

        assert dataset.embedding_model_provider == "provider"
        assert dataset.embedding_model == "default-embedding"
        assert dataset.permission == DatasetPermissionEnum.ONLY_ME
        assert dataset.provider == "vendor"
        model_manager_cls.for_tenant.return_value.get_default_model_instance.assert_called_once_with(
            tenant_id=tenant_id,
            model_type=ModelType.TEXT_EMBEDDING,
        )
        check_embedding.assert_not_called()
        assert dataset_session.get(Dataset, dataset.id) is dataset

    def test_create_empty_dataset_creates_external_binding_for_high_quality_dataset(self, dataset_session: Session):
        tenant_id = str(uuid4())
        account = SimpleNamespace(id=str(uuid4()))
        api_id = str(uuid4())
        retrieval_model = _make_retrieval_model()
        embedding_model = SimpleNamespace(provider="provider", model_name="embedding-model")

        with (
            patch("services.dataset_service.ModelManager") as model_manager_cls,
            patch("services.dataset_service.ExternalDatasetService.get_external_knowledge_api", return_value=object()),
            patch.object(DatasetService, "check_embedding_model_setting") as check_embedding,
            patch.object(DatasetService, "check_reranking_model_setting") as check_reranking,
        ):
            model_manager_cls.for_tenant.return_value.get_model_instance.return_value = embedding_model

            dataset = DatasetService.create_empty_dataset(
                tenant_id=tenant_id,
                name="External Dataset",
                description="Description",
                indexing_technique="high_quality",
                account=account,
                permission=DatasetPermissionEnum.ALL_TEAM,
                provider="external",
                external_knowledge_api_id=api_id,
                external_knowledge_id="knowledge-1",
                embedding_model_provider="provider",
                embedding_model_name="embedding-model",
                retrieval_model=retrieval_model,
                summary_index_setting={"enable": True},
                session=dataset_session,
            )

        assert dataset.embedding_model_provider == "provider"
        assert dataset.embedding_model == "embedding-model"
        assert dataset.retrieval_model == retrieval_model.model_dump()
        assert dataset.summary_index_setting == {"enable": True}
        check_embedding.assert_called_once_with(tenant_id, "provider", "embedding-model")
        check_reranking.assert_called_once_with(tenant_id, "rerank-provider", "rerank-model")
        binding = dataset_session.scalar(
            select(ExternalKnowledgeBindings).where(ExternalKnowledgeBindings.dataset_id == dataset.id)
        )
        assert binding is not None
        assert binding.external_knowledge_api_id == api_id
        assert binding.external_knowledge_id == "knowledge-1"

    def test_create_empty_rag_pipeline_dataset_raises_for_duplicate_name(self, dataset_session: Session):
        entity = RagPipelineDatasetCreateEntity(
            name="Existing Dataset",
            description="Description",
            icon_info=PipelineIconInfo(icon="book", icon_background="#fff"),
            permission=DatasetPermissionEnum.ALL_TEAM,
        )

        tenant_id = str(uuid4())
        _persist_dataset(dataset_session, dataset_id=str(uuid4()), tenant_id=tenant_id, name="Existing Dataset")
        with pytest.raises(DatasetNameDuplicateError, match="Existing Dataset already exists"):
            DatasetService.create_empty_rag_pipeline_dataset(tenant_id, entity, dataset_session)

    def test_create_empty_rag_pipeline_dataset_generates_name_and_creates_dataset(self, dataset_session: Session):
        entity = RagPipelineDatasetCreateEntity(
            name="",
            description="Description",
            icon_info=PipelineIconInfo(icon="book", icon_background="#fff"),
            permission=DatasetPermissionEnum.ALL_TEAM,
        )
        tenant_id = str(uuid4())
        user_id = str(uuid4())
        _persist_dataset(dataset_session, dataset_id=str(uuid4()), tenant_id=tenant_id, name="Untitled")
        _persist_dataset(dataset_session, dataset_id=str(uuid4()), tenant_id=tenant_id, name="Untitled 1")

        with (
            patch("services.dataset_service.current_user", SimpleNamespace(id=user_id)),
            patch("services.dataset_service.generate_incremental_name", return_value="Untitled 2") as generate_name,
        ):
            dataset = DatasetService.create_empty_rag_pipeline_dataset(tenant_id, entity, dataset_session)

        assert entity.name == "Untitled 2"
        assert dataset_session.get(Pipeline, dataset.pipeline_id) is not None
        assert dataset.runtime_mode == "rag_pipeline"
        generate_name.assert_called_once()
        assert set(generate_name.call_args.args[0]) == {"Untitled", "Untitled 1"}
        assert generate_name.call_args.args[1] == "Untitled"

    def test_create_empty_rag_pipeline_dataset_requires_current_user_id(self, dataset_session: Session):
        entity = RagPipelineDatasetCreateEntity(
            name="Dataset",
            description="Description",
            icon_info=PipelineIconInfo(icon="book", icon_background="#fff"),
            permission=DatasetPermissionEnum.ALL_TEAM,
        )

        with (
            patch("services.dataset_service.current_user", SimpleNamespace(id=None)),
        ):
            with pytest.raises(ValueError, match="Current user or current user id not found"):
                DatasetService.create_empty_rag_pipeline_dataset(str(uuid4()), entity, dataset_session)

    def test_update_dataset_raises_when_dataset_is_missing(self, dataset_session: Session):
        with patch.object(DatasetService, "get_dataset", return_value=None):
            with pytest.raises(ValueError, match="Dataset not found"):
                DatasetService.update_dataset(
                    str(uuid4()), {}, SimpleNamespace(id=str(uuid4())), session=dataset_session
                )

    def test_update_dataset_raises_when_new_name_conflicts(self, dataset_session: Session):
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(dataset_id="dataset-1", tenant_id="tenant-1")
        dataset.name = "Old Dataset"

        with (
            patch.object(DatasetService, "get_dataset", return_value=dataset),
            patch.object(DatasetService, "_has_dataset_same_name", return_value=True),
        ):
            with pytest.raises(ValueError, match="Dataset name already exists"):
                DatasetService.update_dataset(
                    "dataset-1", {"name": "New Dataset"}, SimpleNamespace(id="user-1"), session=dataset_session
                )

    def test_update_dataset_routes_external_datasets_to_external_helper(self, dataset_session: Session):
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(dataset_id="dataset-1", tenant_id="tenant-1")
        dataset.provider = "external"
        user = DatasetServiceUnitDataFactory.create_user_mock()

        with (
            patch.object(DatasetService, "get_dataset", return_value=dataset),
            patch.object(DatasetService, "check_dataset_permission") as check_permission,
            patch.object(DatasetService, "_update_external_dataset", return_value="updated") as update_external,
        ):
            result = DatasetService.update_dataset("dataset-1", {"name": dataset.name}, user, session=dataset_session)

        assert result == "updated"
        check_permission.assert_called_once()
        assert check_permission.call_args.args[:2] == (dataset, user)
        assert len(check_permission.call_args.args) == 3
        update_external.assert_called_once_with(dataset, {"name": dataset.name}, user, dataset_session)

    def test_update_dataset_routes_internal_datasets_to_internal_helper(self, dataset_session: Session):
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(dataset_id="dataset-1", tenant_id="tenant-1")
        dataset.provider = "vendor"
        user = DatasetServiceUnitDataFactory.create_user_mock()

        with (
            patch.object(DatasetService, "get_dataset", return_value=dataset),
            patch.object(DatasetService, "check_dataset_permission") as check_permission,
            patch.object(DatasetService, "_update_internal_dataset", return_value="updated") as update_internal,
        ):
            result = DatasetService.update_dataset("dataset-1", {"name": dataset.name}, user, session=dataset_session)

        assert result == "updated"
        check_permission.assert_called_once()
        assert check_permission.call_args.args[:2] == (dataset, user)
        assert len(check_permission.call_args.args) == 3
        update_internal.assert_called_once_with(dataset, {"name": dataset.name}, user, dataset_session)

    def test_has_dataset_same_name_returns_true_when_query_matches(self, dataset_session: Session):
        tenant_id = str(uuid4())
        existing = _persist_dataset(dataset_session, dataset_id=str(uuid4()), tenant_id=tenant_id)
        result = DatasetService._has_dataset_same_name(tenant_id, str(uuid4()), existing.name, dataset_session)

        assert result is True

    def test_update_external_dataset_updates_dataset_and_binding(self, dataset_session: Session):
        tenant_id = str(uuid4())
        dataset = _persist_dataset(dataset_session, dataset_id=str(uuid4()), tenant_id=tenant_id, provider="external")
        user = SimpleNamespace(id=str(uuid4()))
        api_id = str(uuid4())
        now = datetime(2026, 1, 1)

        with (
            patch.object(DatasetService, "_update_external_knowledge_binding") as update_binding,
            patch(
                "services.dataset_service.ExternalDatasetService.get_external_knowledge_api", return_value=object()
            ) as get_external_knowledge_api,
            patch("services.dataset_service.naive_utc_now", return_value=now),
        ):
            result = DatasetService._update_external_dataset(
                dataset,
                {
                    "external_retrieval_model": {"top_k": 3},
                    "summary_index_setting": {"enable": True},
                    "name": "Updated Dataset",
                    "description": "Updated description",
                    "permission": DatasetPermissionEnum.PARTIAL_TEAM,
                    "external_knowledge_id": "knowledge-1",
                    "external_knowledge_api_id": api_id,
                },
                user,
                dataset_session,
            )

        assert result is dataset
        assert dataset.retrieval_model == {"top_k": 3}
        assert dataset.summary_index_setting == {"enable": True}
        assert dataset.name == "Updated Dataset"
        assert dataset.description == "Updated description"
        assert dataset.permission == DatasetPermissionEnum.PARTIAL_TEAM
        assert dataset.updated_by == user.id
        assert dataset.updated_at is now
        get_external_knowledge_api.assert_called_once_with(api_id, dataset.tenant_id, session=dataset_session)
        update_binding.assert_called_once_with(dataset.id, "knowledge-1", api_id, dataset_session)
        dataset_session.expire_all()
        assert dataset_session.get(Dataset, dataset.id).name == "Updated Dataset"

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            ({"external_knowledge_api_id": "api-1"}, "External knowledge id is required"),
            ({"external_knowledge_id": "knowledge-1"}, "External knowledge api id is required"),
        ],
    )
    def test_update_external_dataset_requires_external_binding_fields(self, payload, message, dataset_session: Session):
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(dataset_id="dataset-1")

        with pytest.raises(ValueError, match=message):
            DatasetService._update_external_dataset(dataset, payload, SimpleNamespace(id="user-1"), dataset_session)

    def test_update_external_dataset_rejects_cross_tenant_external_api_id(self, dataset_session: Session):
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(dataset_id="dataset-1")

        with (
            patch(
                "services.dataset_service.ExternalDatasetService.get_external_knowledge_api",
                side_effect=ValueError("api template not found"),
            ) as get_external_knowledge_api,
            patch.object(DatasetService, "_update_external_knowledge_binding") as update_binding,
        ):
            with pytest.raises(ValueError, match="api template not found"):
                DatasetService._update_external_dataset(
                    dataset,
                    {
                        "external_knowledge_id": "knowledge-1",
                        "external_knowledge_api_id": "foreign-api",
                    },
                    SimpleNamespace(id="user-1"),
                    dataset_session,
                )

        get_external_knowledge_api.assert_called_once_with("foreign-api", dataset.tenant_id, session=dataset_session)
        update_binding.assert_not_called()

    def test_update_external_knowledge_binding_updates_changed_binding_values(self, dataset_session: Session):
        dataset_id = str(uuid4())
        new_api_id = str(uuid4())
        binding = ExternalKnowledgeBindings(
            tenant_id=str(uuid4()),
            dataset_id=dataset_id,
            external_knowledge_api_id=str(uuid4()),
            external_knowledge_id="old-knowledge",
            created_by=str(uuid4()),
        )
        dataset_session.add(binding)
        dataset_session.commit()
        DatasetService._update_external_knowledge_binding(dataset_id, "new-knowledge", new_api_id, dataset_session)

        assert binding.external_knowledge_id == "new-knowledge"
        assert binding.external_knowledge_api_id == new_api_id

    def test_update_external_knowledge_binding_raises_for_missing_binding(self, dataset_session: Session):
        with pytest.raises(ValueError, match="External knowledge binding not found"):
            DatasetService._update_external_knowledge_binding(
                str(uuid4()), "knowledge-1", str(uuid4()), dataset_session
            )

    def test_update_internal_dataset_updates_fields_and_dispatches_regeneration_tasks(self, dataset_session: Session):
        tenant_id = str(uuid4())
        dataset = _persist_dataset(dataset_session, dataset_id=str(uuid4()), tenant_id=tenant_id)
        user = SimpleNamespace(id=str(uuid4()))
        update_payload = {
            "name": "Updated Dataset",
            "description": None,
            "partial_member_list": [{"user_id": "member-1"}],
            "external_knowledge_api_id": "api-1",
            "external_knowledge_id": "knowledge-1",
            "external_retrieval_model": {"top_k": 2},
            "retrieval_model": {"top_k": 4},
            "summary_index_setting": {"enable": True},
            "icon_info": {"icon": "book"},
        }

        with (
            patch.object(DatasetService, "_handle_indexing_technique_change", return_value="update"),
            patch.object(DatasetService, "_update_pipeline_knowledge_base_node_data") as update_pipeline,
            patch("services.dataset_service.deal_dataset_vector_index_task") as vector_task,
            patch("services.dataset_service.regenerate_summary_index_task") as regenerate_task,
        ):
            result = DatasetService._update_internal_dataset(dataset, update_payload.copy(), user, dataset_session)

        assert result is dataset
        assert dataset.name == "Updated Dataset"
        assert dataset.description is None
        assert dataset.retrieval_model == {"top_k": 4}
        assert dataset.summary_index_setting == {"enable": True}
        assert dataset.icon_info == {"icon": "book"}
        assert dataset.updated_by == user.id
        update_pipeline.assert_called_once_with(dataset, user.id, dataset_session)
        vector_task.delay.assert_called_once_with(dataset.id, "update")
        regenerate_task.delay.assert_called_once_with(
            dataset.id,
            regenerate_reason="embedding_model_changed",
            regenerate_vectors_only=True,
        )

    def test_update_pipeline_knowledge_base_node_data_returns_early_for_non_pipeline_dataset(
        self, dataset_session: Session
    ):
        dataset = SimpleNamespace(runtime_mode="workflow", pipeline_id="pipeline-1")
        DatasetService._update_pipeline_knowledge_base_node_data(dataset, "user-1", dataset_session)

    def test_update_pipeline_knowledge_base_node_data_returns_when_pipeline_is_missing(self, dataset_session: Session):
        dataset = SimpleNamespace(runtime_mode="rag_pipeline", pipeline_id=str(uuid4()))
        DatasetService._update_pipeline_knowledge_base_node_data(dataset, str(uuid4()), dataset_session)

    def test_update_pipeline_knowledge_base_node_data_updates_published_and_draft_workflows(
        self, dataset_session: Session
    ):
        tenant_id = str(uuid4())
        pipeline_id = str(uuid4())
        user_id = str(uuid4())
        dataset = SimpleNamespace(
            id=str(uuid4()),
            runtime_mode="rag_pipeline",
            pipeline_id=pipeline_id,
            embedding_model="embedding-model",
            embedding_model_provider="provider",
            retrieval_model={"top_k": 5},
            chunk_structure="paragraph",
            indexing_technique="high_quality",
            keyword_number=8,
            summary_index_setting={"enable": True},
        )
        pipeline = Pipeline(tenant_id=tenant_id, name="Pipeline", description="", created_by=user_id)
        pipeline.id = pipeline_id
        published_workflow = Workflow.new(
            tenant_id=tenant_id,
            app_id=pipeline_id,
            type="rag-pipeline",
            version="published",
            graph=json.dumps({"nodes": [{"data": {"type": "knowledge-index"}}, {"data": {"type": "start"}}]}),
            features="{}",
            created_by=user_id,
            environment_variables=[],
            conversation_variables=[],
            rag_pipeline_variables=[],
        )
        draft_workflow = Workflow.new(
            tenant_id=tenant_id,
            app_id=pipeline_id,
            type="rag-pipeline",
            version="draft",
            graph=json.dumps({"nodes": [{"data": {"type": "knowledge-index"}}]}),
            features="{}",
            created_by=user_id,
            environment_variables=[],
            conversation_variables=[],
            rag_pipeline_variables=[],
        )
        dataset_session.add_all([pipeline, draft_workflow])
        dataset_session.commit()
        rag_pipeline_service = MagicMock()
        rag_pipeline_service.get_published_workflow.return_value = published_workflow
        rag_pipeline_service.get_draft_workflow.return_value = draft_workflow

        with (
            patch("services.dataset_service.RagPipelineService", return_value=rag_pipeline_service),
        ):
            DatasetService._update_pipeline_knowledge_base_node_data(dataset, user_id, dataset_session)

        workflows = dataset_session.scalars(select(Workflow).where(Workflow.app_id == pipeline_id)).all()
        published_copy = next(workflow for workflow in workflows if workflow.version != "draft")
        published_graph = json.loads(published_copy.graph)
        assert published_graph["nodes"][0]["data"]["embedding_model"] == "embedding-model"
        assert published_graph["nodes"][0]["data"]["summary_index_setting"] == {"enable": True}
        assert json.loads(draft_workflow.graph)["nodes"][0]["data"]["embedding_model_provider"] == "provider"

    def test_update_pipeline_knowledge_base_node_data_rolls_back_when_update_fails(self, dataset_session: Session):
        pipeline_id = str(uuid4())
        dataset = SimpleNamespace(runtime_mode="rag_pipeline", pipeline_id=pipeline_id)
        pipeline = Pipeline(tenant_id=str(uuid4()), name="Pipeline", description="", created_by=str(uuid4()))
        pipeline.id = pipeline_id
        dataset_session.add(pipeline)
        dataset_session.commit()
        rag_pipeline_service = MagicMock()
        rag_pipeline_service.get_published_workflow.side_effect = RuntimeError("boom")

        with (
            patch("services.dataset_service.RagPipelineService", return_value=rag_pipeline_service),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                DatasetService._update_pipeline_knowledge_base_node_data(dataset, str(uuid4()), dataset_session)

        assert not dataset_session.in_transaction()

    def test_handle_indexing_technique_change_returns_none_without_indexing_technique(self, dataset_session: Session):
        filtered_data: dict[str, object] = {}
        dataset = SimpleNamespace(indexing_technique="economy")
        result = DatasetService._handle_indexing_technique_change(dataset, {}, filtered_data, dataset_session)

        assert result is None
        assert filtered_data == {}

    def test_handle_indexing_technique_change_switches_to_economy(self, dataset_session: Session):
        filtered_data: dict[str, object] = {}
        dataset = SimpleNamespace(indexing_technique="high_quality")
        result = DatasetService._handle_indexing_technique_change(
            dataset,
            {"indexing_technique": "economy"},
            filtered_data,
            dataset_session,
        )

        assert result == "remove"
        assert filtered_data == {
            "embedding_model": None,
            "embedding_model_provider": None,
            "collection_binding_id": None,
        }

    def test_handle_indexing_technique_change_switches_to_high_quality(self, dataset_session: Session):
        filtered_data: dict[str, object] = {}
        dataset = SimpleNamespace(indexing_technique="economy")
        with patch.object(DatasetService, "_configure_embedding_model_for_high_quality") as configure_embedding:
            result = DatasetService._handle_indexing_technique_change(
                dataset,
                {"indexing_technique": "high_quality"},
                filtered_data,
                dataset_session,
            )

        assert result == "add"
        configure_embedding.assert_called_once_with(
            {"indexing_technique": "high_quality"}, filtered_data, dataset_session
        )

    def test_handle_indexing_technique_change_delegates_when_technique_is_unchanged(self, dataset_session: Session):
        filtered_data: dict[str, object] = {}
        dataset = SimpleNamespace(indexing_technique="high_quality")
        with patch.object(
            DatasetService,
            "_handle_embedding_model_update_when_technique_unchanged",
            return_value="update",
        ) as update_embedding:
            result = DatasetService._handle_indexing_technique_change(
                dataset,
                {"indexing_technique": "high_quality"},
                filtered_data,
                dataset_session,
            )

        assert result == "update"
        update_embedding.assert_called_once_with(
            dataset,
            {"indexing_technique": "high_quality"},
            filtered_data,
            dataset_session,
        )

    def test_configure_embedding_model_for_high_quality_updates_filtered_data(self, dataset_session: Session):
        class FakeAccount:
            pass

        current_user = FakeAccount()
        current_user.current_tenant_id = "tenant-1"
        embedding_model = SimpleNamespace(provider="provider", model_name="embedding-model")
        filtered_data: dict[str, object] = {}
        with (
            patch("services.dataset_service.Account", FakeAccount),
            patch("services.dataset_service.current_user", current_user),
            patch("services.dataset_service.ModelManager") as model_manager_cls,
            patch(
                "services.dataset_service.DatasetCollectionBindingService.get_dataset_collection_binding",
                return_value=SimpleNamespace(id="binding-1"),
            ),
        ):
            model_manager_cls.for_tenant.return_value.get_model_instance.return_value = embedding_model

            DatasetService._configure_embedding_model_for_high_quality(
                {"embedding_model_provider": "provider", "embedding_model": "embedding-model"},
                filtered_data,
                dataset_session,
            )

        assert filtered_data == {
            "embedding_model": "embedding-model",
            "embedding_model_provider": "provider",
            "collection_binding_id": "binding-1",
        }

    @pytest.mark.parametrize(
        ("error", "message"),
        [
            (LLMBadRequestError(), "No Embedding Model available"),
            (ProviderTokenNotInitError("provider setup"), "provider setup"),
        ],
    )
    def test_configure_embedding_model_for_high_quality_wraps_model_errors(
        self, error, message, dataset_session: Session
    ):
        class FakeAccount:
            pass

        current_user = FakeAccount()
        current_user.current_tenant_id = "tenant-1"
        with (
            patch("services.dataset_service.Account", FakeAccount),
            patch("services.dataset_service.current_user", current_user),
            patch("services.dataset_service.ModelManager") as model_manager_cls,
        ):
            model_manager_cls.for_tenant.return_value.get_model_instance.side_effect = error

            with pytest.raises(ValueError, match=message):
                DatasetService._configure_embedding_model_for_high_quality(
                    {"embedding_model_provider": "provider", "embedding_model": "embedding-model"},
                    {},
                    dataset_session,
                )

    def test_handle_embedding_model_update_when_technique_unchanged_preserves_existing_settings(
        self, dataset_session: Session
    ):
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(
            embedding_model_provider="provider",
            embedding_model="embedding-model",
        )
        filtered_data: dict[str, object] = {}
        with patch.object(DatasetService, "_preserve_existing_embedding_settings") as preserve_settings:
            result = DatasetService._handle_embedding_model_update_when_technique_unchanged(
                dataset,
                {},
                filtered_data,
                dataset_session,
            )

        assert result is None
        preserve_settings.assert_called_once_with(dataset, filtered_data)

    def test_handle_embedding_model_update_when_technique_unchanged_updates_when_model_is_provided(
        self, dataset_session: Session
    ):
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(
            embedding_model_provider="provider",
            embedding_model="embedding-model",
        )
        with patch.object(DatasetService, "_update_embedding_model_settings", return_value="update") as update_settings:
            result = DatasetService._handle_embedding_model_update_when_technique_unchanged(
                dataset,
                {"embedding_model_provider": "provider-two", "embedding_model": "embedding-model-two"},
                {},
                dataset_session,
            )

        assert result == "update"
        update_settings.assert_called_once_with(
            dataset,
            {"embedding_model_provider": "provider-two", "embedding_model": "embedding-model-two"},
            {},
            dataset_session,
        )

    def test_preserve_existing_embedding_settings_keeps_current_binding(self):
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(
            embedding_model_provider="provider",
            embedding_model="embedding-model",
            collection_binding_id="binding-1",
        )
        filtered_data = {"embedding_model_provider": "", "embedding_model": ""}

        DatasetService._preserve_existing_embedding_settings(dataset, filtered_data)

        assert filtered_data == {
            "embedding_model_provider": "provider",
            "embedding_model": "embedding-model",
            "collection_binding_id": "binding-1",
        }

    def test_preserve_existing_embedding_settings_removes_empty_placeholders_without_existing_values(self):
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(
            embedding_model_provider=None,
            embedding_model=None,
            collection_binding_id=None,
        )
        filtered_data = {"embedding_model_provider": "", "embedding_model": ""}

        DatasetService._preserve_existing_embedding_settings(dataset, filtered_data)

        assert filtered_data == {}

    def test_update_embedding_model_settings_returns_update_for_changed_values(self, dataset_session: Session):
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(
            embedding_model_provider="provider",
            embedding_model="embedding-model",
        )
        with patch.object(DatasetService, "_apply_new_embedding_settings") as apply_settings:
            result = DatasetService._update_embedding_model_settings(
                dataset,
                {"embedding_model_provider": "provider-two", "embedding_model": "embedding-model-two"},
                {},
                dataset_session,
            )

        assert result == "update"
        apply_settings.assert_called_once_with(
            dataset,
            {"embedding_model_provider": "provider-two", "embedding_model": "embedding-model-two"},
            {},
            dataset_session,
        )

    def test_update_embedding_model_settings_returns_none_for_unchanged_values(self, dataset_session: Session):
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(
            embedding_model_provider="provider",
            embedding_model="embedding-model",
        )
        result = DatasetService._update_embedding_model_settings(
            dataset,
            {"embedding_model_provider": "provider", "embedding_model": "embedding-model"},
            {},
            dataset_session,
        )

        assert result is None

    def test_update_embedding_model_settings_wraps_bad_request_errors(self, dataset_session: Session):
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(
            embedding_model_provider="provider",
            embedding_model="embedding-model",
        )
        with patch.object(DatasetService, "_apply_new_embedding_settings", side_effect=LLMBadRequestError()):
            with pytest.raises(ValueError, match="No Embedding Model available"):
                DatasetService._update_embedding_model_settings(
                    dataset,
                    {"embedding_model_provider": "provider-two", "embedding_model": "embedding-model-two"},
                    {},
                    dataset_session,
                )

    def test_apply_new_embedding_settings_updates_binding_for_new_model(self, dataset_session: Session):
        class FakeAccount:
            pass

        current_user = FakeAccount()
        current_user.current_tenant_id = "tenant-1"
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(collection_binding_id="binding-1")
        filtered_data: dict[str, object] = {}
        with (
            patch("services.dataset_service.Account", FakeAccount),
            patch("services.dataset_service.current_user", current_user),
            patch("services.dataset_service.ModelManager") as model_manager_cls,
            patch(
                "services.dataset_service.DatasetCollectionBindingService.get_dataset_collection_binding",
                return_value=SimpleNamespace(id="binding-2"),
            ),
        ):
            model_manager_cls.for_tenant.return_value.get_model_instance.return_value = SimpleNamespace(
                provider="provider-two",
                model_name="embedding-model-two",
            )

            DatasetService._apply_new_embedding_settings(
                dataset,
                {"embedding_model_provider": "provider-two", "embedding_model": "embedding-model-two"},
                filtered_data,
                dataset_session,
            )

        assert filtered_data == {
            "embedding_model": "embedding-model-two",
            "embedding_model_provider": "provider-two",
            "collection_binding_id": "binding-2",
        }

    def test_apply_new_embedding_settings_preserves_existing_values_when_provider_token_is_missing(
        self, dataset_session: Session
    ):
        class FakeAccount:
            pass

        current_user = FakeAccount()
        current_user.current_tenant_id = "tenant-1"
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(
            embedding_model_provider="provider",
            embedding_model="embedding-model",
            collection_binding_id="binding-1",
        )
        filtered_data: dict[str, object] = {}
        with (
            patch("services.dataset_service.Account", FakeAccount),
            patch("services.dataset_service.current_user", current_user),
            patch("services.dataset_service.ModelManager") as model_manager_cls,
        ):
            model_manager_cls.for_tenant.return_value.get_model_instance.side_effect = ProviderTokenNotInitError(
                "token missing"
            )

            DatasetService._apply_new_embedding_settings(
                dataset,
                {"embedding_model_provider": "provider-two", "embedding_model": "embedding-model-two"},
                filtered_data,
                dataset_session,
            )

        assert filtered_data == {
            "embedding_model_provider": "provider",
            "embedding_model": "embedding-model",
            "collection_binding_id": "binding-1",
        }

    @pytest.mark.parametrize(
        ("summary_index_setting", "expected"),
        [
            (None, False),
            ({"enable": False}, False),
            ({"enable": True, "model_name": "old-model", "model_provider_name": "provider"}, False),
            ({"enable": True, "model_name": "new-model", "model_provider_name": "provider-two"}, True),
        ],
    )
    def test_check_summary_index_setting_model_changed(self, summary_index_setting, expected):
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(
            dataset_id="dataset-1",
            summary_index_setting={"enable": True, "model_name": "old-model", "model_provider_name": "provider"},
        )

        result = DatasetService._check_summary_index_setting_model_changed(
            dataset,
            {"summary_index_setting": summary_index_setting} if summary_index_setting is not None else {},
        )

        assert result is expected


class TestDatasetServiceRagPipelineSettings:
    """Unit tests for rag-pipeline dataset setting updates."""

    def test_update_rag_pipeline_dataset_settings_requires_current_tenant(self, dataset_session: Session):
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(dataset_id="dataset-1")
        knowledge_configuration = _make_knowledge_configuration()

        with patch("services.dataset_service.current_user", SimpleNamespace(current_tenant_id=None)):
            with pytest.raises(ValueError, match="Current user or current tenant not found"):
                DatasetService.update_rag_pipeline_dataset_settings(
                    dataset, knowledge_configuration, session=dataset_session
                )

    def test_update_rag_pipeline_dataset_settings_without_published_high_quality_updates_embedding_settings(
        self, dataset_session: Session
    ):
        tenant_id = str(uuid4())
        dataset = _persist_dataset(dataset_session, dataset_id=str(uuid4()), tenant_id=tenant_id)
        knowledge_configuration = _make_knowledge_configuration(summary_index_setting={"enable": True})
        embedding_model = SimpleNamespace(provider="provider", model_name="embedding-model")

        with (
            patch("services.dataset_service.current_user", SimpleNamespace(current_tenant_id=tenant_id)),
            patch("services.dataset_service.ModelManager") as model_manager_cls,
            patch.object(DatasetService, "check_is_multimodal_model", return_value=True) as check_multimodal,
            patch(
                "services.dataset_service.DatasetCollectionBindingService.get_dataset_collection_binding",
                return_value=SimpleNamespace(id="binding-1"),
            ),
        ):
            model_manager_cls.for_tenant.return_value.get_model_instance.return_value = embedding_model

            DatasetService.update_rag_pipeline_dataset_settings(
                dataset, knowledge_configuration, session=dataset_session
            )

        assert dataset.chunk_structure == "paragraph"
        assert dataset.indexing_technique == "high_quality"
        assert dataset.embedding_model == "embedding-model"
        assert dataset.embedding_model_provider == "provider"
        assert dataset.collection_binding_id == "binding-1"
        assert dataset.is_multimodal is True
        assert dataset.retrieval_model == knowledge_configuration.retrieval_model.model_dump()
        assert dataset.summary_index_setting == {"enable": True}
        check_multimodal.assert_called_once_with(tenant_id, "provider", "embedding-model")

    def test_update_rag_pipeline_dataset_settings_without_published_economy_updates_keyword_number(
        self, dataset_session: Session
    ):
        tenant_id = str(uuid4())
        dataset = _persist_dataset(dataset_session, dataset_id=str(uuid4()), tenant_id=tenant_id)
        knowledge_configuration = _make_knowledge_configuration(
            indexing_technique="economy",
            embedding_model_provider="",
            embedding_model="",
            keyword_number=12,
        )

        with patch("services.dataset_service.current_user", SimpleNamespace(current_tenant_id=tenant_id)):
            DatasetService.update_rag_pipeline_dataset_settings(
                dataset, knowledge_configuration, session=dataset_session
            )

        assert dataset.indexing_technique == "economy"
        assert dataset.keyword_number == 12
        assert dataset.retrieval_model == knowledge_configuration.retrieval_model.model_dump()

    def test_update_rag_pipeline_dataset_settings_with_published_rejects_chunk_structure_changes(
        self, dataset_session: Session
    ):
        tenant_id = str(uuid4())
        dataset = _persist_dataset(dataset_session, dataset_id=str(uuid4()), tenant_id=tenant_id)
        dataset.chunk_structure = "paragraph"
        knowledge_configuration = _make_knowledge_configuration(chunk_structure="sentence")

        with patch("services.dataset_service.current_user", SimpleNamespace(current_tenant_id=tenant_id)):
            with pytest.raises(ValueError, match="Chunk structure is not allowed to be updated"):
                DatasetService.update_rag_pipeline_dataset_settings(
                    dataset, knowledge_configuration, has_published=True, session=dataset_session
                )

    def test_update_rag_pipeline_dataset_settings_with_published_rejects_switch_to_economy(
        self, dataset_session: Session
    ):
        tenant_id = str(uuid4())
        dataset = _persist_dataset(dataset_session, dataset_id=str(uuid4()), tenant_id=tenant_id)
        dataset.chunk_structure = "paragraph"
        dataset.indexing_technique = "high_quality"
        knowledge_configuration = _make_knowledge_configuration(
            indexing_technique="economy",
            embedding_model_provider="",
            embedding_model="",
        )

        with patch("services.dataset_service.current_user", SimpleNamespace(current_tenant_id=tenant_id)):
            with pytest.raises(
                ValueError,
                match="Knowledge base indexing technique is not allowed to be updated to economy",
            ):
                DatasetService.update_rag_pipeline_dataset_settings(
                    dataset, knowledge_configuration, has_published=True, session=dataset_session
                )

    def test_update_rag_pipeline_dataset_settings_with_published_adds_high_quality_index(
        self, dataset_session: Session
    ):
        tenant_id = str(uuid4())
        dataset = _persist_dataset(dataset_session, dataset_id=str(uuid4()), tenant_id=tenant_id)
        dataset.chunk_structure = "paragraph"
        dataset.indexing_technique = "economy"
        knowledge_configuration = _make_knowledge_configuration()
        embedding_model = SimpleNamespace(provider="provider", model_name="embedding-model")

        with (
            patch("services.dataset_service.current_user", SimpleNamespace(current_tenant_id=tenant_id)),
            patch("services.dataset_service.ModelManager") as model_manager_cls,
            patch.object(DatasetService, "check_is_multimodal_model", return_value=False),
            patch(
                "services.dataset_service.DatasetCollectionBindingService.get_dataset_collection_binding",
                return_value=SimpleNamespace(id="binding-1"),
            ),
            patch("services.dataset_service.deal_dataset_index_update_task") as update_task,
        ):
            model_manager_cls.for_tenant.return_value.get_model_instance.return_value = embedding_model

            DatasetService.update_rag_pipeline_dataset_settings(
                dataset, knowledge_configuration, has_published=True, session=dataset_session
            )

        assert dataset.indexing_technique == "high_quality"
        assert dataset.embedding_model == "embedding-model"
        assert dataset.embedding_model_provider == "provider"
        assert dataset.collection_binding_id == "binding-1"
        assert dataset.is_multimodal is False
        assert dataset.retrieval_model == knowledge_configuration.retrieval_model.model_dump()
        update_task.delay.assert_called_once_with(dataset.id, "add")

    def test_update_rag_pipeline_dataset_settings_with_published_updates_changed_embedding_model(
        self, dataset_session: Session
    ):
        tenant_id = str(uuid4())
        dataset = _persist_dataset(dataset_session, dataset_id=str(uuid4()), tenant_id=tenant_id)
        dataset.chunk_structure = "paragraph"
        dataset.indexing_technique = "high_quality"
        dataset.embedding_model_provider = "provider"
        dataset.embedding_model = "embedding-model"
        knowledge_configuration = _make_knowledge_configuration(
            embedding_model_provider="provider-two",
            embedding_model="embedding-model-two",
            summary_index_setting={"enable": True},
        )

        with (
            patch("services.dataset_service.current_user", SimpleNamespace(current_tenant_id=tenant_id)),
            patch("services.dataset_service.ModelManager") as model_manager_cls,
            patch.object(DatasetService, "check_is_multimodal_model", return_value=True),
            patch(
                "services.dataset_service.DatasetCollectionBindingService.get_dataset_collection_binding",
                return_value=SimpleNamespace(id="binding-2"),
            ),
            patch("services.dataset_service.deal_dataset_index_update_task") as update_task,
        ):
            model_manager_cls.for_tenant.return_value.get_model_instance.return_value = SimpleNamespace(
                provider="provider-two",
                model_name="embedding-model-two",
            )

            DatasetService.update_rag_pipeline_dataset_settings(
                dataset, knowledge_configuration, has_published=True, session=dataset_session
            )

        assert dataset.embedding_model_provider == "provider-two"
        assert dataset.embedding_model == "embedding-model-two"
        assert dataset.collection_binding_id == "binding-2"
        assert dataset.is_multimodal is True
        assert dataset.summary_index_setting == {"enable": True}
        update_task.delay.assert_called_once_with(dataset.id, "update")

    def test_update_rag_pipeline_dataset_settings_with_published_skips_embedding_update_when_token_is_missing(
        self, dataset_session: Session
    ):
        tenant_id = str(uuid4())
        dataset = _persist_dataset(dataset_session, dataset_id=str(uuid4()), tenant_id=tenant_id)
        dataset.chunk_structure = "paragraph"
        dataset.indexing_technique = "high_quality"
        dataset.embedding_model_provider = "provider"
        dataset.embedding_model = "embedding-model"
        knowledge_configuration = _make_knowledge_configuration(
            embedding_model_provider="provider-two",
            embedding_model="embedding-model-two",
        )

        with (
            patch("services.dataset_service.current_user", SimpleNamespace(current_tenant_id=tenant_id)),
            patch("services.dataset_service.ModelManager") as model_manager_cls,
            patch("services.dataset_service.deal_dataset_index_update_task") as update_task,
        ):
            model_manager_cls.for_tenant.return_value.get_model_instance.side_effect = ProviderTokenNotInitError(
                "token missing"
            )

            DatasetService.update_rag_pipeline_dataset_settings(
                dataset, knowledge_configuration, has_published=True, session=dataset_session
            )

        assert dataset.embedding_model_provider == "provider"
        assert dataset.embedding_model == "embedding-model"
        assert dataset.retrieval_model == knowledge_configuration.retrieval_model.model_dump()
        update_task.delay.assert_called_once_with(dataset.id, "update")

    def test_update_rag_pipeline_dataset_settings_with_published_updates_economy_keyword_number(
        self, dataset_session: Session
    ):
        tenant_id = str(uuid4())
        dataset = _persist_dataset(dataset_session, dataset_id=str(uuid4()), tenant_id=tenant_id)
        dataset.chunk_structure = "paragraph"
        dataset.indexing_technique = "economy"
        dataset.keyword_number = 5
        knowledge_configuration = _make_knowledge_configuration(
            indexing_technique="economy",
            embedding_model_provider="",
            embedding_model="",
            keyword_number=9,
        )

        with (
            patch("services.dataset_service.current_user", SimpleNamespace(current_tenant_id=tenant_id)),
            patch("services.dataset_service.deal_dataset_index_update_task") as update_task,
        ):
            DatasetService.update_rag_pipeline_dataset_settings(
                dataset, knowledge_configuration, has_published=True, session=dataset_session
            )

        assert dataset.keyword_number == 9
        assert dataset.retrieval_model == knowledge_configuration.retrieval_model.model_dump()
        update_task.delay.assert_not_called()


class TestDatasetServicePermissionsAndLifecycle:
    """Unit tests for dataset permissions, deletion, and metadata helpers."""

    def test_check_dataset_operator_permission_validates_required_arguments(self, dataset_session: Session):
        with pytest.raises(ValueError, match="Dataset not found"):
            DatasetService.check_dataset_operator_permission(
                user=SimpleNamespace(id="user-1"),
                dataset=None,
                session=dataset_session,
            )

        with pytest.raises(ValueError, match="User not found"):
            DatasetService.check_dataset_operator_permission(
                user=None,
                dataset=SimpleNamespace(id="dataset-1"),
                session=dataset_session,
            )


class TestDatasetCollectionBindingService:
    """Unit tests for dataset collection binding lookups and creation."""


class TestDatasetPermissionService:
    """Unit tests for dataset partial-member management helpers."""

    def test_dataset_permission_constraint_failure_rolls_back(self, dataset_session: Session):
        permission = DatasetPermission(
            dataset_id=str(uuid4()),
            account_id=str(uuid4()),
            tenant_id=None,  # type: ignore[arg-type]
        )
        dataset_session.add(permission)

        with pytest.raises(IntegrityError):
            dataset_session.commit()

        dataset_session.rollback()
        assert dataset_session.scalars(select(DatasetPermission)).all() == []

    def test_update_partial_member_list_rolls_back_on_exception(self, dataset_session: Session):
        def fail_flush(_session, _flush_context, _instances):
            raise RuntimeError("boom")

        event.listen(dataset_session, "before_flush", fail_flush)
        try:
            with pytest.raises(RuntimeError, match="boom"):
                DatasetPermissionService.update_partial_member_list(
                    str(uuid4()),
                    str(uuid4()),
                    [{"user_id": str(uuid4())}],
                    dataset_session,
                )
        finally:
            event.remove(dataset_session, "before_flush", fail_flush)

        assert not dataset_session.in_transaction()

    def test_check_permission_requires_dataset_editor(self, dataset_session: Session):
        user = SimpleNamespace(is_dataset_editor=False, is_dataset_operator=False)
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock()
        with pytest.raises(NoPermissionError, match="does not have permission"):
            DatasetPermissionService.check_permission(user, dataset, "all_team", [], session=dataset_session)

    def test_check_permission_prevents_dataset_operator_from_changing_permission_mode(self, dataset_session: Session):
        user = SimpleNamespace(is_dataset_editor=True, is_dataset_operator=True)
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(permission="all_team")
        with pytest.raises(NoPermissionError, match="cannot change the dataset permissions"):
            DatasetPermissionService.check_permission(user, dataset, "only_me", [], session=dataset_session)

    def test_check_permission_requires_partial_member_list_for_partial_members_mode(self, dataset_session: Session):
        user = SimpleNamespace(is_dataset_editor=True, is_dataset_operator=True)
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(permission="partial_members")
        with pytest.raises(ValueError, match="Partial member list is required"):
            DatasetPermissionService.check_permission(user, dataset, "partial_members", [], session=dataset_session)

    def test_check_permission_rejects_dataset_operator_member_list_changes(self, dataset_session: Session):
        user = SimpleNamespace(is_dataset_editor=True, is_dataset_operator=True)
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(
            dataset_id="dataset-1", permission="partial_members"
        )
        with patch.object(DatasetPermissionService, "get_dataset_partial_member_list", return_value=["user-1"]):
            with pytest.raises(ValueError, match="cannot change the dataset permissions"):
                DatasetPermissionService.check_permission(
                    user, dataset, "partial_members", [{"user_id": "user-2"}], session=dataset_session
                )

    def test_check_permission_allows_dataset_operator_when_member_list_is_unchanged(self, dataset_session: Session):
        user = SimpleNamespace(is_dataset_editor=True, is_dataset_operator=True)
        dataset = DatasetServiceUnitDataFactory.create_dataset_mock(
            dataset_id="dataset-1", permission="partial_members"
        )
        with patch.object(DatasetPermissionService, "get_dataset_partial_member_list", return_value=["user-1"]):
            DatasetPermissionService.check_permission(
                user, dataset, "partial_members", [{"user_id": "user-1"}], session=dataset_session
            )

    def test_clear_partial_member_list_rolls_back_on_exception(self, dataset_session: Session):
        engine = dataset_session.get_bind()

        def fail_delete(_conn, _cursor, statement, _parameters, _context, _executemany):
            if statement.lstrip().startswith("DELETE FROM dataset_permissions"):
                raise RuntimeError("boom")

        event.listen(engine, "before_cursor_execute", fail_delete)
        try:
            with pytest.raises(RuntimeError, match="boom"):
                DatasetPermissionService.clear_partial_member_list(str(uuid4()), dataset_session)
        finally:
            event.remove(engine, "before_cursor_execute", fail_delete)

        assert not dataset_session.in_transaction()

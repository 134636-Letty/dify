import inspect
import json
from collections.abc import Iterator
from dataclasses import dataclass
from unittest.mock import PropertyMock, patch

import pytest
from flask import Flask
from sqlalchemy import Engine, event, func, select
from sqlalchemy.orm import Session, sessionmaker
from werkzeug.exceptions import Forbidden, NotFound

from controllers.console import console_ns
from controllers.console.datasets.error import DatasetNameDuplicateError
from controllers.console.datasets.external import (
    BedrockRetrievalApi,
    ExternalApiTemplateApi,
    ExternalApiTemplateListApi,
    ExternalApiUseCheckApi,
    ExternalDatasetCreateApi,
    ExternalKnowledgeHitTestingApi,
)
from extensions import ext_database
from models import dataset as dataset_models
from models.account import Account, TenantAccountRole
from models.dataset import Dataset, ExternalKnowledgeApis, ExternalKnowledgeBindings
from services.external_knowledge_service import ExternalDatasetService
from services.hit_testing_service import HitTestingService
from services.knowledge_service import ExternalDatasetTestService


@dataclass(frozen=True)
class Database:
    """Expose the real test session through the subset of ``db`` used by models and pagination."""

    session: Session


@pytest.fixture
def app() -> Flask:
    app = Flask("test_external_dataset")
    app.config["TESTING"] = True
    return app


@pytest.fixture
def current_user() -> Account:
    user = Account(name="Test User", email="user-1@example.com")
    user.id = "user-1"
    user.role = TenantAccountRole.EDITOR
    return user


@pytest.fixture
def database(sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[Database]:
    tables = [
        Dataset.__table__,
        ExternalKnowledgeApis.__table__,
        ExternalKnowledgeBindings.__table__,
    ]
    Dataset.metadata.create_all(sqlite_engine, tables=tables)
    session_factory = sessionmaker(bind=sqlite_engine, expire_on_commit=False)
    with session_factory() as session:
        database = Database(session=session)
        # Pagination imports ext_database.db lazily, while Dataset model properties
        # retain the module-level binding imported when the model was loaded.
        monkeypatch.setattr(ext_database, "db", database)
        monkeypatch.setattr(dataset_models, "db", database)
        yield database


def _add_external_api(
    session: Session,
    *,
    api_id: str,
    tenant_id: str = "tenant-1",
    name: str | None = None,
) -> ExternalKnowledgeApis:
    external_api = ExternalKnowledgeApis(
        name=name or f"External API {api_id}",
        description=f"Description for {api_id}",
        tenant_id=tenant_id,
        settings=json.dumps(
            {
                "endpoint": f"https://external.example.com/{api_id}",
                "api_key": "secret",
                "headers": {},
                "timeout": 30,
            }
        ),
        created_by="user-1",
        updated_by="user-1",
    )
    external_api.id = api_id
    session.add(external_api)
    session.commit()
    return external_api


def _add_dataset(
    session: Session,
    *,
    dataset_id: str,
    tenant_id: str = "tenant-1",
    name: str | None = None,
) -> Dataset:
    dataset = Dataset(
        id=dataset_id,
        tenant_id=tenant_id,
        name=name or f"Dataset {dataset_id}",
        description="External support articles",
        provider="external",
        retrieval_model={"top_k": 4, "score_threshold": 0.5, "score_threshold_enabled": True},
        created_by="user-1",
        maintainer="user-1",
    )
    session.add(dataset)
    session.commit()
    return dataset


def _bind_dataset(
    session: Session,
    *,
    binding_id: str,
    dataset_id: str,
    api_id: str,
    tenant_id: str = "tenant-1",
) -> ExternalKnowledgeBindings:
    binding = ExternalKnowledgeBindings(
        tenant_id=tenant_id,
        external_knowledge_api_id=api_id,
        dataset_id=dataset_id,
        external_knowledge_id=f"knowledge-{binding_id}",
        created_by="user-1",
    )
    binding.id = binding_id
    session.add(binding)
    session.commit()
    return binding


class TestExternalApiTemplateListApi:
    def test_get_returns_only_matching_tenant_rows(self, app: Flask, database: Database):
        session = database.session
        visible = _add_external_api(session, api_id="api-visible", name="Vector Search")
        dataset = _add_dataset(session, dataset_id="dataset-visible")
        _bind_dataset(session, binding_id="binding-visible", dataset_id=dataset.id, api_id=visible.id)
        _add_external_api(session, api_id="api-other", tenant_id="tenant-2", name="Vector Other")
        _add_external_api(session, api_id="api-nonmatch", name="Keyword Search")

        method = inspect.unwrap(ExternalApiTemplateListApi().get)
        with app.test_request_context("/?page=1&limit=10&keyword=vector"):
            response, status = method(ExternalApiTemplateListApi(), "tenant-1")

        assert status == 200
        assert response["total"] == 1
        assert [item["id"] for item in response["data"]] == [visible.id]
        assert response["data"][0]["dataset_bindings"] == [{"id": dataset.id, "name": dataset.name}]

    def test_get_empty_page(self, app: Flask, database: Database):
        method = inspect.unwrap(ExternalApiTemplateListApi().get)
        with app.test_request_context("/?page=1&limit=10"):
            response, status = method(ExternalApiTemplateListApi(), "tenant-1")

        assert status == 200
        assert response == {"data": [], "has_more": False, "limit": 10, "total": 0, "page": 1}

    def test_post_persists_template(self, app: Flask, current_user: Account, database: Database):
        payload = {
            "name": "Vendor Search",
            "settings": {"endpoint": "https://external.example.com/search", "api_key": "secret"},
        }
        method = inspect.unwrap(ExternalApiTemplateListApi().post)

        with (
            app.test_request_context("/", json=payload),
            patch.object(type(console_ns), "payload", new_callable=PropertyMock, return_value=payload),
            patch.object(ExternalDatasetService, "check_endpoint_and_api_key") as endpoint_check,
        ):
            response, status = method(ExternalApiTemplateListApi(), database.session, "tenant-1", current_user)

        assert status == 201
        persisted = database.session.get(ExternalKnowledgeApis, response["id"])
        assert persisted is not None
        assert (persisted.tenant_id, persisted.name, persisted.settings_dict) == (
            "tenant-1",
            "Vendor Search",
            payload["settings"],
        )
        endpoint_check.assert_called_once_with(payload["settings"])

    def test_post_rolls_back_failed_insert(self, app: Flask, current_user: Account, database: Database):
        payload = {
            "name": "Broken API",
            "settings": {"endpoint": "https://external.example.com/search", "api_key": "secret"},
        }
        method = inspect.unwrap(ExternalApiTemplateListApi().post)

        def fail_insert(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("forced insert failure")

        event.listen(ExternalKnowledgeApis, "before_insert", fail_insert)
        try:
            with (
                app.test_request_context("/", json=payload),
                patch.object(type(console_ns), "payload", new_callable=PropertyMock, return_value=payload),
                patch.object(ExternalDatasetService, "check_endpoint_and_api_key"),
                pytest.raises(RuntimeError, match="forced insert failure"),
            ):
                method(ExternalApiTemplateListApi(), database.session, "tenant-1", current_user)
        finally:
            event.remove(ExternalKnowledgeApis, "before_insert", fail_insert)
            database.session.rollback()

        assert database.session.scalar(select(func.count(ExternalKnowledgeApis.id))) == 0

    def test_post_forbidden_uses_real_session(self, app: Flask, current_user: Account, database: Database):
        current_user.role = TenantAccountRole.NORMAL
        payload = {
            "name": "Vendor Search",
            "settings": {"endpoint": "https://external.example.com/search", "api_key": "secret"},
        }
        method = inspect.unwrap(ExternalApiTemplateListApi().post)

        with (
            app.test_request_context("/", json=payload),
            patch.object(type(console_ns), "payload", new_callable=PropertyMock, return_value=payload),
            pytest.raises(Forbidden),
        ):
            method(ExternalApiTemplateListApi(), database.session, "tenant-1", current_user)

        assert database.session.scalar(select(func.count(ExternalKnowledgeApis.id))) == 0


class TestExternalApiTemplateApi:
    def test_get_reads_persisted_template(self, app: Flask, database: Database):
        external_api = _add_external_api(database.session, api_id="api-detail")
        method = inspect.unwrap(ExternalApiTemplateApi().get)

        with app.test_request_context("/"):
            response, status = method(ExternalApiTemplateApi(), database.session, "tenant-1", external_api.id)

        assert status == 200
        assert response["id"] == external_api.id
        assert response["settings"] == external_api.settings_dict

    def test_get_does_not_cross_tenant_boundary(self, app: Flask, database: Database):
        external_api = _add_external_api(database.session, api_id="api-private", tenant_id="tenant-2")
        method = inspect.unwrap(ExternalApiTemplateApi().get)

        with app.test_request_context("/"), pytest.raises(ValueError, match="api template not found"):
            method(ExternalApiTemplateApi(), database.session, "tenant-1", external_api.id)

    def test_patch_updates_only_scoped_template(self, app: Flask, current_user: Account, database: Database):
        external_api = _add_external_api(database.session, api_id="api-update")
        other = _add_external_api(database.session, api_id="api-other", tenant_id="tenant-2", name="Other")
        payload = {
            "name": "Updated API",
            "settings": {"endpoint": "https://external.example.com/updated", "api_key": "new-secret"},
        }
        method = inspect.unwrap(ExternalApiTemplateApi().patch)

        with (
            app.test_request_context("/", json=payload),
            patch.object(type(console_ns), "payload", new_callable=PropertyMock, return_value=payload),
        ):
            response, status = method(
                ExternalApiTemplateApi(), database.session, "tenant-1", current_user, external_api.id
            )

        assert status == 200
        assert response["name"] == "Updated API"
        database.session.refresh(external_api)
        database.session.refresh(other)
        assert external_api.settings_dict == payload["settings"]
        assert (other.name, other.tenant_id) == ("Other", "tenant-2")

    def test_delete_removes_persisted_template(self, app: Flask, current_user: Account, database: Database):
        external_api = _add_external_api(database.session, api_id="api-delete")
        method = inspect.unwrap(ExternalApiTemplateApi().delete)

        with app.test_request_context("/"):
            response, status = method(
                ExternalApiTemplateApi(), database.session, "tenant-1", current_user, external_api.id
            )

        assert (response, status) == ("", 204)
        assert database.session.get(ExternalKnowledgeApis, external_api.id) is None

    def test_delete_forbidden_preserves_template(self, app: Flask, current_user: Account, database: Database):
        current_user.role = TenantAccountRole.NORMAL
        external_api = _add_external_api(database.session, api_id="api-keep")
        method = inspect.unwrap(ExternalApiTemplateApi().delete)

        with app.test_request_context("/"), pytest.raises(Forbidden):
            method(ExternalApiTemplateApi(), database.session, "tenant-1", current_user, external_api.id)

        assert database.session.get(ExternalKnowledgeApis, external_api.id) is not None


class TestExternalApiUseCheckApi:
    def test_get_counts_only_current_tenant_bindings(self, app: Flask, database: Database):
        session = database.session
        external_api = _add_external_api(session, api_id="api-use")
        first = _add_dataset(session, dataset_id="dataset-first")
        second = _add_dataset(session, dataset_id="dataset-second")
        other = _add_dataset(session, dataset_id="dataset-other", tenant_id="tenant-2")
        _bind_dataset(session, binding_id="binding-first", dataset_id=first.id, api_id=external_api.id)
        _bind_dataset(session, binding_id="binding-second", dataset_id=second.id, api_id=external_api.id)
        _bind_dataset(
            session,
            binding_id="binding-other",
            dataset_id=other.id,
            api_id=external_api.id,
            tenant_id="tenant-2",
        )
        method = inspect.unwrap(ExternalApiUseCheckApi().get)

        with app.test_request_context("/"):
            response, status = method(ExternalApiUseCheckApi(), session, "tenant-1", external_api.id)

        assert status == 200
        assert response == {"is_using": True, "count": 2}

    def test_get_returns_empty_usage(self, app: Flask, database: Database):
        external_api = _add_external_api(database.session, api_id="api-unused")
        method = inspect.unwrap(ExternalApiUseCheckApi().get)

        with app.test_request_context("/"):
            response, status = method(ExternalApiUseCheckApi(), database.session, "tenant-1", external_api.id)

        assert status == 200
        assert response == {"is_using": False, "count": 0}


class TestExternalDatasetCreateApi:
    def test_create_persists_dataset_and_binding(self, app: Flask, current_user: Account, database: Database):
        external_api = _add_external_api(database.session, api_id="api-create")
        payload = {
            "external_knowledge_api_id": external_api.id,
            "external_knowledge_id": "knowledge-1",
            "name": "Support knowledge",
            "description": "External support articles",
            "external_retrieval_model": {
                "top_k": 4,
                "score_threshold": 0.5,
                "score_threshold_enabled": True,
            },
        }
        method = inspect.unwrap(ExternalDatasetCreateApi().post)

        with (
            app.test_request_context("/", json=payload),
            patch.object(type(console_ns), "payload", new_callable=PropertyMock, return_value=payload),
            patch(
                "controllers.console.datasets.external.enterprise_rbac_service.RBACService.DatasetPermissions.batch_get",
                return_value={},
            ),
            patch("controllers.console.datasets.external.DatasetDetailResponse.model_validate") as validate_response,
        ):
            persisted_response = {"id": "captured", "permission_keys": []}
            validate_response.return_value.model_dump.return_value = persisted_response
            response, status = method(ExternalDatasetCreateApi(), database.session, "tenant-1", current_user)

        assert status == 201
        dataset = database.session.scalar(select(Dataset).where(Dataset.name == "Support knowledge"))
        assert dataset is not None
        binding = database.session.scalar(
            select(ExternalKnowledgeBindings).where(ExternalKnowledgeBindings.dataset_id == dataset.id)
        )
        assert binding is not None
        assert (
            dataset.tenant_id,
            dataset.provider,
            binding.external_knowledge_api_id,
            binding.external_knowledge_id,
        ) == (
            "tenant-1",
            "external",
            external_api.id,
            "knowledge-1",
        )
        assert response["permission_keys"] == []
        validate_response.assert_called_once_with(dataset)

    def test_create_duplicate_name_preserves_existing_state(
        self, app: Flask, current_user: Account, database: Database
    ):
        external_api = _add_external_api(database.session, api_id="api-duplicate")
        existing = _add_dataset(database.session, dataset_id="dataset-existing", name="Duplicate")
        payload = {
            "external_knowledge_api_id": external_api.id,
            "external_knowledge_id": "knowledge-1",
            "name": "Duplicate",
        }
        method = inspect.unwrap(ExternalDatasetCreateApi().post)

        with (
            app.test_request_context("/", json=payload),
            patch.object(type(console_ns), "payload", new_callable=PropertyMock, return_value=payload),
            pytest.raises(DatasetNameDuplicateError),
        ):
            method(ExternalDatasetCreateApi(), database.session, "tenant-1", current_user)

        assert database.session.scalars(select(Dataset).where(Dataset.name == "Duplicate")).all() == [existing]
        assert database.session.scalar(select(func.count(ExternalKnowledgeBindings.id))) == 0

    def test_create_forbidden_uses_real_session(self, app: Flask, current_user: Account, database: Database):
        current_user.role = TenantAccountRole.NORMAL
        payload = {
            "external_knowledge_api_id": "api-1",
            "external_knowledge_id": "knowledge-1",
            "name": "Forbidden",
        }
        method = inspect.unwrap(ExternalDatasetCreateApi().post)

        with (
            app.test_request_context("/", json=payload),
            patch.object(type(console_ns), "payload", new_callable=PropertyMock, return_value=payload),
            pytest.raises(Forbidden),
        ):
            method(ExternalDatasetCreateApi(), database.session, "tenant-1", current_user)

        assert database.session.scalar(select(func.count(Dataset.id))) == 0


class TestExternalKnowledgeHitTestingApi:
    def test_hit_testing_dataset_not_found(self, app: Flask, current_user: Account, database: Database):
        method = inspect.unwrap(ExternalKnowledgeHitTestingApi().post)

        with app.test_request_context("/"), pytest.raises(NotFound):
            method(ExternalKnowledgeHitTestingApi(), database.session, current_user, "missing-dataset")

    def test_hit_testing_uses_persisted_dataset(self, app: Flask, current_user: Account, database: Database):
        dataset = _add_dataset(database.session, dataset_id="dataset-hit-test")
        payload = {
            "query": "hello",
            "external_retrieval_model": {"top_k": 3, "score_threshold": 0.25},
            "metadata_filtering_conditions": {"logical_operator": "and", "conditions": []},
        }
        retrieval_response = {
            "query": {"content": "hello"},
            "records": [{"content": "answer", "title": "doc", "score": 0.9, "metadata": {"page": 2}}],
        }
        method = inspect.unwrap(ExternalKnowledgeHitTestingApi().post)

        with (
            app.test_request_context("/", json=payload),
            patch.object(type(console_ns), "payload", new_callable=PropertyMock, return_value=payload),
            patch("controllers.console.datasets.external.DatasetService.check_dataset_permission") as permission_check,
            patch.object(HitTestingService, "hit_testing_args_check") as args_check,
            patch.object(HitTestingService, "external_retrieve", return_value=retrieval_response) as retrieve,
        ):
            response = method(ExternalKnowledgeHitTestingApi(), database.session, current_user, dataset.id)

        assert response == retrieval_response
        permission_check.assert_called_once_with(dataset, current_user, database.session)
        args_check.assert_called_once_with(payload)
        assert retrieve.call_args.kwargs["session"] is database.session
        assert retrieve.call_args.kwargs["dataset"] is dataset


class TestBedrockRetrievalApi:
    def test_bedrock_retrieval_keeps_provider_boundary_mocked(self, app: Flask):
        payload = {
            "retrieval_setting": {"top_k": 5, "score_threshold": 0.72},
            "query": "hello bedrock",
            "knowledge_id": "knowledge-base-1",
        }
        retrieval_response = {
            "records": [{"metadata": {"source": "bedrock"}, "score": 0.8, "title": "doc", "content": "answer"}]
        }
        method = inspect.unwrap(BedrockRetrievalApi().post)

        with (
            app.test_request_context("/", json=payload),
            patch.object(type(console_ns), "payload", new_callable=PropertyMock, return_value=payload),
            patch.object(
                ExternalDatasetTestService, "knowledge_retrieval", return_value=retrieval_response
            ) as retrieve,
        ):
            response, status = method()

        assert status == 200
        assert response == retrieval_response
        retrieval_setting, query, knowledge_id = retrieve.call_args.args
        assert retrieval_setting.model_dump() == payload["retrieval_setting"]
        assert (query, knowledge_id) == ("hello bedrock", "knowledge-base-1")

    def test_bedrock_retrieval_propagates_invalid_setting(self, app: Flask):
        payload = {"retrieval_setting": {}, "query": "test", "knowledge_id": "k-1"}
        method = inspect.unwrap(BedrockRetrievalApi().post)

        with (
            app.test_request_context("/", json=payload),
            patch.object(type(console_ns), "payload", new_callable=PropertyMock, return_value=payload),
            patch.object(ExternalDatasetTestService, "knowledge_retrieval", side_effect=ValueError("Invalid settings")),
            pytest.raises(ValueError, match="Invalid settings"),
        ):
            method()

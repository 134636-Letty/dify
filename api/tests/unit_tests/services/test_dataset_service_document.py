"""SQLite-backed unit tests for DocumentService behaviors in dataset_service."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy import Engine, event, func, select
from sqlalchemy.orm import Session, sessionmaker

from core.rag.entities import PreProcessingRule, Rule, Segmentation
from core.rag.index_processor.constant.built_in_field import BuiltInField
from core.rag.index_processor.constant.index_type import IndexStructureType
from extensions.storage.storage_type import StorageType
from models import dataset as dataset_model_module
from models.account import Account, Tenant
from models.dataset import Dataset, DatasetProcessRule, Document
from models.enums import CreatorUserRole, DataSourceType, DocumentCreatedFrom, DocumentDocType, IndexingStatus
from models.model import UploadFile
from services import dataset_service as dataset_service_module
from services.dataset_ref_service import DatasetRef
from services.dataset_service import DatasetService, DocumentService
from services.entities.knowledge_entities.knowledge_entities import (
    DataSource,
    FileInfo,
    InfoList,
    KnowledgeConfig,
    ProcessRule,
)
from services.errors.document import DocumentIndexingError
from services.errors.file import FileNotExistsError


@dataclass(frozen=True)
class Database:
    """Typed database binding exposing the real SQLite session used by services."""

    engine: Engine
    session: Session


@pytest.fixture
def database(sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[Database]:
    Dataset.metadata.create_all(
        sqlite_engine,
        tables=[Dataset.__table__, Document.__table__, DatasetProcessRule.__table__, UploadFile.__table__],
    )
    session_factory = sessionmaker(bind=sqlite_engine, expire_on_commit=False)
    with session_factory() as session:
        database = Database(engine=sqlite_engine, session=session)
        monkeypatch.setattr(dataset_service_module, "db", database)
        monkeypatch.setattr(dataset_model_module, "db", database)
        yield database


@pytest.fixture
def account(monkeypatch: pytest.MonkeyPatch) -> Account:
    account = Account(name="Owner", email="owner@example.com")
    account.id = "user-1"
    tenant = Tenant(name="Tenant")
    tenant.id = "tenant-1"
    account._current_tenant = tenant
    monkeypatch.setattr(dataset_service_module, "current_user", account)
    return account


def _dataset(
    session: Session,
    *,
    dataset_id: str = "dataset-1",
    tenant_id: str = "tenant-1",
    built_in_field_enabled: bool = False,
    indexing_technique: str | None = "economy",
) -> Dataset:
    dataset = Dataset(
        id=dataset_id,
        tenant_id=tenant_id,
        name=f"Dataset {dataset_id}",
        description="Document service fixture",
        provider="vendor",
        data_source_type=DataSourceType.UPLOAD_FILE,
        indexing_technique=indexing_technique,
        created_by="user-1",
        maintainer="user-1",
        built_in_field_enabled=built_in_field_enabled,
    )
    session.add(dataset)
    session.commit()
    return dataset


def _document(
    session: Session,
    *,
    document_id: str = "doc-1",
    dataset_id: str = "dataset-1",
    tenant_id: str = "tenant-1",
    name: str = "Document 1",
    enabled: bool = True,
    archived: bool = False,
    indexing_status: IndexingStatus = IndexingStatus.COMPLETED,
    data_source_type: DataSourceType = DataSourceType.UPLOAD_FILE,
    data_source_info: dict[str, object] | None = None,
) -> Document:
    document = Document(
        id=document_id,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        position=1,
        data_source_type=data_source_type,
        data_source_info=json.dumps(data_source_info or {}),
        dataset_process_rule_id=None,
        batch="batch-1",
        name=name,
        created_from=DocumentCreatedFrom.API,
        created_by="user-1",
        word_count=10,
        tokens=5,
        indexing_status=indexing_status,
        completed_at=datetime.now(UTC).replace(tzinfo=None) if indexing_status == IndexingStatus.COMPLETED else None,
        enabled=enabled,
        archived=archived,
        doc_type=DocumentDocType.BOOK,
        doc_metadata=None,
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
        doc_language="English",
        need_summary=False,
        is_paused=False,
    )
    session.add(document)
    session.commit()
    return document


def _upload_file(
    session: Session,
    *,
    file_id: str = "file-1",
    tenant_id: str = "tenant-1",
    name: str = "file.txt",
) -> UploadFile:
    upload_file = UploadFile(
        tenant_id=tenant_id,
        storage_type=StorageType.LOCAL,
        key=f"files/{file_id}",
        name=name,
        size=12,
        extension="txt",
        mime_type="text/plain",
        created_by_role=CreatorUserRole.ACCOUNT,
        created_by="user-1",
        created_at=datetime.now(UTC).replace(tzinfo=None),
        used=False,
    )
    upload_file.id = file_id
    session.add(upload_file)
    session.commit()
    return upload_file


def _upload_config(file_ids: list[str], *, process_rule: ProcessRule | None = None) -> KnowledgeConfig:
    return KnowledgeConfig(
        indexing_technique="economy",
        data_source=DataSource(
            info_list=InfoList(
                data_source_type="upload_file",
                file_info_list=FileInfo(file_ids=file_ids),
            )
        ),
        process_rule=process_rule,
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
        doc_language="English",
    )


class TestDisplayStatus:
    @pytest.mark.parametrize(
        ("raw_status", "expected"),
        [("enabled", "available"), ("AVAILABLE", "available"), ("paused", "paused"), ("unknown", None)],
    )
    def test_normalize_display_status(self, raw_status, expected):
        assert DocumentService.normalize_display_status(raw_status) == expected

    def test_apply_display_status_filter_executes_real_statement(self, database: Database):
        _document(database.session, document_id="available")
        _document(database.session, document_id="archived", archived=True)
        stmt = DocumentService.apply_display_status_filter(select(Document), "available")
        assert [document.id for document in database.session.scalars(stmt)] == ["available"]


class TestMutations:
    def test_delete_documents_is_tenant_and_dataset_scoped(self, database: Database):
        _dataset(database.session)
        visible = _document(database.session, document_id="visible")
        _document(database.session, document_id="other-tenant", tenant_id="tenant-2")
        _document(database.session, document_id="other-dataset", dataset_id="dataset-2")

        with patch("services.dataset_service.batch_clean_document_task") as clean_task:
            DocumentService.delete_documents(
                DatasetRef(tenant_id="tenant-1", dataset_id="dataset-1"),
                ["visible", "other-tenant", "other-dataset"],
                IndexStructureType.PARAGRAPH_INDEX,
                database.session,
            )

        assert database.session.get(Document, visible.id) is None
        assert database.session.get(Document, "other-tenant") is not None
        assert database.session.get(Document, "other-dataset") is not None
        clean_task.delay.assert_called_once_with(["visible"], "dataset-1", IndexStructureType.PARAGRAPH_INDEX, [])

    def test_rename_missing_dataset_and_document(self, database: Database, account: Account):
        with pytest.raises(ValueError, match="Dataset not found"):
            DocumentService.rename_document("missing", "doc-1", "New", database.session)

        _dataset(database.session)
        with pytest.raises(ValueError, match="Document not found"):
            DocumentService.rename_document("dataset-1", "missing", "New", database.session)

    def test_rename_rejects_cross_tenant_document(self, database: Database, account: Account):
        _dataset(database.session)
        _document(database.session, tenant_id="tenant-2")
        with pytest.raises(ValueError, match="No permission"):
            DocumentService.rename_document("dataset-1", "doc-1", "New", database.session)

    def test_rename_updates_document_metadata_and_upload_file(self, database: Database, account: Account):
        _dataset(database.session, built_in_field_enabled=True)
        upload_file = _upload_file(database.session)
        document = _document(database.session, data_source_info={"upload_file_id": upload_file.id})
        document.doc_metadata = {BuiltInField.document_name: "Old"}
        database.session.commit()

        result = DocumentService.rename_document("dataset-1", "doc-1", "New Name", database.session)

        database.session.refresh(upload_file)
        assert result.name == "New Name"
        assert result.doc_metadata[BuiltInField.document_name] == "New Name"
        assert upload_file.name == "New Name"

    def test_recover_requires_paused_document(self, database: Database):
        document = _document(database.session)
        with pytest.raises(DocumentIndexingError):
            DocumentService.recover_document(document, database.session)

    def test_sync_website_document_persists_and_dispatches(self, database: Database):
        document = _document(
            database.session,
            data_source_type=DataSourceType.WEBSITE_CRAWL,
            data_source_info={"mode": "crawl"},
        )
        with (
            patch("services.dataset_service.redis_client") as redis,
            patch("services.dataset_service.sync_website_document_indexing_task") as task,
        ):
            redis.get.return_value = None
            DocumentService.sync_website_document("dataset-1", document, database.session)

        database.session.refresh(document)
        assert document.indexing_status == IndexingStatus.WAITING
        assert document.data_source_info_dict["mode"] == "scrape"
        task.delay.assert_called_once_with("dataset-1", "doc-1")


class TestSaveAndLookup:
    def test_save_without_dataset_persists_dataset(self, database: Database, account: Account):
        config = KnowledgeConfig(
            indexing_technique="high_quality",
            data_source=DataSource(
                info_list=InfoList(data_source_type="upload_file", file_info_list=FileInfo(file_ids=["file-1"]))
            ),
            embedding_model="embedding-model",
            embedding_model_provider="provider",
            is_multimodal=True,
        )
        created_document = Document(
            id="created-doc",
            tenant_id="tenant-1",
            dataset_id="placeholder",
            position=1,
            data_source_type=DataSourceType.UPLOAD_FILE,
            data_source_info="{}",
            batch="batch-1",
            name="VeryLongDocumentNameForDataset.txt",
            created_from=DocumentCreatedFrom.API,
            created_by="user-1",
        )

        def save_document(dataset, _config, _account, *, session):
            created_document.dataset_id = dataset.id
            session.add(created_document)
            session.flush()
            return [created_document], "batch-1"

        with (
            patch("services.dataset_service.FeatureService.get_features") as features,
            patch(
                "services.dataset_service.DatasetCollectionBindingService.get_dataset_collection_binding",
                return_value=SimpleNamespace(id="binding-1"),
            ),
            patch.object(DocumentService, "save_document_with_dataset_id", side_effect=save_document),
        ):
            features.return_value.billing.enabled = False
            dataset, documents, batch = DocumentService.save_document_without_dataset_id(
                "tenant-1", config, account, database.session
            )

        assert database.session.get(Dataset, dataset.id) is dataset
        assert database.session.get(Document, "created-doc") is created_document
        assert documents == [created_document]
        assert batch == "batch-1"
        assert dataset.collection_binding_id == "binding-1"
        assert dataset.retrieval_model["top_k"] == 4
        assert dataset.name == "VeryLongDocumentNa..."

    def test_save_upload_documents_uses_persisted_files_and_tenant_scope(self, database: Database, account: Account):
        dataset = _dataset(database.session)
        _upload_file(database.session, file_id="file-1", name="first.txt")
        _upload_file(database.session, file_id="file-2", name="other.txt", tenant_id="tenant-2")
        process_rule = DatasetProcessRule(
            dataset_id=dataset.id,
            mode="automatic",
            rules=json.dumps(DatasetProcessRule.AUTOMATIC_RULES),
            created_by=account.id,
        )
        database.session.add(process_rule)
        database.session.commit()

        with (
            patch("services.dataset_service.FeatureService.get_features") as features,
            patch("services.dataset_service.redis_client") as redis,
            patch("services.dataset_service.DocumentIndexingTaskProxy") as indexing_task,
        ):
            features.return_value.billing.enabled = False
            redis.lock.return_value.__enter__.return_value = None
            documents, _ = DocumentService.save_document_with_dataset_id(
                dataset,
                _upload_config(["file-1"]),
                account,
                dataset_process_rule=process_rule,
                session=database.session,
            )

        assert len(documents) == 1
        assert documents[0].name == "first.txt"
        assert database.session.get(Document, documents[0].id) is documents[0]
        indexing_task.assert_called_once_with("tenant-1", "dataset-1", [documents[0].id])

    def test_save_upload_documents_rejects_missing_or_cross_tenant_file(self, database: Database, account: Account):
        dataset = _dataset(database.session)
        _upload_file(database.session, file_id="file-1", tenant_id="tenant-2")
        process_rule = DatasetProcessRule(dataset_id=dataset.id, mode="automatic", rules="{}", created_by=account.id)
        database.session.add(process_rule)
        database.session.commit()

        with (
            patch("services.dataset_service.FeatureService.get_features") as features,
            patch("services.dataset_service.redis_client") as redis,
        ):
            features.return_value.billing.enabled = False
            redis.lock.return_value.__enter__.return_value = None
            with pytest.raises(FileNotExistsError, match="One or more files not found"):
                DocumentService.save_document_with_dataset_id(
                    dataset,
                    _upload_config(["file-1"]),
                    account,
                    dataset_process_rule=process_rule,
                    session=database.session,
                )

        database.session.rollback()
        assert database.session.scalar(select(func.count(Document.id))) == 0

    def test_get_tenant_documents_count_is_tenant_scoped(self, database: Database, account: Account):
        _document(database.session, document_id="tenant-one")
        _document(database.session, document_id="tenant-two", tenant_id="tenant-2")
        assert DocumentService.get_tenant_documents_count(session=database.session) == 1


class TestBatchStatusTransactions:
    def test_batch_update_persists_enabled_state(self, database: Database):
        dataset = _dataset(database.session)
        document = _document(database.session, enabled=False)
        with (
            patch("services.dataset_service.redis_client") as redis,
            patch("services.dataset_service.add_document_to_index_task") as task,
        ):
            redis.get.return_value = None
            DocumentService.batch_update_document_status(
                dataset, [document.id], "enable", SimpleNamespace(id="user-1"), database.session
            )

        database.session.refresh(document)
        assert document.enabled is True
        task.delay.assert_called_once_with(document.id)

    def test_batch_update_rolls_back_flush_failure(self, database: Database):
        dataset = _dataset(database.session)
        document = _document(database.session, enabled=False)

        def fail_flush(_session, _flush_context, _instances) -> None:
            raise RuntimeError("forced flush failure")

        event.listen(database.session, "before_flush", fail_flush)
        try:
            with patch("services.dataset_service.redis_client") as redis:
                redis.get.return_value = None
                with pytest.raises(RuntimeError, match="forced flush failure"):
                    DocumentService.batch_update_document_status(
                        dataset, [document.id], "enable", SimpleNamespace(id="user-1"), database.session
                    )
        finally:
            event.remove(database.session, "before_flush", fail_flush)

        database.session.refresh(document)
        assert document.enabled is False
        assert not database.session.in_transaction() or database.session.is_active


class TestValidation:
    def test_document_create_requires_source_or_rule(self):
        with pytest.raises(ValueError, match="Data source or Process rule is required"):
            DocumentService.document_create_args_validate(KnowledgeConfig(indexing_technique="economy"))

    @pytest.mark.parametrize(
        ("source_type", "message"),
        [
            ("upload_file", "File source info is required"),
            ("notion_import", "Notion source info is required"),
            ("website_crawl", "Website source info is required"),
        ],
    )
    def test_data_source_requires_source_specific_info(self, source_type: str, message: str):
        config = KnowledgeConfig(
            indexing_technique="economy",
            data_source=DataSource(info_list=InfoList(data_source_type=source_type)),
        )
        with pytest.raises(ValueError, match=message):
            DocumentService.data_source_args_validate(config)

    def test_automatic_process_rule_clears_rules(self):
        config = KnowledgeConfig(
            indexing_technique="economy",
            process_rule=ProcessRule(
                mode="automatic",
                rules=Rule(
                    pre_processing_rules=[PreProcessingRule(id="remove_stopwords", enabled=True)],
                    segmentation=Segmentation(separator="\n", max_tokens=128),
                ),
            ),
        )
        DocumentService.process_rule_args_validate(config)
        assert config.process_rule.rules is None

    def test_custom_process_rule_deduplicates_rules(self):
        config = KnowledgeConfig(
            indexing_technique="economy",
            process_rule=ProcessRule(
                mode="custom",
                rules=Rule(
                    pre_processing_rules=[
                        PreProcessingRule(id="remove_stopwords", enabled=True),
                        PreProcessingRule(id="remove_stopwords", enabled=False),
                    ],
                    segmentation=Segmentation(separator="\n", max_tokens=128),
                ),
            ),
        )
        DocumentService.process_rule_args_validate(config)
        assert config.process_rule.rules.pre_processing_rules == [
            PreProcessingRule(id="remove_stopwords", enabled=False)
        ]

    def test_estimate_args_validate_sets_automatic_rules_empty(self):
        args = {
            "info_list": {"data_source_type": "upload_file"},
            "process_rule": {"mode": "automatic", "rules": {"ignored": True}},
        }
        DocumentService.estimate_args_validate(args)
        assert args["process_rule"]["rules"] == {}

    def test_update_requires_upload_file_info(self, database: Database, account: Account):
        dataset = _dataset(database.session)
        document = _document(database.session)
        config = KnowledgeConfig(
            original_document_id=document.id,
            indexing_technique="economy",
            data_source=DataSource(info_list=InfoList(data_source_type="upload_file")),
        )
        with patch.object(DatasetService, "check_dataset_model_setting"):
            with pytest.raises(ValueError, match="No file info list found"):
                DocumentService.update_document_with_dataset_id(dataset, config, account, session=database.session)

    def test_update_rejects_missing_upload_file(self, database: Database, account: Account):
        dataset = _dataset(database.session)
        document = _document(database.session)
        config = _upload_config(["missing"])
        config.original_document_id = document.id
        with patch.object(DatasetService, "check_dataset_model_setting"):
            with pytest.raises(FileNotExistsError):
                DocumentService.update_document_with_dataset_id(dataset, config, account, session=database.session)

    @pytest.mark.parametrize(
        ("source_type", "message"),
        [("notion_import", "No notion info list found"), ("website_crawl", "No website info list found")],
    )
    def test_save_requires_source_payload(self, database: Database, account: Account, source_type: str, message: str):
        dataset = _dataset(database.session)
        config = KnowledgeConfig(
            indexing_technique="economy",
            data_source=DataSource(info_list=InfoList(data_source_type=source_type)),
            doc_form=IndexStructureType.PARAGRAPH_INDEX,
            doc_language="English",
        )
        rule = DatasetProcessRule(dataset_id=dataset.id, mode="automatic", rules="{}", created_by=account.id)
        database.session.add(rule)
        database.session.commit()
        with (
            patch("services.dataset_service.FeatureService.get_features") as features,
            patch("services.dataset_service.redis_client") as redis,
        ):
            features.return_value.billing.enabled = False
            redis.lock.return_value.__enter__.return_value = None
            with pytest.raises(ValueError, match=message):
                DocumentService.save_document_with_dataset_id(
                    dataset, config, account, dataset_process_rule=rule, session=database.session
                )

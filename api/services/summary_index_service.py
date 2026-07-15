"""Summary index service for generating and managing document segment summaries."""

import logging
import time
import uuid
from datetime import UTC, datetime
from typing import TypedDict, cast

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from core.db.session_factory import session_factory
from core.model_manager import ModelManager
from core.rag.datasource.vdb.vector_factory import Vector
from core.rag.index_processor.constant.doc_type import DocType
from core.rag.index_processor.constant.index_type import IndexTechniqueType
from core.rag.index_processor.index_processor_base import SummaryIndexSettingDict
from core.rag.models.document import Document
from graphon.model_runtime.entities.llm_entities import LLMUsage
from graphon.model_runtime.entities.model_entities import ModelType
from libs import helper
from models.dataset import Dataset, DocumentSegment, DocumentSegmentSummary
from models.dataset import Document as DatasetDocument
from models.enums import SummaryStatus

logger = logging.getLogger(__name__)


class SummaryIndexConflictError(RuntimeError):
    pass


class SummaryEntryDict(TypedDict):
    segment_id: str
    segment_position: int
    status: str
    summary_preview: str | None
    error: str | None
    created_at: int | None
    updated_at: int | None


class DocumentSummaryStatusDetailDict(TypedDict):
    total_segments: int
    summary_status: dict[str, int]
    summaries: list[SummaryEntryDict]


class SummaryIndexService:
    """Service for generating and managing summary indexes."""

    @staticmethod
    def _lock_segment_rows(session: Session, dataset_id: str, segment_ids: list[str] | None) -> None:
        if segment_ids == []:
            return

        stmt = select(DocumentSegment.id).where(DocumentSegment.dataset_id == dataset_id)
        if segment_ids is not None:
            stmt = stmt.where(DocumentSegment.id.in_(sorted(set(segment_ids))))
        session.execute(stmt.order_by(DocumentSegment.id).with_for_update()).all()

    @staticmethod
    def _create_cleanup_vector(dataset: Dataset, session: Session) -> Vector | None:
        if dataset.indexing_technique != IndexTechniqueType.HIGH_QUALITY:
            return None
        return Vector(dataset, session=session)

    @staticmethod
    def _get_summary_record(
        session: Session,
        segment_id: str,
        dataset_id: str,
        *,
        for_update: bool = False,
    ) -> DocumentSegmentSummary | None:
        stmt = (
            select(DocumentSegmentSummary)
            .where(
                DocumentSegmentSummary.chunk_id == segment_id,
                DocumentSegmentSummary.dataset_id == dataset_id,
            )
            .limit(1)
        )
        if for_update:
            stmt = stmt.with_for_update()
        return session.scalar(stmt)

    @staticmethod
    def _summary_allowed_segment_ids(session: Session, dataset_id: str, segment_ids: list[str]) -> set[str]:
        return set(
            session.scalars(
                select(DocumentSegment.id)
                .join(DatasetDocument, DatasetDocument.id == DocumentSegment.document_id)
                .where(
                    DocumentSegment.id.in_(segment_ids),
                    DocumentSegment.dataset_id == dataset_id,
                    DocumentSegment.enabled.is_(True),
                    DocumentSegment.status == "completed",
                    DatasetDocument.dataset_id == dataset_id,
                    DatasetDocument.enabled.is_(True),
                    DatasetDocument.archived.is_(False),
                    DatasetDocument.indexing_status == "completed",
                )
            ).all()
        )

    @staticmethod
    def _segment_allows_summary(session: Session, dataset_id: str, segment_id: str) -> bool:
        stmt = (
            select(DocumentSegment.id)
            .join(DatasetDocument, DatasetDocument.id == DocumentSegment.document_id)
            .where(
                DocumentSegment.id == segment_id,
                DocumentSegment.dataset_id == dataset_id,
                DocumentSegment.enabled.is_(True),
                DocumentSegment.status == "completed",
                DatasetDocument.dataset_id == dataset_id,
                DatasetDocument.enabled.is_(True),
                DatasetDocument.archived.is_(False),
                DatasetDocument.indexing_status == "completed",
            )
        )
        return session.execute(stmt).scalar_one_or_none() is not None

    @staticmethod
    def _reenable_summary_record(summary_record: DocumentSegmentSummary) -> None:
        if summary_record.enabled:
            return

        summary_record.enabled = True
        summary_record.disabled_at = None
        summary_record.disabled_by = None

    @staticmethod
    def _mark_summary_generation_started(
        segment: DocumentSegment,
        dataset: Dataset,
    ) -> DocumentSegmentSummary:
        with session_factory.create_session() as session:
            SummaryIndexService._lock_segment_rows(session, dataset.id, [segment.id])
            if not SummaryIndexService._segment_allows_summary(session, dataset.id, segment.id):
                raise SummaryIndexConflictError(f"Segment {segment.id} no longer accepts summaries")
            summary_record = SummaryIndexService._get_summary_record(
                session,
                segment.id,
                dataset.id,
                for_update=True,
            )

            if not summary_record:
                logger.warning("Summary record not found for segment %s, creating one", segment.id)
                summary_record = DocumentSegmentSummary(
                    dataset_id=dataset.id,
                    document_id=segment.document_id,
                    chunk_id=segment.id,
                    summary_content="",
                    status=SummaryStatus.GENERATING,
                    enabled=True,
                )
            else:
                SummaryIndexService._reenable_summary_record(summary_record)

            summary_record.status = SummaryStatus.GENERATING
            summary_record.error = None
            session.add(summary_record)
            session.commit()
            return summary_record

    @staticmethod
    def _save_summary_content(
        segment: DocumentSegment,
        dataset: Dataset,
        summary_content: str,
        *,
        summary_record_id: str | None = None,
        status: SummaryStatus = SummaryStatus.GENERATING,
    ) -> DocumentSegmentSummary:
        with session_factory.create_session() as session:
            SummaryIndexService._lock_segment_rows(session, dataset.id, [segment.id])
            if not SummaryIndexService._segment_allows_summary(session, dataset.id, segment.id):
                raise SummaryIndexConflictError(f"Segment {segment.id} no longer accepts summaries")
            if summary_record_id:
                summary_record = session.get(DocumentSegmentSummary, summary_record_id, with_for_update=True)
                if not summary_record:
                    raise SummaryIndexConflictError(
                        f"Summary {summary_record_id} was deleted while segment {segment.id} was being generated"
                    )
            else:
                summary_record = SummaryIndexService._get_summary_record(
                    session,
                    segment.id,
                    dataset.id,
                    for_update=True,
                )

            if not summary_record:
                summary_record = DocumentSegmentSummary(
                    dataset_id=dataset.id,
                    document_id=segment.document_id,
                    chunk_id=segment.id,
                    summary_content=summary_content,
                    status=status,
                    enabled=True,
                )
            else:
                summary_record.summary_content = summary_content
                summary_record.status = status
                summary_record.error = None
                SummaryIndexService._reenable_summary_record(summary_record)

            session.add(summary_record)
            session.commit()
            return summary_record

    @staticmethod
    def _enable_summary_record(
        summary_record_id: str,
        segment_id: str,
        dataset_id: str,
    ) -> bool:
        with session_factory.create_session() as session:
            SummaryIndexService._lock_segment_rows(session, dataset_id, [segment_id])
            summary_record = session.get(DocumentSegmentSummary, summary_record_id, with_for_update=True)
            if (
                not summary_record
                or summary_record.dataset_id != dataset_id
                or summary_record.chunk_id != segment_id
                or not SummaryIndexService._segment_allows_summary(session, dataset_id, segment_id)
            ):
                return False

            SummaryIndexService._reenable_summary_record(summary_record)
            session.add(summary_record)
            session.commit()
            return True

    @staticmethod
    def generate_summary_for_segment(
        segment: DocumentSegment,
        dataset: Dataset,
        summary_index_setting: SummaryIndexSettingDict,
    ) -> tuple[str, LLMUsage]:
        """
        Generate summary for a single segment.

        Args:
            segment: DocumentSegment to generate summary for
            dataset: Dataset containing the segment
            summary_index_setting: Summary index configuration

        Returns:
            Tuple of (summary_content, llm_usage) where llm_usage is LLMUsage object

        Raises:
            ValueError: If summary_index_setting is invalid or generation fails
        """
        # Reuse the existing generate_summary method from ParagraphIndexProcessor
        # Use lazy import to avoid circular import
        from core.rag.index_processor.processor.paragraph_index_processor import ParagraphIndexProcessor

        with session_factory.create_session() as session:
            document_language = session.scalar(
                select(DatasetDocument.doc_language).where(
                    DatasetDocument.id == segment.document_id,
                    DatasetDocument.dataset_id == dataset.id,
                )
            )

        summary_content, usage = ParagraphIndexProcessor.generate_summary(
            tenant_id=dataset.tenant_id,
            text=segment.content,
            summary_index_setting=summary_index_setting,
            segment_id=segment.id,
            document_language=document_language,
        )

        if not summary_content:
            raise ValueError("Generated summary is empty")

        return summary_content, usage

    @staticmethod
    def create_summary_record(
        segment: DocumentSegment,
        dataset: Dataset,
        summary_content: str,
        status: SummaryStatus = SummaryStatus.GENERATING,
        *,
        session: Session | None = None,
    ) -> DocumentSegmentSummary:
        """
        Create or update a DocumentSegmentSummary record.
        If a summary record already exists for this segment, it will be updated instead of creating a new one.
        The write is committed before returning so follow-up vectorization can run without a dirty DB session.

        Args:
            segment: DocumentSegment to create summary for
            dataset: Dataset containing the segment
            summary_content: Generated summary content
            status: Summary status (default: SummaryStatus.GENERATING)

        Returns:
            Created or updated DocumentSegmentSummary instance
        """
        return SummaryIndexService._save_summary_content(
            segment=segment,
            dataset=dataset,
            summary_content=summary_content,
            status=status,
        )

    @staticmethod
    def vectorize_summary(
        summary_record: DocumentSegmentSummary,
        segment: DocumentSegment,
        dataset: Dataset,
        session: Session | None = None,
    ) -> None:
        """
        Vectorize summary and store in vector database.

        Args:
            summary_record: DocumentSegmentSummary record
            segment: Original DocumentSegment
            dataset: Dataset containing the segment
        """
        if dataset.indexing_technique != IndexTechniqueType.HIGH_QUALITY:
            logger.warning(
                "Summary vectorization skipped for dataset %s: indexing_technique is not high_quality",
                dataset.id,
            )
            return
        original_session = session
        if session is not None:
            session.commit()
            session = None

        # Get summary_record_id for later session queries
        summary_record_id = summary_record.id
        logger.debug(
            "Starting vectorization for segment %s, summary_record_id=%s, using_provided_session=%s",
            segment.id,
            summary_record_id,
            original_session is not None,
        )

        old_summary_node_id = summary_record.summary_index_node_id
        summary_index_node_id = str(uuid.uuid4())
        expected_enabled = summary_record.enabled

        # Always regenerate hash (in case summary content changed)
        summary_content = summary_record.summary_content
        if not summary_content or not summary_content.strip():
            raise ValueError(f"Summary content is empty for segment {segment.id}, cannot vectorize")
        summary_hash = helper.generate_text_hash(summary_content)

        # Calculate embedding tokens for summary (for logging and statistics)
        embedding_tokens = 0
        try:
            model_manager = ModelManager.for_tenant(tenant_id=dataset.tenant_id)
            embedding_model = model_manager.get_model_instance(
                tenant_id=dataset.tenant_id,
                provider=dataset.embedding_model_provider,
                model_type=ModelType.TEXT_EMBEDDING,
                model=dataset.embedding_model,
            )
            if embedding_model:
                tokens_list = embedding_model.get_text_embedding_num_tokens([summary_content])
                raw_embedding_tokens = tokens_list[0] if tokens_list else 0
                embedding_tokens = raw_embedding_tokens if isinstance(raw_embedding_tokens, int) else 0
        except Exception as e:
            logger.warning("Failed to calculate embedding tokens for summary: %s", str(e))

        # Create document with summary content and metadata
        summary_document = Document(
            page_content=summary_content,
            metadata={
                "doc_id": summary_index_node_id,
                "doc_hash": summary_hash,
                "dataset_id": dataset.id,
                "document_id": segment.document_id,
                "original_chunk_id": segment.id,  # Key: link to original chunk
                "doc_type": DocType.TEXT,
                "is_summary": True,  # Identifier for summary documents
            },
        )

        # Vectorize and store with retry mechanism for connection errors
        max_retries = 3
        retry_delay = 2.0
        vector: Vector | None = None

        for attempt in range(max_retries):
            try:
                logger.debug(
                    "Attempting to vectorize summary for segment %s (attempt %s/%s)",
                    segment.id,
                    attempt + 1,
                    max_retries,
                )
                vector = Vector(dataset)
                vector.add_texts([summary_document], duplicate_check=False)
                logger.debug(
                    "Successfully added summary vector to database for segment %s (attempt %s/%s)",
                    segment.id,
                    attempt + 1,
                    max_retries,
                )

                # Log embedding token usage
                if embedding_tokens > 0:
                    logger.info(
                        "Summary embedding for segment %s used %s tokens",
                        segment.id,
                        embedding_tokens,
                    )

                # Success - update summary record with index node info
                # Use provided session if available, otherwise create a new one
                use_provided_session = session is not None
                if not use_provided_session:
                    logger.debug("Creating new session for vectorization of segment %s", segment.id)
                    session_context = session_factory.create_session()
                    session = session_context.__enter__()
                else:
                    logger.debug("Using provided session for vectorization of segment %s", segment.id)
                    session_context = None  # Don't use context manager for provided session

                # At this point, session is guaranteed to be not None
                # Type narrowing: session is definitely not None after the if/else above
                if session is None:
                    raise RuntimeError("Session should not be None at this point")

                try:
                    SummaryIndexService._lock_segment_rows(session, dataset.id, [segment.id])
                    # Declare summary_record_in_session variable
                    summary_record_in_session: DocumentSegmentSummary | None

                    # If using provided session, merge the summary_record into it
                    if use_provided_session:
                        # Merge the summary_record into the provided session
                        logger.debug(
                            "Merging summary_record (id=%s) into provided session for segment %s",
                            summary_record_id,
                            segment.id,
                        )
                        summary_record_in_session = session.merge(summary_record)
                        logger.debug(
                            "Successfully merged summary_record for segment %s, merged_id=%s",
                            segment.id,
                            summary_record_in_session.id,
                        )
                    else:
                        # Query the summary record in the new session
                        logger.debug(
                            "Querying summary_record by id=%s for segment %s in new session",
                            summary_record_id,
                            segment.id,
                        )
                        summary_record_in_session = session.scalar(
                            select(DocumentSegmentSummary)
                            .where(DocumentSegmentSummary.id == summary_record_id)
                            .limit(1)
                            .with_for_update()
                        )

                        if not summary_record_in_session:
                            # Record not found - try to find by chunk_id and dataset_id instead
                            logger.debug(
                                "Summary record not found by id=%s, trying chunk_id=%s and dataset_id=%s "
                                "for segment %s",
                                summary_record_id,
                                segment.id,
                                dataset.id,
                                segment.id,
                            )
                            summary_record_in_session = session.scalar(
                                select(DocumentSegmentSummary)
                                .where(
                                    DocumentSegmentSummary.chunk_id == segment.id,
                                    DocumentSegmentSummary.dataset_id == dataset.id,
                                )
                                .limit(1)
                            )

                            if not summary_record_in_session:
                                raise SummaryIndexConflictError(
                                    f"Summary {summary_record_id} was deleted while segment {segment.id} "
                                    "was being vectorized"
                                )
                            else:
                                raise SummaryIndexConflictError(
                                    f"Summary {summary_record_id} was replaced by {summary_record_in_session.id} "
                                    f"while segment {segment.id} was being vectorized"
                                )
                        else:
                            logger.debug(
                                "Found summary_record (id=%s) for segment %s in new session",
                                summary_record_id,
                                segment.id,
                            )

                        # At this point, summary_record_in_session is guaranteed to be not None
                        if summary_record_in_session is None:
                            raise RuntimeError("summary_record_in_session should not be None at this point")

                    if (
                        summary_record_in_session.summary_index_node_id != old_summary_node_id
                        or summary_record_in_session.summary_content != summary_content
                        or summary_record_in_session.enabled != expected_enabled
                        or not SummaryIndexService._segment_allows_summary(session, dataset.id, segment.id)
                    ):
                        raise SummaryIndexConflictError(f"Summary {summary_record_id} vectorization was superseded")

                    if old_summary_node_id:
                        vector.delete_by_ids([old_summary_node_id])

                    # Update all fields including summary_content
                    # Always use the summary_content from the parameter (which is the latest from outer session)
                    # rather than relying on what's in the database, in case outer session hasn't committed yet
                    summary_record_in_session.summary_index_node_id = summary_index_node_id
                    summary_record_in_session.summary_index_node_hash = summary_hash
                    summary_record_in_session.tokens = embedding_tokens  # Save embedding tokens
                    summary_record_in_session.status = SummaryStatus.COMPLETED
                    # Ensure summary_content is preserved (use the latest from summary_record parameter)
                    # This is critical: use the parameter value, not the database value
                    summary_record_in_session.summary_content = summary_content
                    # Explicitly update updated_at to ensure it's refreshed even if other fields haven't changed
                    summary_record_in_session.updated_at = datetime.now(UTC).replace(tzinfo=None)
                    session.add(summary_record_in_session)

                    # Only commit if we created the session ourselves
                    if not use_provided_session:
                        logger.debug("Committing session for segment %s (self-created session)", segment.id)
                        session.commit()
                        logger.debug("Successfully committed session for segment %s", segment.id)
                    else:
                        # When using provided session, flush to ensure changes are written to database
                        # This prevents refresh() from overwriting our changes
                        logger.debug(
                            "Flushing session for segment %s (using provided session, caller will commit)",
                            segment.id,
                        )
                        session.flush()
                        logger.debug("Successfully flushed session for segment %s", segment.id)
                    # If using provided session, let the caller handle commit

                    logger.info(
                        "Successfully vectorized summary for segment %s, index_node_id=%s, index_node_hash=%s, "
                        "tokens=%s, summary_record_id=%s, use_provided_session=%s",
                        segment.id,
                        summary_index_node_id,
                        summary_hash,
                        embedding_tokens,
                        summary_record_in_session.id,
                        use_provided_session,
                    )
                    # Update the original object for consistency
                    summary_record.summary_index_node_id = summary_index_node_id
                    summary_record.summary_index_node_hash = summary_hash
                    summary_record.tokens = embedding_tokens
                    summary_record.status = SummaryStatus.COMPLETED
                    summary_record.summary_content = summary_content
                    if summary_record_in_session.updated_at:
                        summary_record.updated_at = summary_record_in_session.updated_at
                finally:
                    # Only close session if we created it ourselves
                    if not use_provided_session and session_context:
                        session_context.__exit__(None, None, None)
                        session = None
                # Success, exit function
                return

            except (ConnectionError, Exception) as e:
                error_str = str(e).lower()
                # Check if it's a connection-related error that might be transient
                is_connection_error = any(
                    keyword in error_str
                    for keyword in [
                        "connection",
                        "disconnected",
                        "timeout",
                        "network",
                        "could not connect",
                        "server disconnected",
                        "weaviate",
                    ]
                )

                if isinstance(e, SummaryIndexConflictError):
                    if vector is not None:
                        try:
                            vector.delete_by_ids([summary_index_node_id])
                        except Exception:
                            logger.warning(
                                "Failed to compensate summary vector %s", summary_index_node_id, exc_info=True
                            )
                    raise

                if is_connection_error and attempt < max_retries - 1:
                    # Retry for connection errors
                    wait_time = retry_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        "Vectorization attempt %s/%s failed for segment %s (connection error): %s. "
                        "Retrying in %.1f seconds...",
                        attempt + 1,
                        max_retries,
                        segment.id,
                        str(e),
                        wait_time,
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed or non-connection error - log and update status
                    logger.error(
                        "Failed to vectorize summary for segment %s after %s attempts: %s. "
                        "summary_record_id=%s, index_node_id=%s, use_provided_session=%s",
                        segment.id,
                        attempt + 1,
                        str(e),
                        summary_record_id,
                        summary_index_node_id,
                        session is not None,
                        exc_info=True,
                    )
                    # Update error status in session
                    # Use the original_session saved at function start (the function parameter)
                    logger.debug(
                        "Updating error status for segment %s, summary_record_id=%s, has_original_session=%s",
                        segment.id,
                        summary_record_id,
                        original_session is not None,
                    )
                    # Always create a new session for error handling to avoid issues with closed sessions
                    # Even if original_session was provided, we create a new one for safety
                    with session_factory.create_session() as error_session:
                        SummaryIndexService._lock_segment_rows(error_session, dataset.id, [segment.id])
                        # Try to find the record by id first
                        # Note: Using assignment only (no type annotation) to avoid redeclaration error
                        summary_record_in_session = error_session.scalar(
                            select(DocumentSegmentSummary)
                            .where(DocumentSegmentSummary.id == summary_record_id)
                            .limit(1)
                            .with_for_update()
                        )
                        if not summary_record_in_session:
                            # Try to find by chunk_id and dataset_id
                            logger.debug(
                                "Summary record not found by id=%s, trying chunk_id=%s and dataset_id=%s "
                                "for segment %s",
                                summary_record_id,
                                segment.id,
                                dataset.id,
                                segment.id,
                            )
                            summary_record_in_session = error_session.scalar(
                                select(DocumentSegmentSummary)
                                .where(
                                    DocumentSegmentSummary.chunk_id == segment.id,
                                    DocumentSegmentSummary.dataset_id == dataset.id,
                                )
                                .limit(1)
                            )

                        if summary_record_in_session and (
                            summary_record_in_session.summary_index_node_id == old_summary_node_id
                            and summary_record_in_session.summary_content == summary_content
                            and summary_record_in_session.enabled == expected_enabled
                        ):
                            summary_record_in_session.status = SummaryStatus.ERROR
                            summary_record_in_session.error = f"Vectorization failed: {str(e)}"
                            summary_record_in_session.updated_at = datetime.now(UTC).replace(tzinfo=None)
                            error_session.add(summary_record_in_session)
                            error_session.commit()
                            logger.info(
                                "Updated error status in new session for segment %s, record_id=%s",
                                segment.id,
                                summary_record_in_session.id,
                            )
                            # Update the original object for consistency
                            summary_record.status = SummaryStatus.ERROR
                            summary_record.error = summary_record_in_session.error
                            summary_record.updated_at = summary_record_in_session.updated_at
                        else:
                            logger.warning(
                                "Could not update error status: summary record not found for segment %s (id=%s). "
                                "This may indicate a session isolation issue.",
                                segment.id,
                                summary_record_id,
                            )
                    raise

    @staticmethod
    def batch_create_summary_records(
        segments: list[DocumentSegment],
        dataset: Dataset,
        status: SummaryStatus = SummaryStatus.NOT_STARTED,
    ) -> None:
        """
        Batch create summary records for segments with specified status.
        If a record already exists, update its status.

        Args:
            segments: List of DocumentSegment instances
            dataset: Dataset containing the segments
            status: Initial status for the records (default: SummaryStatus.NOT_STARTED)
        """
        segment_ids = [segment.id for segment in segments]
        if not segment_ids:
            return

        with session_factory.create_session() as session:
            SummaryIndexService._lock_segment_rows(session, dataset.id, segment_ids)
            allowed_segment_ids = SummaryIndexService._summary_allowed_segment_ids(session, dataset.id, segment_ids)
            # Query existing summary records
            existing_summaries = session.scalars(
                select(DocumentSegmentSummary).where(
                    DocumentSegmentSummary.chunk_id.in_(segment_ids),
                    DocumentSegmentSummary.dataset_id == dataset.id,
                )
            ).all()
            existing_summary_map = {summary.chunk_id: summary for summary in existing_summaries}

            # Create or update records
            for segment in segments:
                if segment.id not in allowed_segment_ids:
                    continue
                existing_summary = existing_summary_map.get(segment.id)
                if existing_summary:
                    # Update existing record
                    existing_summary.status = status
                    existing_summary.error = None  # Clear any previous errors
                    if not existing_summary.enabled:
                        existing_summary.enabled = True
                        existing_summary.disabled_at = None
                        existing_summary.disabled_by = None
                    session.add(existing_summary)
                else:
                    # Create new record
                    summary_record = DocumentSegmentSummary(
                        dataset_id=dataset.id,
                        document_id=segment.document_id,
                        chunk_id=segment.id,
                        summary_content=None,  # Will be filled later
                        status=status,
                        enabled=True,
                    )
                    session.add(summary_record)

            # Commit the batch created records
            session.commit()

    @staticmethod
    def update_summary_record_error(
        segment: DocumentSegment,
        dataset: Dataset,
        error: str,
    ) -> None:
        """
        Update summary record with error status.

        Args:
            segment: DocumentSegment
            dataset: Dataset containing the segment
            error: Error message
        """
        with session_factory.create_session() as session:
            summary_record = session.scalar(
                select(DocumentSegmentSummary)
                .where(
                    DocumentSegmentSummary.chunk_id == segment.id,
                    DocumentSegmentSummary.dataset_id == dataset.id,
                )
                .limit(1)
            )

            if summary_record:
                summary_record.status = SummaryStatus.ERROR
                summary_record.error = error
                session.add(summary_record)
                session.commit()
            else:
                logger.warning("Summary record not found for segment %s when updating error", segment.id)

    @staticmethod
    def generate_and_vectorize_summary(
        segment: DocumentSegment,
        dataset: Dataset,
        summary_index_setting: SummaryIndexSettingDict,
        *,
        session: Session | None = None,
    ) -> DocumentSegmentSummary:
        """
        Generate summary for a segment and vectorize it.
        Caller state is committed before service-owned LLM/vector work.

        Args:
            segment: DocumentSegment to generate summary for
            dataset: Dataset containing the segment
            summary_index_setting: Summary index configuration

        Returns:
            Created DocumentSegmentSummary instance

        Raises:
            ValueError: If summary generation fails
        """
        if session is not None:
            session.commit()
        summary_record = SummaryIndexService._mark_summary_generation_started(segment, dataset)
        vectorization_started = False

        try:
            summary_content, llm_usage = SummaryIndexService.generate_summary_for_segment(
                segment, dataset, summary_index_setting
            )

            summary_record = SummaryIndexService._save_summary_content(
                segment=segment,
                dataset=dataset,
                summary_content=summary_content,
                summary_record_id=summary_record.id,
                status=SummaryStatus.GENERATING,
            )

            # Log LLM usage for summary generation
            if llm_usage and llm_usage.total_tokens > 0:
                logger.info(
                    "Summary generation for segment %s used %s tokens (prompt: %s, completion: %s)",
                    segment.id,
                    llm_usage.total_tokens,
                    llm_usage.prompt_tokens,
                    llm_usage.completion_tokens,
                )

            vectorization_started = True
            SummaryIndexService.vectorize_summary(summary_record, segment, dataset)
            logger.info("Successfully generated and vectorized summary for segment %s", segment.id)
            return summary_record
        except SummaryIndexConflictError:
            logger.info("Summary generation for segment %s was superseded", segment.id)
            raise
        except Exception as e:
            logger.exception("Failed to generate summary for segment %s", segment.id)
            if not vectorization_started:
                SummaryIndexService.update_summary_record_error(
                    segment=segment,
                    dataset=dataset,
                    error=str(e),
                )
            summary_record.status = SummaryStatus.ERROR
            summary_record.error = str(e)
            raise

    @staticmethod
    def generate_summaries_for_document(
        dataset: Dataset,
        document: DatasetDocument,
        summary_index_setting: SummaryIndexSettingDict,
        session: Session | None = None,
        segment_ids: list[str] | None = None,
        only_parent_chunks: bool = False,
    ) -> list[DocumentSegmentSummary]:
        """
        Generate summaries for all segments in a document including vectorization.

        Args:
            dataset: Dataset containing the document
            document: DatasetDocument to generate summaries for
            summary_index_setting: Summary index configuration
            segment_ids: Optional list of specific segment IDs to process
            only_parent_chunks: If True, only process parent chunks (for parent-child mode)

        Returns:
            List of created DocumentSegmentSummary instances
        """
        # Only generate summary index for high_quality indexing technique
        if dataset.indexing_technique != IndexTechniqueType.HIGH_QUALITY:
            logger.info(
                "Skipping summary generation for dataset %s: indexing_technique is %s, not 'high_quality'",
                dataset.id,
                dataset.indexing_technique,
            )
            return []

        if not summary_index_setting or not summary_index_setting.get("enable"):
            logger.info("Summary index is disabled for dataset %s", dataset.id)
            return []

        # Skip qa_model documents
        if document.doc_form == "qa_model":
            logger.info("Skipping summary generation for qa_model document %s", document.id)
            return []

        logger.info(
            "Starting summary generation for document %s in dataset %s, segment_ids: %s, only_parent_chunks: %s",
            document.id,
            dataset.id,
            len(segment_ids) if segment_ids else "all",
            only_parent_chunks,
        )

        def _load_segments(query_session: Session) -> list[DocumentSegment]:
            # Query segments (only enabled segments)
            stmt = select(DocumentSegment).where(
                DocumentSegment.dataset_id == dataset.id,
                DocumentSegment.document_id == document.id,
                DocumentSegment.status == "completed",
                DocumentSegment.enabled.is_(True),  # Only generate summaries for enabled segments
            )

            if segment_ids:
                stmt = stmt.where(DocumentSegment.id.in_(segment_ids))

            return list(query_session.scalars(stmt).all())

        if session is None:
            with session_factory.create_session() as query_session:
                segments = _load_segments(query_session)
        else:
            segments = _load_segments(session)
            session.commit()

        if not segments:
            logger.info("No segments found for document %s", document.id)
            return []

        SummaryIndexService.batch_create_summary_records(
            segments=segments,
            dataset=dataset,
            status=SummaryStatus.NOT_STARTED,
        )

        summary_records = []

        for segment in segments:
            try:
                summary_record = SummaryIndexService.generate_and_vectorize_summary(
                    segment, dataset, summary_index_setting
                )
                summary_records.append(summary_record)
            except SummaryIndexConflictError:
                logger.info("Summary generation for segment %s was superseded", segment.id)
                continue
            except Exception:
                logger.exception("Failed to generate summary for segment %s", segment.id)
                continue

        logger.info(
            "Completed summary generation for document %s: %s summaries generated and vectorized",
            document.id,
            len(summary_records),
        )
        return summary_records

    @staticmethod
    def disable_summaries_for_segments(
        dataset: Dataset,
        session: Session | None = None,
        segment_ids: list[str] | None = None,
        disabled_by: str | None = None,
    ) -> None:
        """
        Disable summary records and remove vectors from vector database for segments.
        Unlike delete, this preserves the summary records but marks them as disabled.

        Args:
            dataset: Dataset containing the segments
            segment_ids: List of segment IDs to disable summaries for. If None, disable all.
            disabled_by: User ID who disabled the summaries
        """
        from libs.datetime_utils import naive_utc_now

        if segment_ids == []:
            return

        def _disable_with_session(write_session: Session) -> None:
            SummaryIndexService._lock_segment_rows(write_session, dataset.id, segment_ids)
            stmt = select(DocumentSegmentSummary).where(
                DocumentSegmentSummary.dataset_id == dataset.id,
                DocumentSegmentSummary.enabled.is_(True),  # Only disable enabled summaries
            )

            if segment_ids is not None:
                stmt = stmt.where(DocumentSegmentSummary.chunk_id.in_(segment_ids))

            summaries = write_session.scalars(
                stmt.order_by(DocumentSegmentSummary.chunk_id, DocumentSegmentSummary.id).with_for_update()
            ).all()

            if not summaries:
                return

            logger.info(
                "Disabling %s summary records for dataset %s, segment_ids: %s",
                len(summaries),
                dataset.id,
                len(segment_ids) if segment_ids else "all",
            )

            # Disable summary records (don't delete)
            summary_node_ids = [node_id for summary in summaries if (node_id := summary.summary_index_node_id)]
            vector = SummaryIndexService._create_cleanup_vector(dataset, write_session) if summary_node_ids else None
            now = naive_utc_now()
            write_session.execute(
                update(DocumentSegmentSummary)
                .where(DocumentSegmentSummary.id.in_(s.id for s in summaries))
                .values(enabled=False, disabled_at=now, disabled_by=disabled_by)
            )
            if vector is not None and summary_node_ids:
                vector.delete_by_ids(summary_node_ids)
            logger.info("Disabled %s summary records for dataset %s", len(summaries), dataset.id)

        if session is None:
            with session_factory.create_session() as write_session:
                _disable_with_session(write_session)
                write_session.commit()
        else:
            try:
                with session.begin_nested():
                    _disable_with_session(session)
            except Exception:
                session.refresh(dataset)
                session.commit()
                raise
            session.commit()

    @staticmethod
    def enable_summaries_for_segments(
        dataset: Dataset,
        session: Session | None = None,
        segment_ids: list[str] | None = None,
    ) -> None:
        """
        Enable summary records and re-add vectors to vector database for segments.

        Note: This method enables summaries based on chunk status, not summary_index_setting.enable.
        The summary_index_setting.enable flag only controls automatic generation,
        not whether existing summaries can be used.
        Summary.enabled should always be kept in sync with chunk.enabled.

        Args:
            dataset: Dataset containing the segments
            segment_ids: List of segment IDs to enable summaries for. If None, enable all.
        """
        # Only enable summary index for high_quality indexing technique
        if segment_ids == []:
            return
        if dataset.indexing_technique != IndexTechniqueType.HIGH_QUALITY:
            return

        summary_segment_pairs: list[tuple[DocumentSegmentSummary, DocumentSegment]] = []

        def _collect_candidates(query_session: Session) -> None:
            stmt = select(DocumentSegmentSummary).where(
                DocumentSegmentSummary.dataset_id == dataset.id,
                DocumentSegmentSummary.enabled.is_(False),  # Only enable disabled summaries
            )

            if segment_ids is not None:
                stmt = stmt.where(DocumentSegmentSummary.chunk_id.in_(segment_ids))

            summaries = query_session.scalars(stmt).all()

            if not summaries:
                return

            logger.info(
                "Enabling %s summary records for dataset %s, segment_ids: %s",
                len(summaries),
                dataset.id,
                len(segment_ids) if segment_ids else "all",
            )

            # Re-vectorize and re-add to vector database
            for summary in summaries:
                # Get the original segment
                segment = query_session.scalar(
                    select(DocumentSegment)
                    .where(
                        DocumentSegment.id == summary.chunk_id,
                        DocumentSegment.dataset_id == dataset.id,
                    )
                    .limit(1)
                )

                # Summary.enabled stays in sync with chunk.enabled,
                # only enable summary if the associated chunk is enabled.
                if not segment or not segment.enabled or segment.status != "completed":
                    continue

                if not summary.summary_content:
                    continue

                summary_segment_pairs.append((summary, segment))

        if session is None:
            with session_factory.create_session() as query_session:
                _collect_candidates(query_session)
        else:
            _collect_candidates(session)
            session.commit()

        enabled_count = 0
        for summary, segment in summary_segment_pairs:
            try:
                SummaryIndexService.vectorize_summary(summary, segment, dataset)
                if SummaryIndexService._enable_summary_record(summary.id, segment.id, dataset.id):
                    enabled_count += 1
            except Exception:
                logger.exception("Failed to re-vectorize summary %s", summary.id)
                continue

        logger.info("Enabled %s summary records for dataset %s", enabled_count, dataset.id)

    @staticmethod
    def delete_summaries_for_segments(
        dataset: Dataset,
        segment_ids: list[str] | None = None,
        *,
        session: Session | None = None,
    ) -> None:
        """
        Delete summary records and vectors for segments (used only for actual deletion scenarios).
        For disable/enable operations, use disable_summaries_for_segments/enable_summaries_for_segments.

        Args:
            dataset: Dataset containing the segments
            segment_ids: List of segment IDs to delete summaries for. If None, delete all.

        """
        if segment_ids == []:
            return

        def _delete_with_session(write_session: Session) -> None:
            SummaryIndexService._lock_segment_rows(write_session, dataset.id, segment_ids)
            stmt = select(DocumentSegmentSummary).where(DocumentSegmentSummary.dataset_id == dataset.id)

            if segment_ids is not None:
                stmt = stmt.where(DocumentSegmentSummary.chunk_id.in_(segment_ids))

            summaries = write_session.scalars(
                stmt.order_by(DocumentSegmentSummary.chunk_id, DocumentSegmentSummary.id).with_for_update()
            ).all()

            if not summaries:
                return

            summary_node_ids = [node_id for summary in summaries if (node_id := summary.summary_index_node_id)]
            vector = SummaryIndexService._create_cleanup_vector(dataset, write_session) if summary_node_ids else None
            summary_count = len(summaries)
            for summary in summaries:
                write_session.delete(summary)

            if vector is not None and summary_node_ids:
                vector.delete_by_ids(summary_node_ids)
            logger.info("Deleted %s summary records for dataset %s", summary_count, dataset.id)

        if session is None:
            with session_factory.create_session() as write_session:
                _delete_with_session(write_session)
                write_session.commit()
        else:
            try:
                with session.begin_nested():
                    _delete_with_session(session)
            except Exception:
                session.refresh(dataset)
                session.commit()
                raise
            session.commit()

    @staticmethod
    def update_summary_for_segment(
        segment: DocumentSegment,
        dataset: Dataset,
        summary_content: str,
        *,
        session: Session | None = None,
    ) -> DocumentSegmentSummary | None:
        """
        Update summary for a segment and re-vectorize it.

        Args:
            segment: DocumentSegment to update summary for
            dataset: Dataset containing the segment
            summary_content: New summary content

        Returns:
            Updated DocumentSegmentSummary instance, or None if indexing technique is not high_quality
        """
        # Only update summary index for high_quality indexing technique
        if dataset.indexing_technique != IndexTechniqueType.HIGH_QUALITY:
            return None

        # When user manually provides summary, allow saving even if summary_index_setting doesn't exist
        # summary_index_setting is only needed for LLM generation, not for manual summary vectorization
        # Vectorization uses dataset.embedding_model, which doesn't require summary_index_setting

        def _load_doc_form(query_session: Session) -> str | None:
            return query_session.scalar(
                select(DatasetDocument.doc_form).where(
                    DatasetDocument.id == segment.document_id,
                    DatasetDocument.dataset_id == dataset.id,
                )
            )

        if session is None:
            with session_factory.create_session() as query_session:
                doc_form = _load_doc_form(query_session)
        else:
            doc_form = _load_doc_form(session)
        if doc_form == "qa_model":
            return None

        if not summary_content or not summary_content.strip():

            def _delete_with_session(write_session: Session) -> bool:
                SummaryIndexService._lock_segment_rows(write_session, dataset.id, [segment.id])
                summary_record = SummaryIndexService._get_summary_record(
                    write_session,
                    segment.id,
                    dataset.id,
                    for_update=True,
                )

                if summary_record:
                    old_summary_node_id = summary_record.summary_index_node_id
                    vector = (
                        SummaryIndexService._create_cleanup_vector(dataset, write_session)
                        if old_summary_node_id
                        else None
                    )
                    write_session.delete(summary_record)
                    if vector is not None and old_summary_node_id:
                        vector.delete_by_ids([old_summary_node_id])
                return summary_record is not None

            if session is None:
                with session_factory.create_session() as write_session:
                    summary_deleted = _delete_with_session(write_session)
                    write_session.commit()
            else:
                try:
                    with session.begin_nested():
                        summary_deleted = _delete_with_session(session)
                except Exception:
                    session.refresh(dataset)
                    session.commit()
                    raise
                session.commit()

            if summary_deleted:
                logger.info("Deleted summary for segment %s (empty content provided)", segment.id)
            else:
                logger.info("No summary record found for segment %s, nothing to delete", segment.id)
            return None

        if session is not None:
            session.commit()

        summary_record = SummaryIndexService._save_summary_content(
            segment=segment,
            dataset=dataset,
            summary_content=summary_content,
            status=SummaryStatus.GENERATING,
        )

        try:
            SummaryIndexService.vectorize_summary(summary_record, segment, dataset)
            logger.info("Successfully updated and re-vectorized summary for segment %s", segment.id)
            return summary_record
        except SummaryIndexConflictError:
            logger.info("Summary update for segment %s was superseded", segment.id)
            raise
        except Exception as e:
            logger.exception("Failed to vectorize summary for segment %s", segment.id)
            error = f"Vectorization failed: {str(e)}"
            summary_record.status = SummaryStatus.ERROR
            summary_record.error = error
            return summary_record

    @staticmethod
    def get_segment_summary(
        segment_id: str,
        dataset_id: str,
        *,
        session: Session,
    ) -> DocumentSegmentSummary | None:
        """
        Get summary for a single segment.

        Args:
            segment_id: Segment ID (chunk_id)
            dataset_id: Dataset ID

        Keyword Args:
            session: SQLAlchemy session used to read summary records.

        Returns:
            DocumentSegmentSummary instance if found, None otherwise
        """
        return session.scalar(
            select(DocumentSegmentSummary)
            .where(
                DocumentSegmentSummary.chunk_id == segment_id,
                DocumentSegmentSummary.dataset_id == dataset_id,
                DocumentSegmentSummary.enabled.is_(True),
            )
            .limit(1)
        )

    @staticmethod
    def get_segments_summaries(
        segment_ids: list[str],
        dataset_id: str,
        *,
        session: Session,
    ) -> dict[str, DocumentSegmentSummary]:
        """
        Get summaries for multiple segments.

        Args:
            segment_ids: List of segment IDs (chunk_ids)
            dataset_id: Dataset ID

        Keyword Args:
            session: SQLAlchemy session used to read summary records.

        Returns:
            Dictionary mapping segment_id to DocumentSegmentSummary (only enabled summaries)
        """
        if not segment_ids:
            return {}

        summaries = session.scalars(
            select(DocumentSegmentSummary).where(
                DocumentSegmentSummary.chunk_id.in_(segment_ids),
                DocumentSegmentSummary.dataset_id == dataset_id,
                DocumentSegmentSummary.enabled.is_(True),
            )
        ).all()
        return {summary.chunk_id: summary for summary in summaries}

    @staticmethod
    def get_document_summaries(
        document_id: str,
        dataset_id: str,
        segment_ids: list[str] | None = None,
        *,
        session: Session,
    ) -> list[DocumentSegmentSummary]:
        """
        Get all summary records for a document.

        Args:
            document_id: Document ID
            dataset_id: Dataset ID
            segment_ids: Optional list of segment IDs to filter by

        Keyword Args:
            session: SQLAlchemy session used to read summary records.

        Returns:
            List of DocumentSegmentSummary instances (only enabled summaries)
        """
        stmt = select(DocumentSegmentSummary).where(
            DocumentSegmentSummary.document_id == document_id,
            DocumentSegmentSummary.dataset_id == dataset_id,
            DocumentSegmentSummary.enabled.is_(True),
        )

        if segment_ids:
            stmt = stmt.where(DocumentSegmentSummary.chunk_id.in_(segment_ids))

        return list(session.scalars(stmt).all())

    @staticmethod
    def get_document_summary_index_status(
        document_id: str,
        dataset_id: str,
        tenant_id: str,
        *,
        session: Session,
    ) -> str | None:
        """
        Get summary_index_status for a single document.

        Args:
            document_id: Document ID
            dataset_id: Dataset ID
            tenant_id: Tenant ID

        Keyword Args:
            session: SQLAlchemy session used to read summary status.

        Returns:
            "SUMMARIZING" if there are pending summaries, None otherwise
        """
        # Get all segments for this document (excluding qa_model and re_segment)
        segment_ids = list(
            session.scalars(
                select(DocumentSegment.id).where(
                    DocumentSegment.document_id == document_id,
                    DocumentSegment.status != "re_segment",
                    DocumentSegment.tenant_id == tenant_id,
                )
            ).all()
        )

        if not segment_ids:
            return None

        # Get all summary records for these segments
        summaries = SummaryIndexService.get_segments_summaries(segment_ids, dataset_id, session=session)
        summary_status_map = {chunk_id: summary.status for chunk_id, summary in summaries.items()}

        # Check if there are any "not_started" or "generating" status summaries
        has_pending_summaries = any(
            summary_status_map.get(segment_id) is not None  # Ensure summary exists (enabled=True)
            and summary_status_map[segment_id] in (SummaryStatus.NOT_STARTED, SummaryStatus.GENERATING)
            for segment_id in segment_ids
        )

        return "SUMMARIZING" if has_pending_summaries else None

    @staticmethod
    def get_documents_summary_index_status(
        document_ids: list[str],
        dataset_id: str,
        tenant_id: str,
        *,
        session: Session,
    ) -> dict[str, str | None]:
        """
        Get summary_index_status for multiple documents.

        Args:
            document_ids: List of document IDs
            dataset_id: Dataset ID
            tenant_id: Tenant ID

        Keyword Args:
            session: SQLAlchemy session used to read summary status.

        Returns:
            Dictionary mapping document_id to summary_index_status ("SUMMARIZING" or None)
        """
        if not document_ids:
            return {}

        # Get all segments for these documents (excluding qa_model and re_segment)
        segments = session.execute(
            select(DocumentSegment.id, DocumentSegment.document_id).where(
                DocumentSegment.document_id.in_(document_ids),
                DocumentSegment.status != "re_segment",
                DocumentSegment.tenant_id == tenant_id,
            )
        ).all()

        # Group segments by document_id
        document_segments_map: dict[str, list[str]] = {}
        for segment in segments:
            doc_id = str(segment.document_id)
            if doc_id not in document_segments_map:
                document_segments_map[doc_id] = []
            document_segments_map[doc_id].append(segment.id)

        # Get all summary records for these segments
        all_segment_ids = [seg.id for seg in segments]
        summaries = SummaryIndexService.get_segments_summaries(all_segment_ids, dataset_id, session=session)
        summary_status_map = {chunk_id: summary.status for chunk_id, summary in summaries.items()}

        # Calculate summary_index_status for each document
        result: dict[str, str | None] = {}
        for doc_id in document_ids:
            segment_ids = document_segments_map.get(doc_id, [])
            if not segment_ids:
                # No segments, status is None (not started)
                result[doc_id] = None
                continue

            # Check if there are any "not_started" or "generating" status summaries
            # Only check enabled=True summaries (already filtered in query)
            # If segment has no summary record (summary_status_map.get returns None),
            # it means the summary is disabled (enabled=False) or not created yet, ignore it
            has_pending_summaries = any(
                summary_status_map.get(segment_id) is not None  # Ensure summary exists (enabled=True)
                and summary_status_map[segment_id] in (SummaryStatus.NOT_STARTED, SummaryStatus.GENERATING)
                for segment_id in segment_ids
            )

            if has_pending_summaries:
                # Task is still running (not started or generating)
                result[doc_id] = "SUMMARIZING"
            else:
                # All enabled=True summaries are "completed" or "error", task finished
                # Or no enabled=True summaries exist (all disabled)
                result[doc_id] = None

        return result

    @staticmethod
    def get_document_summary_status_detail(
        document_id: str,
        dataset_id: str,
        session: Session,
    ) -> DocumentSummaryStatusDetailDict:
        """
        Get detailed summary status for a document.

        Args:
            document_id: Document ID
            dataset_id: Dataset ID
            session: SQLAlchemy session used for segment lookup

        Returns:
            Dictionary containing:
            - total_segments: Total number of segments in the document
            - summary_status: Dictionary with status counts
              - completed: Number of summaries completed
              - generating: Number of summaries being generated
              - error: Number of summaries with errors
              - not_started: Number of segments without summary records
              - timeout: Number of summaries that timed out
            - summaries: List of summary records with status and content preview
        """
        from services.dataset_service import SegmentService

        # Get all segments for this document
        segments = SegmentService.get_segments_by_document_and_dataset(
            document_id=document_id,
            dataset_id=dataset_id,
            session=session,
            status="completed",
            enabled=True,
        )

        total_segments = len(segments)

        # Get all summary records for these segments
        segment_ids = [segment.id for segment in segments]
        summaries = []
        if segment_ids:
            summaries = SummaryIndexService.get_document_summaries(
                document_id=document_id,
                dataset_id=dataset_id,
                segment_ids=segment_ids,
                session=session,
            )

        # Create a mapping of chunk_id to summary
        summary_map = {summary.chunk_id: summary for summary in summaries}

        # Count statuses
        status_counts = {
            SummaryStatus.COMPLETED: 0,
            SummaryStatus.GENERATING: 0,
            SummaryStatus.ERROR: 0,
            SummaryStatus.NOT_STARTED: 0,
        }

        summary_list: list[SummaryEntryDict] = []
        for segment in segments:
            summary = summary_map.get(segment.id)
            if summary:
                status = SummaryStatus(summary.status)
                status_counts[status] = status_counts.get(status, 0) + 1
                summary_list.append(
                    {
                        "segment_id": segment.id,
                        "segment_position": segment.position,
                        "status": summary.status,
                        "summary_preview": (
                            summary.summary_content[:100] + "..."
                            if summary.summary_content and len(summary.summary_content) > 100
                            else summary.summary_content
                        ),
                        "error": summary.error,
                        "created_at": int(summary.created_at.timestamp()) if summary.created_at else None,
                        "updated_at": int(summary.updated_at.timestamp()) if summary.updated_at else None,
                    }
                )
            else:
                status_counts[SummaryStatus.NOT_STARTED] += 1
                summary_list.append(
                    {
                        "segment_id": segment.id,
                        "segment_position": segment.position,
                        "status": SummaryStatus.NOT_STARTED,
                        "summary_preview": None,
                        "error": None,
                        "created_at": None,
                        "updated_at": None,
                    }
                )

        return DocumentSummaryStatusDetailDict(
            total_segments=total_segments,
            summary_status=cast(dict[str, int], status_counts),
            summaries=summary_list,
        )

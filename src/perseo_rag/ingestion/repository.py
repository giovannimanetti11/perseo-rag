from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from perseo_rag.ingestion.models import NormalizedDocument
from perseo_rag.storage.schema import (
    ChunkRecord,
    CollectionRecord,
    DocumentRecord,
    DocumentVersionRecord,
)


class CollectionNotFound(LookupError):
    pass


class SqlAlchemyIngestionRepository:
    def __init__(self, session: Session, scope_id: UUID) -> None:
        self._session = session
        self._scope_id = scope_id

    def ensure_collection(self, collection_id: UUID) -> None:
        statement = select(CollectionRecord.id).where(
            CollectionRecord.id == collection_id,
            CollectionRecord.scope_id == self._scope_id,
        )
        if self._session.scalar(statement) is None:
            raise CollectionNotFound(f"collection {collection_id} was not found")

    def get_or_create_document(self, document: NormalizedDocument) -> UUID:
        document_id = uuid4()
        statement = (
            insert(DocumentRecord)
            .values(
                id=document_id,
                scope_id=self._scope_id,
                collection_id=document.collection_id,
                source_type=document.source_type,
                source_ref=document.source_ref,
            )
            .on_conflict_do_nothing(constraint="uq_documents_source")
            .returning(DocumentRecord.id)
        )
        created_id = self._session.scalar(statement)

        if created_id is not None:
            return created_id

        existing = self._session.scalar(
            select(DocumentRecord.id).where(
                DocumentRecord.scope_id == self._scope_id,
                DocumentRecord.collection_id == document.collection_id,
                DocumentRecord.source_type == document.source_type,
                DocumentRecord.source_ref == document.source_ref,
            )
        )
        if existing is None:
            raise RuntimeError("document insert conflicted but no existing document was found")

        return existing

    def get_or_create_version(
        self,
        document_id: UUID,
        document: NormalizedDocument,
    ) -> tuple[UUID, bool]:
        version_id = uuid4()
        statement = (
            insert(DocumentVersionRecord)
            .values(
                id=version_id,
                scope_id=self._scope_id,
                document_id=document_id,
                content_hash=document.fingerprint,
                metadata=document.metadata,
            )
            .on_conflict_do_nothing(constraint="uq_document_versions_content")
            .returning(DocumentVersionRecord.id)
        )
        created_id = self._session.scalar(statement)

        if created_id is not None:
            return created_id, True

        existing = self._session.scalar(
            select(DocumentVersionRecord.id).where(
                DocumentVersionRecord.scope_id == self._scope_id,
                DocumentVersionRecord.document_id == document_id,
                DocumentVersionRecord.content_hash == document.fingerprint,
            )
        )
        if existing is None:
            raise RuntimeError("version insert conflicted but no existing version was found")

        return existing, False

    def add_chunks(self, version_id: UUID, chunks: tuple[str, ...]) -> None:
        self._session.add_all(
            ChunkRecord(
                id=uuid4(),
                scope_id=self._scope_id,
                document_version_id=version_id,
                position=position,
                content=content,
            )
            for position, content in enumerate(chunks)
        )

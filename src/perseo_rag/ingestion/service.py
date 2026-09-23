from sqlalchemy.orm import Session, sessionmaker

from perseo_rag.ingestion.chunking import TextChunker
from perseo_rag.ingestion.models import DocumentInput, IngestionResult
from perseo_rag.ingestion.normalization import normalize_document
from perseo_rag.ingestion.repository import SqlAlchemyIngestionRepository
from perseo_rag.security import ScopeContext, scoped_session


class IngestionService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        chunker: TextChunker | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._chunker = chunker or TextChunker()

    def ingest(self, scope: ScopeContext, document: DocumentInput) -> IngestionResult:
        normalized = normalize_document(document)

        with scoped_session(self._session_factory, scope) as session:
            repository = SqlAlchemyIngestionRepository(session, scope.id)
            repository.ensure_collection(normalized.collection_id)

            document_id = repository.get_or_create_document(normalized)
            repository.lock_document(document_id)
            version_id, created = repository.get_or_create_version(document_id, normalized)

            chunk_count = 0
            if created:
                chunks = self._chunker.split(normalized.content)
                repository.add_chunks(version_id, chunks)
                chunk_count = len(chunks)

            repository.set_document_head(document_id, version_id)

            return IngestionResult(
                document_id=document_id,
                version_id=version_id,
                created=created,
                chunk_count=chunk_count,
            )

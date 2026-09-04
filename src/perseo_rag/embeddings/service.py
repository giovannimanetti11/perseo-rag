from sqlalchemy.orm import Session, sessionmaker

from perseo_rag.embeddings.models import EmbeddingIndexResult
from perseo_rag.embeddings.provider import EmbeddingProvider
from perseo_rag.embeddings.repository import SqlAlchemyEmbeddingRepository
from perseo_rag.embeddings.validation import validate_provider, validate_vectors
from perseo_rag.security import ScopeContext, scoped_session


class EmbeddingService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        batch_size: int = 64,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self._session_factory = session_factory
        self._batch_size = batch_size

    def index_scope(
        self,
        scope: ScopeContext,
        provider: EmbeddingProvider,
    ) -> EmbeddingIndexResult:
        validate_provider(provider)
        indexed_chunks = 0

        with scoped_session(self._session_factory, scope) as session:
            repository = SqlAlchemyEmbeddingRepository(session, scope.id)

            while chunks := repository.pending_chunks(provider.key, self._batch_size):
                vectors = validate_vectors(
                    provider.embed([chunk.content for chunk in chunks]),
                    expected_count=len(chunks),
                    dimensions=provider.dimensions,
                )
                repository.add_embeddings(
                    provider.key,
                    provider.dimensions,
                    chunks,
                    vectors,
                )
                session.flush()
                indexed_chunks += len(chunks)

        return EmbeddingIndexResult(
            provider_key=provider.key,
            indexed_chunks=indexed_chunks,
        )

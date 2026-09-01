import math
from collections.abc import Sequence

from sqlalchemy.orm import Session, sessionmaker

from perseo_rag.embeddings.models import EmbeddingIndexResult
from perseo_rag.embeddings.provider import EmbeddingProvider
from perseo_rag.embeddings.repository import SqlAlchemyEmbeddingRepository
from perseo_rag.security import ScopeContext, scoped_session


class InvalidEmbedding(ValueError):
    pass


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
        self._validate_provider(provider)
        indexed_chunks = 0

        with scoped_session(self._session_factory, scope) as session:
            repository = SqlAlchemyEmbeddingRepository(session, scope.id)

            while chunks := repository.pending_chunks(provider.key, self._batch_size):
                vectors = self._validate_vectors(
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

    def _validate_provider(self, provider: EmbeddingProvider) -> None:
        if not provider.key.strip():
            raise InvalidEmbedding("provider key must not be empty")

        if provider.dimensions < 1:
            raise InvalidEmbedding("provider dimensions must be positive")

    def _validate_vectors(
        self,
        vectors: Sequence[Sequence[float]],
        expected_count: int,
        dimensions: int,
    ) -> tuple[tuple[float, ...], ...]:
        if len(vectors) != expected_count:
            raise InvalidEmbedding("provider returned an unexpected number of embeddings")

        validated: list[tuple[float, ...]] = []
        for vector in vectors:
            values = tuple(float(value) for value in vector)
            if len(values) != dimensions:
                raise InvalidEmbedding("provider returned an embedding with invalid dimensions")

            if not all(math.isfinite(value) for value in values):
                raise InvalidEmbedding("provider returned a non-finite embedding")

            validated.append(values)

        return tuple(validated)

from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.orm import Session, sessionmaker

from perseo_rag.embeddings.provider import EmbeddingProvider
from perseo_rag.embeddings.validation import embed_query
from perseo_rag.retrieval.models import RetrievalHit
from perseo_rag.security import ScopeContext, scoped_session
from perseo_rag.storage.schema import (
    ChunkEmbeddingRecord,
    ChunkRecord,
    DocumentRecord,
    DocumentVersionRecord,
)


class DenseRetriever:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        max_results: int = 50,
    ) -> None:
        if max_results < 1:
            raise ValueError("max_results must be positive")

        self._session_factory = session_factory
        self._max_results = max_results

    def search(
        self,
        scope: ScopeContext,
        query: str,
        provider: EmbeddingProvider,
        *,
        limit: int = 10,
        collection_id: UUID | None = None,
    ) -> tuple[RetrievalHit, ...]:
        normalized_query = query.strip()
        if not normalized_query:
            return ()

        if limit < 1 or limit > self._max_results:
            raise ValueError(f"limit must be between 1 and {self._max_results}")

        query_vector = embed_query(provider, normalized_query)

        with scoped_session(self._session_factory, scope) as session:
            distance = ChunkEmbeddingRecord.embedding.cosine_distance(list(query_vector)).label(
                "distance"
            )
            statement = (
                select(
                    ChunkRecord.scope_id,
                    ChunkRecord.id,
                    ChunkRecord.document_version_id,
                    DocumentRecord.id,
                    DocumentRecord.source_ref,
                    ChunkRecord.content,
                    distance,
                )
                .join(
                    ChunkRecord,
                    and_(
                        ChunkRecord.scope_id == ChunkEmbeddingRecord.scope_id,
                        ChunkRecord.id == ChunkEmbeddingRecord.chunk_id,
                    ),
                )
                .join(
                    DocumentVersionRecord,
                    and_(
                        DocumentVersionRecord.scope_id == ChunkRecord.scope_id,
                        DocumentVersionRecord.id == ChunkRecord.document_version_id,
                    ),
                )
                .join(
                    DocumentRecord,
                    and_(
                        DocumentRecord.scope_id == DocumentVersionRecord.scope_id,
                        DocumentRecord.id == DocumentVersionRecord.document_id,
                    ),
                )
                .where(
                    ChunkEmbeddingRecord.scope_id == scope.id,
                    ChunkEmbeddingRecord.provider_key == provider.key,
                    ChunkEmbeddingRecord.dimensions == provider.dimensions,
                )
                .order_by(distance.asc(), ChunkRecord.id)
                .limit(limit)
            )

            if collection_id is not None:
                statement = statement.where(DocumentRecord.collection_id == collection_id)

            return tuple(
                RetrievalHit(
                    scope_id=scope_id,
                    chunk_id=chunk_id,
                    document_version_id=version_id,
                    document_id=document_id,
                    source_ref=source_ref,
                    content=content,
                    score=1.0 - float(cosine_distance),
                )
                for (
                    scope_id,
                    chunk_id,
                    version_id,
                    document_id,
                    source_ref,
                    content,
                    cosine_distance,
                ) in session.execute(statement)
            )

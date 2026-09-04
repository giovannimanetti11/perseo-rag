from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from perseo_rag.embeddings.provider import EmbeddingProvider
from perseo_rag.retrieval.dense import DenseRetriever
from perseo_rag.retrieval.fusion import reciprocal_rank_fusion
from perseo_rag.retrieval.lexical import LexicalRetriever
from perseo_rag.retrieval.models import HybridRetrievalHit
from perseo_rag.security import ScopeContext


class HybridRetriever:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        candidate_limit: int = 20,
        max_results: int = 50,
        rank_constant: int = 60,
    ) -> None:
        if candidate_limit < 1 or candidate_limit > max_results:
            raise ValueError("candidate_limit must be between 1 and max_results")

        if rank_constant < 1:
            raise ValueError("rank_constant must be positive")

        self._candidate_limit = candidate_limit
        self._max_results = max_results
        self._rank_constant = rank_constant
        self._lexical = LexicalRetriever(session_factory, max_results=max_results)
        self._dense = DenseRetriever(session_factory, max_results=max_results)

    def search(
        self,
        scope: ScopeContext,
        query: str,
        provider: EmbeddingProvider,
        *,
        limit: int = 10,
        collection_id: UUID | None = None,
    ) -> tuple[HybridRetrievalHit, ...]:
        if limit < 1 or limit > self._max_results:
            raise ValueError(f"limit must be between 1 and {self._max_results}")

        candidate_limit = max(limit, self._candidate_limit)
        lexical_hits = self._lexical.search(
            scope,
            query,
            limit=candidate_limit,
            collection_id=collection_id,
        )
        dense_hits = self._dense.search(
            scope,
            query,
            provider,
            limit=candidate_limit,
            collection_id=collection_id,
        )

        return reciprocal_rank_fusion(
            lexical_hits,
            dense_hits,
            limit=limit,
            rank_constant=self._rank_constant,
        )

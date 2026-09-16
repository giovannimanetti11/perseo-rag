from typing import Protocol

from perseo_rag.application.models import QueryInput, QueryResult
from perseo_rag.embeddings.provider import EmbeddingProvider
from perseo_rag.generation.provider import GenerationProvider
from perseo_rag.generation.service import GenerationService
from perseo_rag.grounding.builder import GroundedContextBuilder
from perseo_rag.retrieval.hybrid import HybridRetriever
from perseo_rag.security import ScopeContext


class QueryHandler(Protocol):
    def answer(self, scope: ScopeContext, query: QueryInput) -> QueryResult: ...


class QueryService:
    def __init__(
        self,
        retriever: HybridRetriever,
        context_builder: GroundedContextBuilder,
        generation_service: GenerationService,
        embedding_provider: EmbeddingProvider,
        generation_provider: GenerationProvider,
    ) -> None:
        self._retriever = retriever
        self._context_builder = context_builder
        self._generation_service = generation_service
        self._embedding_provider = embedding_provider
        self._generation_provider = generation_provider

    def answer(self, scope: ScopeContext, query: QueryInput) -> QueryResult:
        question = query.question.strip()
        if not question:
            raise ValueError("question must not be empty")

        hits = self._retriever.search(
            scope,
            question,
            self._embedding_provider,
            limit=query.limit,
            collection_id=query.collection_id,
        )
        context = self._context_builder.build(scope, hits)
        answer = self._generation_service.answer(
            question,
            context,
            self._generation_provider,
        )

        return QueryResult(
            answer=answer,
            evidence_count=len(context.evidence),
            context_chars=context.total_chars,
            context_truncated=context.truncated,
        )

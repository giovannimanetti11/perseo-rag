from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from perseo_rag.application import QueryInput, QueryService
from perseo_rag.generation import (
    AnswerStatus,
    GenerationOutput,
    GenerationRequest,
    GenerationService,
)
from perseo_rag.grounding import GroundedContextBuilder
from perseo_rag.retrieval import HybridRetrievalHit
from perseo_rag.security import ScopeContext


@dataclass(frozen=True, slots=True)
class EmbeddingProviderFixture:
    key: str = "embedding-fixture"
    dimensions: int = 2

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[1.0, 0.0] for _ in texts]


@dataclass
class GenerationProviderFixture:
    key: str = "generation-fixture"
    calls: int = 0
    last_request: GenerationRequest | None = None

    def generate(self, request: GenerationRequest) -> GenerationOutput:
        self.calls += 1
        self.last_request = request
        return GenerationOutput(
            text="Grounded result",
            citation_ids=("S1",),
        )


class RetrieverFixture:
    def __init__(self, hits: tuple[HybridRetrievalHit, ...]) -> None:
        self.hits = hits
        self.last_query: str | None = None
        self.last_limit: int | None = None
        self.last_collection_id: UUID | None = None

    def search(
        self,
        scope: ScopeContext,
        query: str,
        provider: EmbeddingProviderFixture,
        *,
        limit: int = 10,
        collection_id: UUID | None = None,
    ) -> tuple[HybridRetrievalHit, ...]:
        self.last_query = query
        self.last_limit = limit
        self.last_collection_id = collection_id
        return self.hits


def _hit(scope: ScopeContext) -> HybridRetrievalHit:
    return HybridRetrievalHit(
        scope_id=scope.id,
        chunk_id=UUID(int=2),
        document_version_id=UUID(int=3),
        document_id=UUID(int=4),
        source_ref="fixture:source",
        content="Evidence for the answer",
        score=0.5,
        lexical_rank=1,
        dense_rank=1,
    )


def test_query_service_orchestrates_retrieval_context_and_generation() -> None:
    scope = ScopeContext(UUID(int=1))
    collection_id = UUID(int=10)
    retriever = RetrieverFixture((_hit(scope),))
    generation_provider = GenerationProviderFixture()
    service = QueryService(
        retriever,  # type: ignore[arg-type]
        GroundedContextBuilder(),
        GenerationService(),
        EmbeddingProviderFixture(),
        generation_provider,
    )

    result = service.answer(
        scope,
        QueryInput(
            question="  explain this  ",
            collection_id=collection_id,
            limit=5,
        ),
    )

    assert result.answer.status is AnswerStatus.ANSWERED
    assert result.answer.text == "Grounded result"
    assert result.evidence_count == 1
    assert result.context_chars == len("Evidence for the answer")
    assert result.context_truncated is False
    assert retriever.last_query == "explain this"
    assert retriever.last_limit == 5
    assert retriever.last_collection_id == collection_id
    assert generation_provider.calls == 1
    assert generation_provider.last_request is not None
    assert generation_provider.last_request.evidence[0].citation_id == "S1"


def test_query_service_abstains_when_retrieval_returns_no_evidence() -> None:
    scope = ScopeContext(UUID(int=1))
    generation_provider = GenerationProviderFixture()
    service = QueryService(
        RetrieverFixture(()),  # type: ignore[arg-type]
        GroundedContextBuilder(),
        GenerationService(),
        EmbeddingProviderFixture(),
        generation_provider,
    )

    result = service.answer(scope, QueryInput(question="question"))

    assert result.answer.status is AnswerStatus.ABSTAINED
    assert result.evidence_count == 0
    assert generation_provider.calls == 0

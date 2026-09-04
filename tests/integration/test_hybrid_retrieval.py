from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session, sessionmaker

from perseo_rag.config import Settings
from perseo_rag.embeddings import EmbeddingService
from perseo_rag.ingestion import DocumentInput, IngestionService
from perseo_rag.retrieval import HybridRetriever
from perseo_rag.security import ScopeContext, scoped_session
from perseo_rag.storage import build_engine, build_session_factory
from perseo_rag.storage.schema import CollectionRecord, ScopeRecord


@dataclass(frozen=True, slots=True)
class FusionProvider:
    key: str = "fusion-fixture"
    dimensions: int = 2

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        normalized = text.lower()
        if normalized.strip() == "canonical":
            return [1.0, 0.0]
        if "semantic-target" in normalized:
            return [1.0, 0.0]
        return [0.0, 1.0]


@pytest.fixture(scope="module")
def session_factory() -> sessionmaker[Session]:
    engine = build_engine(Settings(_env_file=None))
    factory = build_session_factory(engine)
    yield factory
    engine.dispose()


def _create_scope(
    session_factory: sessionmaker[Session],
    label: str,
) -> tuple[ScopeContext, UUID]:
    scope = ScopeContext(uuid4())
    collection_id = uuid4()

    with scoped_session(session_factory, scope) as session:
        session.add(ScopeRecord(id=scope.id, slug=f"{label}-{scope.id.hex}", name=label))
        session.flush()
        session.add(CollectionRecord(id=collection_id, scope_id=scope.id, name="Default"))

    return scope, collection_id


@pytest.mark.integration
def test_hybrid_retrieval_fuses_lexical_and_dense_rankings(
    session_factory: sessionmaker[Session],
) -> None:
    scope, collection_id = _create_scope(session_factory, "hybrid")
    ingestion = IngestionService(session_factory)
    provider = FusionProvider()

    ingestion.ingest(
        scope,
        DocumentInput(
            collection_id=collection_id,
            source_type="text",
            source_ref="fixture:lexical",
            content="Canonical tag is missing from this page.",
        ),
    )
    ingestion.ingest(
        scope,
        DocumentInput(
            collection_id=collection_id,
            source_type="text",
            source_ref="fixture:semantic",
            content="semantic-target metadata configuration requires attention.",
        ),
    )

    EmbeddingService(session_factory).index_scope(scope, provider)
    hits = HybridRetriever(session_factory, candidate_limit=10).search(
        scope,
        "canonical",
        provider,
    )

    assert len(hits) == 2
    assert hits[0].source_ref == "fixture:lexical"
    assert hits[0].lexical_rank == 1
    assert hits[0].dense_rank is not None
    assert hits[1].source_ref == "fixture:semantic"
    assert hits[1].dense_rank == 1


@pytest.mark.integration
def test_hybrid_retrieval_remains_scope_isolated(
    session_factory: sessionmaker[Session],
) -> None:
    first_scope, first_collection = _create_scope(session_factory, "hybrid-first")
    second_scope, second_collection = _create_scope(session_factory, "hybrid-second")
    ingestion = IngestionService(session_factory)
    provider = FusionProvider()
    shared_ref = "fixture:same-source"

    ingestion.ingest(
        first_scope,
        DocumentInput(
            collection_id=first_collection,
            source_type="text",
            source_ref=shared_ref,
            content="Canonical evidence in the first scope.",
        ),
    )
    ingestion.ingest(
        second_scope,
        DocumentInput(
            collection_id=second_collection,
            source_type="text",
            source_ref=shared_ref,
            content="Canonical evidence in the second scope.",
        ),
    )

    EmbeddingService(session_factory).index_scope(first_scope, provider)
    EmbeddingService(session_factory).index_scope(second_scope, provider)

    hits = HybridRetriever(session_factory).search(first_scope, "canonical", provider)

    assert len(hits) == 1
    assert "first scope" in hits[0].content
    assert "second scope" not in hits[0].content

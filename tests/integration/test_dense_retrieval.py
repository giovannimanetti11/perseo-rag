from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session, sessionmaker

from perseo_rag.config import Settings
from perseo_rag.embeddings import EmbeddingService
from perseo_rag.ingestion import DocumentInput, IngestionService
from perseo_rag.retrieval import DenseRetriever
from perseo_rag.security import ScopeContext, scoped_session
from perseo_rag.storage import build_engine, build_session_factory
from perseo_rag.storage.schema import CollectionRecord, ScopeRecord


@dataclass(frozen=True, slots=True)
class KeywordProvider:
    key: str = "keyword-fixture"
    dimensions: int = 3

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        normalized = text.lower()
        if "canonical" in normalized:
            return [1.0, 0.0, 0.0]
        if "performance" in normalized:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


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


def _ingest(
    session_factory: sessionmaker[Session],
    scope: ScopeContext,
    collection_id: UUID,
    source_ref: str,
    content: str,
) -> None:
    IngestionService(session_factory).ingest(
        scope,
        DocumentInput(
            collection_id=collection_id,
            source_type="text",
            source_ref=source_ref,
            content=content,
        ),
    )


@pytest.mark.integration
def test_dense_retrieval_ranks_by_cosine_similarity(
    session_factory: sessionmaker[Session],
) -> None:
    scope, collection_id = _create_scope(session_factory, "dense-ranking")
    provider = KeywordProvider()

    _ingest(
        session_factory,
        scope,
        collection_id,
        "fixture:canonical",
        "Canonical metadata requires attention.",
    )
    _ingest(
        session_factory,
        scope,
        collection_id,
        "fixture:performance",
        "Performance metrics are within budget.",
    )

    EmbeddingService(session_factory).index_scope(scope, provider)
    hits = DenseRetriever(session_factory).search(scope, "canonical issue", provider)

    assert len(hits) == 2
    assert hits[0].source_ref == "fixture:canonical"
    assert hits[0].score > hits[1].score


@pytest.mark.integration
def test_dense_retrieval_never_crosses_scope_boundaries(
    session_factory: sessionmaker[Session],
) -> None:
    first_scope, first_collection = _create_scope(session_factory, "dense-first")
    second_scope, second_collection = _create_scope(session_factory, "dense-second")
    provider = KeywordProvider()
    shared_ref = "fixture:shared"

    _ingest(
        session_factory,
        first_scope,
        first_collection,
        shared_ref,
        "Canonical evidence from the first scope.",
    )
    _ingest(
        session_factory,
        second_scope,
        second_collection,
        shared_ref,
        "Canonical evidence from the second scope.",
    )

    EmbeddingService(session_factory).index_scope(first_scope, provider)
    EmbeddingService(session_factory).index_scope(second_scope, provider)

    hits = DenseRetriever(session_factory).search(first_scope, "canonical", provider)

    assert len(hits) == 1
    assert "first scope" in hits[0].content
    assert "second scope" not in hits[0].content


@pytest.mark.integration
def test_dense_retrieval_respects_collection_filter(
    session_factory: sessionmaker[Session],
) -> None:
    scope, first_collection = _create_scope(session_factory, "dense-collection")
    second_collection = uuid4()

    with scoped_session(session_factory, scope) as session:
        session.add(
            CollectionRecord(
                id=second_collection,
                scope_id=scope.id,
                name="Secondary",
            )
        )

    provider = KeywordProvider()
    _ingest(
        session_factory,
        scope,
        first_collection,
        "fixture:first",
        "Canonical issue in first collection.",
    )
    _ingest(
        session_factory,
        scope,
        second_collection,
        "fixture:second",
        "Canonical issue in second collection.",
    )

    EmbeddingService(session_factory).index_scope(scope, provider)
    hits = DenseRetriever(session_factory).search(
        scope,
        "canonical",
        provider,
        collection_id=second_collection,
    )

    assert len(hits) == 1
    assert hits[0].source_ref == "fixture:second"


@pytest.mark.integration
def test_dense_retrieval_uses_only_current_document_version(
    session_factory: sessionmaker[Session],
) -> None:
    scope, collection_id = _create_scope(session_factory, "dense-current")
    service = IngestionService(session_factory)
    provider = KeywordProvider()
    source_ref = "fixture:versioned"

    service.ingest(
        scope,
        DocumentInput(
            collection_id=collection_id,
            source_type="text",
            source_ref=source_ref,
            content="Canonical evidence from an obsolete version.",
        ),
    )
    current = service.ingest(
        scope,
        DocumentInput(
            collection_id=collection_id,
            source_type="text",
            source_ref=source_ref,
            content="Performance evidence from the current version.",
        ),
    )

    EmbeddingService(session_factory).index_scope(scope, provider)
    hits = DenseRetriever(session_factory).search(scope, "canonical", provider)

    assert len(hits) == 1
    assert hits[0].document_version_id == current.version_id
    assert hits[0].content == "Performance evidence from the current version."

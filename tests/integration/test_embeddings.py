from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from perseo_rag.config import Settings
from perseo_rag.embeddings import EmbeddingService
from perseo_rag.ingestion import DocumentInput, IngestionService, TextChunker
from perseo_rag.security import ScopeContext, scoped_session
from perseo_rag.storage import build_engine, build_session_factory
from perseo_rag.storage.schema import ChunkEmbeddingRecord, CollectionRecord, ScopeRecord


@dataclass(frozen=True, slots=True)
class FixtureProvider:
    key: str
    dimensions: int = 3

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[float(len(text)), float(text.count(" ")), 1.0] for text in texts]


@pytest.fixture(scope="module")
def session_factory() -> sessionmaker[Session]:
    engine = build_engine(Settings(_env_file=None))
    factory = build_session_factory(engine)
    yield factory
    engine.dispose()


def _create_indexable_scope(
    session_factory: sessionmaker[Session],
    label: str,
) -> ScopeContext:
    scope = ScopeContext(uuid4())
    collection_id = uuid4()

    with scoped_session(session_factory, scope) as session:
        session.add(ScopeRecord(id=scope.id, slug=f"{label}-{scope.id.hex}", name=label))
        session.flush()
        session.add(CollectionRecord(id=collection_id, scope_id=scope.id, name="Default"))

    IngestionService(session_factory, TextChunker(max_chars=18)).ingest(
        scope,
        DocumentInput(
            collection_id=collection_id,
            source_type="text",
            source_ref=f"fixture:{label}",
            content="First indexed chunk. Second indexed chunk.",
        ),
    )
    return scope


@pytest.mark.integration
def test_embeddings_are_indexed_once_per_provider(
    session_factory: sessionmaker[Session],
) -> None:
    scope = _create_indexable_scope(session_factory, "embedding-once")
    service = EmbeddingService(session_factory, batch_size=1)
    provider = FixtureProvider("fixture-v1")

    first = service.index_scope(scope, provider)
    second = service.index_scope(scope, provider)

    assert first.indexed_chunks > 0
    assert second.indexed_chunks == 0

    with scoped_session(session_factory, scope) as session:
        count = session.scalar(select(func.count()).select_from(ChunkEmbeddingRecord))
        dimensions = set(session.scalars(select(ChunkEmbeddingRecord.dimensions)))

    assert count == first.indexed_chunks
    assert dimensions == {provider.dimensions}


@pytest.mark.integration
def test_same_chunks_can_be_indexed_by_multiple_providers(
    session_factory: sessionmaker[Session],
) -> None:
    scope = _create_indexable_scope(session_factory, "embedding-providers")
    service = EmbeddingService(session_factory)

    first = service.index_scope(scope, FixtureProvider("fixture-a"))
    second = service.index_scope(scope, FixtureProvider("fixture-b"))

    assert first.indexed_chunks > 0
    assert second.indexed_chunks == first.indexed_chunks

    with scoped_session(session_factory, scope) as session:
        providers = set(session.scalars(select(ChunkEmbeddingRecord.provider_key)))

    assert providers == {"fixture-a", "fixture-b"}


@pytest.mark.integration
def test_embedding_index_is_scope_isolated(
    session_factory: sessionmaker[Session],
) -> None:
    first_scope = _create_indexable_scope(session_factory, "embedding-first")
    second_scope = _create_indexable_scope(session_factory, "embedding-second")
    service = EmbeddingService(session_factory)
    provider = FixtureProvider("fixture-isolated")

    service.index_scope(first_scope, provider)

    with scoped_session(session_factory, first_scope) as session:
        first_count = session.scalar(select(func.count()).select_from(ChunkEmbeddingRecord))

    with scoped_session(session_factory, second_scope) as session:
        second_count = session.scalar(select(func.count()).select_from(ChunkEmbeddingRecord))

    assert first_count is not None and first_count > 0
    assert second_count == 0

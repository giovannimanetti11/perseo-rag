from uuid import uuid4

import pytest
from sqlalchemy.orm import Session, sessionmaker

from perseo_rag.config import Settings
from perseo_rag.ingestion import DocumentInput, IngestionService
from perseo_rag.retrieval import LexicalRetriever
from perseo_rag.security import ScopeContext, scoped_session
from perseo_rag.storage import build_engine, build_session_factory
from perseo_rag.storage.schema import CollectionRecord, ScopeRecord


@pytest.fixture(scope="module")
def session_factory() -> sessionmaker[Session]:
    engine = build_engine(Settings(_env_file=None))
    factory = build_session_factory(engine)
    yield factory
    engine.dispose()


def _create_scope(
    session_factory: sessionmaker[Session],
    label: str,
) -> tuple[ScopeContext, object]:
    scope = ScopeContext(uuid4())
    collection_id = uuid4()

    with scoped_session(session_factory, scope) as session:
        session.add(ScopeRecord(id=scope.id, slug=f"{label}-{scope.id.hex}", name=label))
        session.flush()
        session.add(CollectionRecord(id=collection_id, scope_id=scope.id, name="Default"))

    return scope, collection_id


@pytest.mark.integration
def test_lexical_retrieval_returns_ranked_scope_local_evidence(
    session_factory: sessionmaker[Session],
) -> None:
    scope, collection_id = _create_scope(session_factory, "lexical")
    service = IngestionService(session_factory)

    service.ingest(
        scope,
        DocumentInput(
            collection_id=collection_id,
            source_type="web",
            source_ref="https://example.test/relevant",
            content="Canonical canonical metadata issue with duplicate canonical tags.",
        ),
    )
    service.ingest(
        scope,
        DocumentInput(
            collection_id=collection_id,
            source_type="web",
            source_ref="https://example.test/secondary",
            content="Canonical tag is missing.",
        ),
    )
    service.ingest(
        scope,
        DocumentInput(
            collection_id=collection_id,
            source_type="web",
            source_ref="https://example.test/unrelated",
            content="Image compression and layout stability.",
        ),
    )

    hits = LexicalRetriever(session_factory).search(scope, "canonical")

    assert len(hits) == 2
    assert hits[0].score >= hits[1].score
    assert hits[0].source_ref == "https://example.test/relevant"
    assert all("canonical" in hit.content.lower() for hit in hits)


@pytest.mark.integration
def test_lexical_retrieval_never_crosses_scope_boundaries(
    session_factory: sessionmaker[Session],
) -> None:
    first_scope, first_collection = _create_scope(session_factory, "lexical-first")
    second_scope, second_collection = _create_scope(session_factory, "lexical-second")
    service = IngestionService(session_factory)
    shared_ref = "https://same.example/page"

    service.ingest(
        first_scope,
        DocumentInput(
            collection_id=first_collection,
            source_type="web",
            source_ref=shared_ref,
            content="Canonical evidence belonging only to the first scope.",
        ),
    )
    service.ingest(
        second_scope,
        DocumentInput(
            collection_id=second_collection,
            source_type="web",
            source_ref=shared_ref,
            content="Canonical evidence belonging only to the second scope.",
        ),
    )

    hits = LexicalRetriever(session_factory).search(first_scope, "canonical")

    assert len(hits) == 1
    assert "first scope" in hits[0].content
    assert "second scope" not in hits[0].content


@pytest.mark.integration
def test_lexical_retrieval_accepts_untrusted_search_syntax(
    session_factory: sessionmaker[Session],
) -> None:
    scope, _ = _create_scope(session_factory, "lexical-syntax")

    hits = LexicalRetriever(session_factory).search(scope, '""" )( dummy \\ query <->')

    assert hits == ()

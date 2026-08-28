from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from perseo_rag.config import Settings
from perseo_rag.ingestion import CollectionNotFound, DocumentInput, IngestionService, TextChunker
from perseo_rag.security import ScopeContext, scoped_session
from perseo_rag.storage import build_engine, build_session_factory
from perseo_rag.storage.schema import (
    ChunkRecord,
    CollectionRecord,
    DocumentRecord,
    DocumentVersionRecord,
    ScopeRecord,
)


@pytest.fixture(scope="module")
def session_factory() -> sessionmaker[Session]:
    engine = build_engine(Settings(_env_file=None))
    factory = build_session_factory(engine)
    yield factory
    engine.dispose()


def _create_scope_with_collection(
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
def test_identical_document_state_is_idempotent(
    session_factory: sessionmaker[Session],
) -> None:
    scope, collection_id = _create_scope_with_collection(session_factory, "idempotent")
    service = IngestionService(session_factory, TextChunker(max_chars=32))

    first = service.ingest(
        scope,
        DocumentInput(
            collection_id=collection_id,
            source_type="web",
            source_ref="https://example.test/page",
            content="First section.\r\n\r\nSecond section.",
            metadata={"status": 200, "language": "en"},
        ),
    )
    second = service.ingest(
        scope,
        DocumentInput(
            collection_id=collection_id,
            source_type=" WEB ",
            source_ref=" https://example.test/page ",
            content="First section.\n\nSecond section.",
            metadata={"language": "en", "status": 200},
        ),
    )

    assert first.created is True
    assert second.created is False
    assert second.document_id == first.document_id
    assert second.version_id == first.version_id
    assert second.chunk_count == 0

    with scoped_session(session_factory, scope) as session:
        document_count = session.scalar(select(func.count()).select_from(DocumentRecord))
        version_count = session.scalar(select(func.count()).select_from(DocumentVersionRecord))
        chunk_count = session.scalar(select(func.count()).select_from(ChunkRecord))

    assert document_count == 1
    assert version_count == 1
    assert chunk_count == first.chunk_count


@pytest.mark.integration
def test_changed_document_state_creates_a_new_version(
    session_factory: sessionmaker[Session],
) -> None:
    scope, collection_id = _create_scope_with_collection(session_factory, "versioned")
    service = IngestionService(session_factory)

    first = service.ingest(
        scope,
        DocumentInput(
            collection_id=collection_id,
            source_type="web",
            source_ref="https://example.test/page",
            content="Original content",
        ),
    )
    second = service.ingest(
        scope,
        DocumentInput(
            collection_id=collection_id,
            source_type="web",
            source_ref="https://example.test/page",
            content="Updated content",
        ),
    )

    assert first.created is True
    assert second.created is True
    assert second.document_id == first.document_id
    assert second.version_id != first.version_id

    with scoped_session(session_factory, scope) as session:
        version_count = session.scalar(select(func.count()).select_from(DocumentVersionRecord))

    assert version_count == 2


@pytest.mark.integration
def test_ingestion_cannot_use_collection_from_another_scope(
    session_factory: sessionmaker[Session],
) -> None:
    first_scope, _ = _create_scope_with_collection(session_factory, "owner")
    _, foreign_collection_id = _create_scope_with_collection(session_factory, "foreign")
    service = IngestionService(session_factory)

    with pytest.raises(CollectionNotFound):
        service.ingest(
            first_scope,
            DocumentInput(
                collection_id=foreign_collection_id,
                source_type="web",
                source_ref="https://example.test/",
                content="Private content",
            ),
        )

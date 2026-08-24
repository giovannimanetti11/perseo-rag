from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from perseo_rag.config import Settings
from perseo_rag.security import ScopeContext, scoped_session
from perseo_rag.storage import build_engine, build_session_factory
from perseo_rag.storage.schema import (
    ChunkRecord,
    CollectionRecord,
    DocumentRecord,
    DocumentVersionRecord,
    ScopeRecord,
)


@dataclass(frozen=True, slots=True)
class SeededScope:
    scope_id: UUID
    collection_id: UUID
    content: str


@pytest.fixture(scope="module")
def session_factory() -> sessionmaker[Session]:
    engine = build_engine(Settings(_env_file=None))
    factory = build_session_factory(engine)
    yield factory
    engine.dispose()


def _seed_scope(
    session_factory: sessionmaker[Session],
    label: str,
    source_ref: str,
) -> SeededScope:
    scope_id = uuid4()
    collection_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    content = f"{label} private scan result"

    with scoped_session(session_factory, ScopeContext(scope_id)) as session:
        session.add_all(
            [
                ScopeRecord(id=scope_id, slug=f"{label}-{scope_id.hex}", name=label),
                CollectionRecord(id=collection_id, scope_id=scope_id, name="Default"),
                DocumentRecord(
                    id=document_id,
                    scope_id=scope_id,
                    collection_id=collection_id,
                    source_type="web",
                    source_ref=source_ref,
                ),
                DocumentVersionRecord(
                    id=version_id,
                    scope_id=scope_id,
                    document_id=document_id,
                    content_hash=uuid4().hex,
                    metadata_json={"kind": "scan"},
                ),
                ChunkRecord(
                    id=uuid4(),
                    scope_id=scope_id,
                    document_version_id=version_id,
                    position=0,
                    content=content,
                ),
            ]
        )

    return SeededScope(scope_id=scope_id, collection_id=collection_id, content=content)


@pytest.mark.integration
def test_same_source_is_isolated_by_scope(
    session_factory: sessionmaker[Session],
) -> None:
    source_ref = "https://example.test/"
    first = _seed_scope(session_factory, "first", source_ref)
    second = _seed_scope(session_factory, "second", source_ref)

    with scoped_session(session_factory, ScopeContext(first.scope_id)) as session:
        contents = set(session.scalars(select(ChunkRecord.content)))

    assert contents == {first.content}
    assert second.content not in contents

    with scoped_session(session_factory, ScopeContext(second.scope_id)) as session:
        contents = set(session.scalars(select(ChunkRecord.content)))

    assert contents == {second.content}
    assert first.content not in contents


@pytest.mark.integration
def test_unscoped_queries_return_no_private_rows(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        assert list(session.scalars(select(ChunkRecord.id))) == []


@pytest.mark.integration
def test_cross_scope_parent_links_are_rejected(
    session_factory: sessionmaker[Session],
) -> None:
    first = _seed_scope(session_factory, "parent-first", "https://same.example/")
    second = _seed_scope(session_factory, "parent-second", "https://same.example/")

    with (
        pytest.raises(IntegrityError),
        scoped_session(session_factory, ScopeContext(first.scope_id)) as session,
    ):
        session.add(
            DocumentRecord(
                id=uuid4(),
                scope_id=first.scope_id,
                collection_id=second.collection_id,
                source_type="web",
                source_ref="https://same.example/cross-scope",
            )
        )

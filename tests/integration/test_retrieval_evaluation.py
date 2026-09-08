import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict
from uuid import UUID, uuid4

import pytest
from sqlalchemy import and_, select
from sqlalchemy.orm import Session, sessionmaker

from perseo_rag.config import Settings
from perseo_rag.embeddings import EmbeddingService
from perseo_rag.evaluation import EvaluationCase, evaluate_cases
from perseo_rag.ingestion import DocumentInput, IngestionService
from perseo_rag.retrieval import HybridRetriever
from perseo_rag.security import ScopeContext, scoped_session
from perseo_rag.storage import build_engine, build_session_factory
from perseo_rag.storage.schema import (
    ChunkRecord,
    CollectionRecord,
    DocumentRecord,
    DocumentVersionRecord,
    ScopeRecord,
)


class FixtureDocument(TypedDict):
    key: str
    source_ref: str
    content: str


class FixtureQuery(TypedDict):
    query: str
    relevant: list[str]


class FixtureData(TypedDict):
    documents: list[FixtureDocument]
    queries: list[FixtureQuery]


@dataclass(frozen=True, slots=True)
class EvaluationProvider:
    key: str = "evaluation-fixture"
    dimensions: int = 3

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        normalized = text.lower()
        return [
            float("canonical" in normalized),
            float("performance" in normalized or "loading" in normalized),
            float("accessibility" in normalized or "keyboard" in normalized),
        ]


@pytest.fixture(scope="module")
def session_factory() -> sessionmaker[Session]:
    engine = build_engine(Settings(_env_file=None))
    factory = build_session_factory(engine)
    yield factory
    engine.dispose()


def _load_fixture() -> FixtureData:
    path = Path("eval/synthetic_cases.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _create_scope(
    session_factory: sessionmaker[Session],
    label: str,
) -> tuple[ScopeContext, UUID]:
    scope = ScopeContext(uuid4())
    collection_id = uuid4()

    with scoped_session(session_factory, scope) as session:
        session.add(ScopeRecord(id=scope.id, slug=f"{label}-{scope.id.hex}", name=label))
        session.flush()
        session.add(CollectionRecord(id=collection_id, scope_id=scope.id, name="Evaluation"))

    return scope, collection_id


def _ingest_fixture(
    session_factory: sessionmaker[Session],
    scope: ScopeContext,
    collection_id: UUID,
    documents: list[FixtureDocument],
) -> None:
    service = IngestionService(session_factory)
    for document in documents:
        service.ingest(
            scope,
            DocumentInput(
                collection_id=collection_id,
                source_type="fixture",
                source_ref=document["source_ref"],
                content=document["content"],
            ),
        )


def _chunk_ids_by_source(
    session_factory: sessionmaker[Session],
    scope: ScopeContext,
) -> dict[str, UUID]:
    with scoped_session(session_factory, scope) as session:
        statement = (
            select(DocumentRecord.source_ref, ChunkRecord.id)
            .join(
                DocumentVersionRecord,
                and_(
                    DocumentVersionRecord.scope_id == DocumentRecord.scope_id,
                    DocumentVersionRecord.document_id == DocumentRecord.id,
                ),
            )
            .join(
                ChunkRecord,
                and_(
                    ChunkRecord.scope_id == DocumentVersionRecord.scope_id,
                    ChunkRecord.document_version_id == DocumentVersionRecord.id,
                ),
            )
            .where(DocumentRecord.scope_id == scope.id)
        )
        return dict(session.execute(statement))


@pytest.mark.integration
def test_hybrid_retrieval_meets_synthetic_quality_and_isolation_baseline(
    session_factory: sessionmaker[Session],
) -> None:
    fixture = _load_fixture()
    owner_scope, owner_collection = _create_scope(session_factory, "evaluation-owner")
    foreign_scope, foreign_collection = _create_scope(session_factory, "evaluation-foreign")

    _ingest_fixture(session_factory, owner_scope, owner_collection, fixture["documents"])
    _ingest_fixture(session_factory, foreign_scope, foreign_collection, fixture["documents"])

    provider = EvaluationProvider()
    embeddings = EmbeddingService(session_factory)
    embeddings.index_scope(owner_scope, provider)
    embeddings.index_scope(foreign_scope, provider)

    owner_chunks = _chunk_ids_by_source(session_factory, owner_scope)
    foreign_chunks = frozenset(_chunk_ids_by_source(session_factory, foreign_scope).values())
    retriever = HybridRetriever(session_factory)
    cases: list[EvaluationCase] = []

    documents_by_key = {document["key"]: document for document in fixture["documents"]}
    for query in fixture["queries"]:
        hits = retriever.search(owner_scope, query["query"], provider, limit=3)
        relevant = frozenset(
            owner_chunks[documents_by_key[key]["source_ref"]] for key in query["relevant"]
        )
        cases.append(
            EvaluationCase(
                name=query["query"],
                ranked_chunk_ids=tuple(hit.chunk_id for hit in hits),
                relevant_chunk_ids=relevant,
                forbidden_chunk_ids=foreign_chunks,
            )
        )

    summary = evaluate_cases(cases, k=3)

    assert summary.recall_at_k == 1.0
    assert summary.mean_reciprocal_rank == 1.0
    assert summary.leakage_rate == 0.0
    assert summary.leaking_cases == 0

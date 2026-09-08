from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    scope_id: UUID
    chunk_id: UUID
    document_version_id: UUID
    document_id: UUID
    source_ref: str
    content: str
    score: float


@dataclass(frozen=True, slots=True)
class HybridRetrievalHit:
    scope_id: UUID
    chunk_id: UUID
    document_version_id: UUID
    document_id: UUID
    source_ref: str
    content: str
    score: float
    lexical_rank: int | None
    dense_rank: int | None

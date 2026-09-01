from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    chunk_id: UUID
    document_version_id: UUID
    document_id: UUID
    source_ref: str
    content: str
    score: float

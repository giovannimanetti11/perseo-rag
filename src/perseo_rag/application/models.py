from dataclasses import dataclass
from uuid import UUID

from perseo_rag.generation.models import GroundedAnswer


@dataclass(frozen=True, slots=True)
class QueryInput:
    question: str
    collection_id: UUID | None = None
    limit: int = 10


@dataclass(frozen=True, slots=True)
class QueryResult:
    answer: GroundedAnswer
    evidence_count: int
    context_chars: int
    context_truncated: bool

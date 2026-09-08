from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class EvidenceCitation:
    id: str
    scope_id: UUID
    chunk_id: UUID
    document_version_id: UUID
    document_id: UUID
    source_ref: str


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    citation: EvidenceCitation
    content: str
    retrieval_score: float
    truncated: bool


@dataclass(frozen=True, slots=True)
class GroundedContext:
    evidence: tuple[EvidenceItem, ...]
    total_chars: int
    truncated: bool

    @property
    def citation_ids(self) -> frozenset[str]:
        return frozenset(item.citation.id for item in self.evidence)

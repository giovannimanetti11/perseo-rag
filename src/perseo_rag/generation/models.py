from dataclasses import dataclass
from enum import StrEnum

from perseo_rag.grounding.models import EvidenceCitation


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"


class AbstentionReason(StrEnum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class GenerationEvidence:
    citation_id: str
    source_ref: str
    content: str


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    question: str
    evidence: tuple[GenerationEvidence, ...]


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    text: str
    citation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    status: AnswerStatus
    text: str | None
    citations: tuple[EvidenceCitation, ...]
    abstention_reason: AbstentionReason | None

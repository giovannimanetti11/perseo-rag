from dataclasses import dataclass
from uuid import UUID

import pytest

from perseo_rag.generation import (
    AbstentionReason,
    AnswerStatus,
    EvidencePolicy,
    GenerationOutput,
    GenerationRequest,
    GenerationService,
)
from perseo_rag.grounding import EvidenceCitation, EvidenceItem, GroundedContext


@dataclass
class RecordingProvider:
    key: str = "fixture-provider"
    calls: int = 0
    last_request: GenerationRequest | None = None

    def generate(self, request: GenerationRequest) -> GenerationOutput:
        self.calls += 1
        self.last_request = request
        return GenerationOutput(
            text="Grounded answer",
            citation_ids=("S1",),
        )


def _context() -> GroundedContext:
    citation = EvidenceCitation(
        id="S1",
        scope_id=UUID(int=1),
        chunk_id=UUID(int=2),
        document_version_id=UUID(int=3),
        document_id=UUID(int=4),
        source_ref="fixture:source",
    )
    content = "Retrieved evidence"
    return GroundedContext(
        evidence=(
            EvidenceItem(
                citation=citation,
                content=content,
                retrieval_score=0.5,
                truncated=False,
            ),
        ),
        total_chars=len(content),
        truncated=False,
    )


def test_service_abstains_before_calling_provider() -> None:
    provider = RecordingProvider()
    service = GenerationService(
        evidence_policy=EvidencePolicy(min_items=2),
    )

    answer = service.answer("question", _context(), provider)

    assert answer.status is AnswerStatus.ABSTAINED
    assert answer.text is None
    assert answer.citations == ()
    assert answer.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE
    assert provider.calls == 0


def test_service_sends_structured_evidence_to_provider() -> None:
    provider = RecordingProvider()

    answer = GenerationService().answer("  question  ", _context(), provider)

    assert answer.status is AnswerStatus.ANSWERED
    assert answer.text == "Grounded answer"
    assert answer.abstention_reason is None
    assert [citation.id for citation in answer.citations] == ["S1"]
    assert provider.calls == 1
    assert provider.last_request is not None
    assert provider.last_request.question == "question"
    assert provider.last_request.evidence[0].citation_id == "S1"
    assert provider.last_request.evidence[0].content == "Retrieved evidence"


def test_service_rejects_empty_question() -> None:
    with pytest.raises(ValueError):
        GenerationService().answer("  ", _context(), RecordingProvider())


def test_service_rejects_empty_provider_key() -> None:
    provider = RecordingProvider(key=" ")

    with pytest.raises(ValueError):
        GenerationService().answer("question", _context(), provider)

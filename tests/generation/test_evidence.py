from uuid import UUID

import pytest

from perseo_rag.generation import EvidencePolicy
from perseo_rag.grounding import EvidenceCitation, EvidenceItem, GroundedContext


def _context(content: str = "Retrieved evidence") -> GroundedContext:
    citation = EvidenceCitation(
        id="S1",
        scope_id=UUID(int=1),
        chunk_id=UUID(int=2),
        document_version_id=UUID(int=3),
        document_id=UUID(int=4),
        source_ref="fixture:source",
    )
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


def test_evidence_policy_accepts_sufficient_context() -> None:
    assert EvidencePolicy().is_sufficient(_context()) is True


def test_evidence_policy_rejects_context_below_threshold() -> None:
    policy = EvidencePolicy(min_items=2, min_total_chars=20)

    assert policy.is_sufficient(_context()) is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_items": 0},
        {"min_total_chars": 0},
    ],
)
def test_evidence_policy_rejects_invalid_limits(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        EvidencePolicy(**kwargs)

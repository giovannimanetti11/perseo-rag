from uuid import UUID

import pytest

from perseo_rag.generation import (
    GenerationOutput,
    InvalidCitation,
    InvalidGenerationOutput,
    validate_generation_output,
)
from perseo_rag.grounding import EvidenceCitation, EvidenceItem, GroundedContext


def _context() -> GroundedContext:
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
                content="Retrieved evidence",
                retrieval_score=0.5,
                truncated=False,
            ),
        ),
        total_chars=len("Retrieved evidence"),
        truncated=False,
    )


def test_generation_output_resolves_context_citations() -> None:
    text, citations = validate_generation_output(
        GenerationOutput(text=" Answer ", citation_ids=("S1", "S1")),
        _context(),
    )

    assert text == "Answer"
    assert len(citations) == 1
    assert citations[0].id == "S1"


def test_generation_output_rejects_unknown_citations() -> None:
    with pytest.raises(InvalidCitation):
        validate_generation_output(
            GenerationOutput(text="Answer", citation_ids=("S2",)),
            _context(),
        )


def test_generation_output_requires_citations() -> None:
    with pytest.raises(InvalidCitation):
        validate_generation_output(
            GenerationOutput(text="Answer", citation_ids=()),
            _context(),
        )


def test_generation_output_rejects_empty_text() -> None:
    with pytest.raises(InvalidGenerationOutput):
        validate_generation_output(
            GenerationOutput(text="   ", citation_ids=("S1",)),
            _context(),
        )

from collections.abc import Sequence

from perseo_rag.generation.models import GenerationOutput
from perseo_rag.grounding.models import EvidenceCitation, GroundedContext


class InvalidGenerationOutput(ValueError):
    pass


class InvalidCitation(InvalidGenerationOutput):
    pass


def validate_generation_output(
    output: GenerationOutput,
    context: GroundedContext,
) -> tuple[str, tuple[EvidenceCitation, ...]]:
    text = output.text.strip()
    if not text:
        raise InvalidGenerationOutput("generated text must not be empty")

    citations_by_id = {item.citation.id: item.citation for item in context.evidence}
    citation_ids = _deduplicate(output.citation_ids)

    if not citation_ids:
        raise InvalidCitation("grounded answers must include at least one citation")

    unknown = [citation_id for citation_id in citation_ids if citation_id not in citations_by_id]
    if unknown:
        raise InvalidCitation(f"unknown citation ids: {', '.join(unknown)}")

    citations = tuple(citations_by_id[citation_id] for citation_id in citation_ids)
    return text, citations


def _deduplicate(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))

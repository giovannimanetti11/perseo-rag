from perseo_rag.generation.citations import validate_generation_output
from perseo_rag.generation.evidence import EvidencePolicy
from perseo_rag.generation.models import (
    AbstentionReason,
    AnswerStatus,
    GenerationEvidence,
    GenerationRequest,
    GroundedAnswer,
)
from perseo_rag.generation.provider import GenerationProvider
from perseo_rag.grounding.models import GroundedContext


class GenerationService:
    def __init__(
        self,
        *,
        evidence_policy: EvidencePolicy | None = None,
    ) -> None:
        self._evidence_policy = evidence_policy or EvidencePolicy()

    def answer(
        self,
        question: str,
        context: GroundedContext,
        provider: GenerationProvider,
    ) -> GroundedAnswer:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        if not provider.key.strip():
            raise ValueError("provider key must not be empty")

        if not self._evidence_policy.is_sufficient(context):
            return GroundedAnswer(
                status=AnswerStatus.ABSTAINED,
                text=None,
                citations=(),
                abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
            )

        request = GenerationRequest(
            question=normalized_question,
            evidence=tuple(
                GenerationEvidence(
                    citation_id=item.citation.id,
                    source_ref=item.citation.source_ref,
                    content=item.content,
                )
                for item in context.evidence
            ),
        )
        output = provider.generate(request)
        text, citations = validate_generation_output(output, context)

        return GroundedAnswer(
            status=AnswerStatus.ANSWERED,
            text=text,
            citations=citations,
            abstention_reason=None,
        )

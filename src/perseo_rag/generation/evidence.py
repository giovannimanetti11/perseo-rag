from dataclasses import dataclass

from perseo_rag.grounding.models import GroundedContext


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    min_items: int = 1
    min_total_chars: int = 1

    def __post_init__(self) -> None:
        if self.min_items < 1:
            raise ValueError("min_items must be positive")
        if self.min_total_chars < 1:
            raise ValueError("min_total_chars must be positive")

    def is_sufficient(self, context: GroundedContext) -> bool:
        return len(context.evidence) >= self.min_items and context.total_chars >= self.min_total_chars

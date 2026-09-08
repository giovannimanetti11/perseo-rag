from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    name: str
    ranked_chunk_ids: tuple[UUID, ...]
    relevant_chunk_ids: frozenset[UUID]
    forbidden_chunk_ids: frozenset[UUID] = frozenset()


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    k: int
    cases: int
    recall_at_k: float
    mean_reciprocal_rank: float
    leakage_rate: float
    leaking_cases: int

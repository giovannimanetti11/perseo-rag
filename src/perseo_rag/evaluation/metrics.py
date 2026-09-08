from collections.abc import Sequence

from perseo_rag.evaluation.models import EvaluationCase, EvaluationSummary


def evaluate_cases(
    cases: Sequence[EvaluationCase],
    *,
    k: int = 5,
) -> EvaluationSummary:
    if k < 1:
        raise ValueError("k must be positive")
    if not cases:
        raise ValueError("at least one evaluation case is required")
    if any(not case.relevant_chunk_ids for case in cases):
        raise ValueError("each evaluation case must define relevant chunks")

    recall_total = 0.0
    reciprocal_rank_total = 0.0
    leaked_hits = 0
    retrieved_hits = 0
    leaking_cases = 0

    for case in cases:
        top_k = case.ranked_chunk_ids[:k]
        relevant_retrieved = case.relevant_chunk_ids.intersection(top_k)
        recall_total += len(relevant_retrieved) / len(case.relevant_chunk_ids)

        first_relevant_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(case.ranked_chunk_ids, start=1)
                if chunk_id in case.relevant_chunk_ids
            ),
            None,
        )
        if first_relevant_rank is not None:
            reciprocal_rank_total += 1.0 / first_relevant_rank

        case_leaks = sum(chunk_id in case.forbidden_chunk_ids for chunk_id in top_k)
        leaked_hits += case_leaks
        retrieved_hits += len(top_k)
        leaking_cases += int(case_leaks > 0)

    count = len(cases)
    leakage_rate = leaked_hits / retrieved_hits if retrieved_hits else 0.0

    return EvaluationSummary(
        k=k,
        cases=count,
        recall_at_k=recall_total / count,
        mean_reciprocal_rank=reciprocal_rank_total / count,
        leakage_rate=leakage_rate,
        leaking_cases=leaking_cases,
    )

from uuid import UUID

import pytest

from perseo_rag.evaluation import EvaluationCase, evaluate_cases


def test_evaluation_reports_recall_mrr_and_zero_leakage() -> None:
    relevant = UUID(int=1)
    distractor = UUID(int=2)
    forbidden = UUID(int=3)

    summary = evaluate_cases(
        [
            EvaluationCase(
                name="first",
                ranked_chunk_ids=(distractor, relevant),
                relevant_chunk_ids=frozenset({relevant}),
                forbidden_chunk_ids=frozenset({forbidden}),
            ),
            EvaluationCase(
                name="second",
                ranked_chunk_ids=(relevant,),
                relevant_chunk_ids=frozenset({relevant}),
                forbidden_chunk_ids=frozenset({forbidden}),
            ),
        ],
        k=2,
    )

    assert summary.cases == 2
    assert summary.recall_at_k == 1.0
    assert summary.mean_reciprocal_rank == 0.75
    assert summary.leakage_rate == 0.0
    assert summary.leaking_cases == 0


def test_evaluation_counts_forbidden_hits() -> None:
    relevant = UUID(int=1)
    forbidden = UUID(int=2)

    summary = evaluate_cases(
        [
            EvaluationCase(
                name="leak",
                ranked_chunk_ids=(forbidden, relevant),
                relevant_chunk_ids=frozenset({relevant}),
                forbidden_chunk_ids=frozenset({forbidden}),
            )
        ],
        k=2,
    )

    assert summary.leakage_rate == 0.5
    assert summary.leaking_cases == 1


@pytest.mark.parametrize(
    ("cases", "k"),
    [
        ([], 5),
        (
            [
                EvaluationCase(
                    name="empty-relevance",
                    ranked_chunk_ids=(),
                    relevant_chunk_ids=frozenset(),
                )
            ],
            5,
        ),
        (
            [
                EvaluationCase(
                    name="invalid-k",
                    ranked_chunk_ids=(UUID(int=1),),
                    relevant_chunk_ids=frozenset({UUID(int=1)}),
                )
            ],
            0,
        ),
    ],
)
def test_evaluation_rejects_invalid_inputs(cases: list[EvaluationCase], k: int) -> None:
    with pytest.raises(ValueError):
        evaluate_cases(cases, k=k)

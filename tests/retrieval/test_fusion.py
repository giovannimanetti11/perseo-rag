from uuid import UUID

import pytest

from perseo_rag.retrieval.fusion import InconsistentRetrievalHit, reciprocal_rank_fusion
from perseo_rag.retrieval.models import RetrievalHit


SCOPE_ID = UUID(int=100)


def _hit(value: int, score: float, *, scope_id: UUID = SCOPE_ID) -> RetrievalHit:
    identifier = UUID(int=value)
    return RetrievalHit(
        scope_id=scope_id,
        chunk_id=identifier,
        document_version_id=identifier,
        document_id=identifier,
        source_ref=f"fixture:{value}",
        content=f"content {value}",
        score=score,
    )


def test_rrf_rewards_results_present_in_both_rankings() -> None:
    shared = _hit(1, 0.4)
    lexical_only = _hit(2, 0.9)
    dense_only = _hit(3, 0.99)

    hits = reciprocal_rank_fusion(
        [shared, lexical_only],
        [dense_only, shared],
        rank_constant=60,
    )

    assert hits[0].chunk_id == shared.chunk_id
    assert hits[0].scope_id == SCOPE_ID
    assert hits[0].lexical_rank == 1
    assert hits[0].dense_rank == 2
    assert hits[1].dense_rank == 1
    assert hits[1].lexical_rank is None


def test_rrf_uses_rank_not_source_score_scale() -> None:
    lexical_first = _hit(1, 0.0001)
    lexical_second = _hit(2, 1000.0)

    hits = reciprocal_rank_fusion([lexical_first, lexical_second], [])

    assert hits[0].chunk_id == lexical_first.chunk_id


def test_rrf_rejects_conflicting_provenance_for_same_chunk() -> None:
    shared = _hit(1, 0.5)
    conflicting = _hit(1, 0.7, scope_id=UUID(int=200))

    with pytest.raises(InconsistentRetrievalHit):
        reciprocal_rank_fusion([shared], [conflicting])


@pytest.mark.parametrize(
    ("limit", "rank_constant"),
    [
        (0, 60),
        (10, 0),
    ],
)
def test_rrf_rejects_invalid_configuration(limit: int, rank_constant: int) -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], [], limit=limit, rank_constant=rank_constant)

from uuid import UUID

import pytest

from perseo_rag.grounding import CrossScopeEvidence, GroundedContextBuilder
from perseo_rag.retrieval import HybridRetrievalHit
from perseo_rag.security import ScopeContext


def _hit(
    value: int,
    *,
    scope_id: UUID,
    content: str,
    score: float = 0.1,
) -> HybridRetrievalHit:
    identifier = UUID(int=value)
    return HybridRetrievalHit(
        scope_id=scope_id,
        chunk_id=identifier,
        document_version_id=identifier,
        document_id=identifier,
        source_ref=f"fixture:{value}",
        content=content,
        score=score,
        lexical_rank=1,
        dense_rank=1,
    )


def test_builder_assigns_stable_citation_ids() -> None:
    scope = ScopeContext(UUID(int=1))
    hits = [
        _hit(10, scope_id=scope.id, content="First evidence"),
        _hit(20, scope_id=scope.id, content="Second evidence"),
    ]

    context = GroundedContextBuilder().build(scope, hits)

    assert [item.citation.id for item in context.evidence] == ["S1", "S2"]
    assert context.citation_ids == frozenset({"S1", "S2"})
    assert context.total_chars == len("First evidenceSecond evidence")
    assert context.truncated is False


def test_builder_rejects_cross_scope_evidence() -> None:
    scope = ScopeContext(UUID(int=1))
    foreign_hit = _hit(10, scope_id=UUID(int=2), content="Foreign evidence")

    with pytest.raises(CrossScopeEvidence):
        GroundedContextBuilder().build(scope, [foreign_hit])


def test_builder_respects_character_budget() -> None:
    scope = ScopeContext(UUID(int=1))
    hit = _hit(10, scope_id=scope.id, content="abcdefghij")

    context = GroundedContextBuilder(max_chars=5).build(scope, [hit])

    assert len(context.evidence) == 1
    assert context.evidence[0].content == "abcde"
    assert context.evidence[0].truncated is True
    assert context.total_chars == 5
    assert context.truncated is True


def test_builder_deduplicates_chunks() -> None:
    scope = ScopeContext(UUID(int=1))
    hit = _hit(10, scope_id=scope.id, content="Evidence")

    context = GroundedContextBuilder().build(scope, [hit, hit])

    assert len(context.evidence) == 1
    assert context.truncated is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_chars": 0},
        {"max_items": 0},
    ],
)
def test_builder_rejects_invalid_limits(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        GroundedContextBuilder(**kwargs)

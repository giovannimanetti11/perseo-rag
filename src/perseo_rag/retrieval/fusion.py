from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from perseo_rag.retrieval.models import HybridRetrievalHit, RetrievalHit


class InconsistentRetrievalHit(ValueError):
    pass


@dataclass(slots=True)
class _FusionEntry:
    hit: RetrievalHit
    score: float = 0.0
    lexical_rank: int | None = None
    dense_rank: int | None = None


def reciprocal_rank_fusion(
    lexical_hits: Sequence[RetrievalHit],
    dense_hits: Sequence[RetrievalHit],
    *,
    limit: int = 10,
    rank_constant: int = 60,
) -> tuple[HybridRetrievalHit, ...]:
    if limit < 1:
        raise ValueError("limit must be positive")

    if rank_constant < 1:
        raise ValueError("rank_constant must be positive")

    entries: dict[UUID, _FusionEntry] = {}
    _accumulate(entries, lexical_hits, rank_constant, "lexical")
    _accumulate(entries, dense_hits, rank_constant, "dense")

    ranked = sorted(
        entries.values(),
        key=lambda entry: (-entry.score, str(entry.hit.chunk_id)),
    )

    return tuple(
        HybridRetrievalHit(
            scope_id=entry.hit.scope_id,
            chunk_id=entry.hit.chunk_id,
            document_version_id=entry.hit.document_version_id,
            document_id=entry.hit.document_id,
            source_ref=entry.hit.source_ref,
            content=entry.hit.content,
            score=entry.score,
            lexical_rank=entry.lexical_rank,
            dense_rank=entry.dense_rank,
        )
        for entry in ranked[:limit]
    )


def _accumulate(
    entries: dict[UUID, _FusionEntry],
    hits: Sequence[RetrievalHit],
    rank_constant: int,
    channel: str,
) -> None:
    for rank, hit in enumerate(hits, start=1):
        entry = entries.get(hit.chunk_id)
        if entry is None:
            entry = _FusionEntry(hit=hit)
            entries[hit.chunk_id] = entry
        elif not _same_provenance(entry.hit, hit):
            raise InconsistentRetrievalHit(f"conflicting provenance for chunk {hit.chunk_id}")

        entry.score += 1.0 / (rank_constant + rank)

        if channel == "lexical":
            entry.lexical_rank = rank
        else:
            entry.dense_rank = rank


def _same_provenance(first: RetrievalHit, second: RetrievalHit) -> bool:
    return (
        first.scope_id == second.scope_id
        and first.document_version_id == second.document_version_id
        and first.document_id == second.document_id
        and first.source_ref == second.source_ref
        and first.content == second.content
    )

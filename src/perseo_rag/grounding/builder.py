from collections.abc import Sequence
from uuid import UUID

from perseo_rag.grounding.models import EvidenceCitation, EvidenceItem, GroundedContext
from perseo_rag.retrieval.models import HybridRetrievalHit
from perseo_rag.security import ScopeContext


class CrossScopeEvidence(ValueError):
    pass


class GroundedContextBuilder:
    def __init__(self, *, max_chars: int = 12000, max_items: int = 8) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        if max_items < 1:
            raise ValueError("max_items must be positive")

        self._max_chars = max_chars
        self._max_items = max_items

    def build(
        self,
        scope: ScopeContext,
        hits: Sequence[HybridRetrievalHit],
    ) -> GroundedContext:
        evidence: list[EvidenceItem] = []
        seen_chunks: set[UUID] = set()
        used_chars = 0
        truncated = False

        for hit in hits:
            if hit.scope_id != scope.id:
                raise CrossScopeEvidence(f"chunk {hit.chunk_id} belongs to another scope")

            if hit.chunk_id in seen_chunks:
                continue

            if len(evidence) >= self._max_items or used_chars >= self._max_chars:
                truncated = True
                break

            remaining = self._max_chars - used_chars
            item_truncated = len(hit.content) > remaining
            excerpt = hit.content[:remaining].rstrip() if item_truncated else hit.content

            if not excerpt:
                truncated = True
                break

            citation = EvidenceCitation(
                id=f"S{len(evidence) + 1}",
                scope_id=hit.scope_id,
                chunk_id=hit.chunk_id,
                document_version_id=hit.document_version_id,
                document_id=hit.document_id,
                source_ref=hit.source_ref,
            )
            evidence.append(
                EvidenceItem(
                    citation=citation,
                    content=excerpt,
                    retrieval_score=hit.score,
                    truncated=item_truncated,
                )
            )
            seen_chunks.add(hit.chunk_id)
            used_chars += len(excerpt)

            if item_truncated:
                truncated = True
                break

        if len(evidence) < len({hit.chunk_id for hit in hits}):
            truncated = True

        return GroundedContext(
            evidence=tuple(evidence),
            total_chars=used_chars,
            truncated=truncated,
        )

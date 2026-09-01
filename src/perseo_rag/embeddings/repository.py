from dataclasses import dataclass
from uuid import UUID, uuid4

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from perseo_rag.storage.schema import ChunkEmbeddingRecord, ChunkRecord


@dataclass(frozen=True, slots=True)
class PendingChunk:
    id: UUID
    content: str


class SqlAlchemyEmbeddingRepository:
    def __init__(self, session: Session, scope_id: UUID) -> None:
        self._session = session
        self._scope_id = scope_id

    def pending_chunks(self, provider_key: str, limit: int) -> tuple[PendingChunk, ...]:
        indexed = exists().where(
            ChunkEmbeddingRecord.scope_id == ChunkRecord.scope_id,
            ChunkEmbeddingRecord.chunk_id == ChunkRecord.id,
            ChunkEmbeddingRecord.provider_key == provider_key,
        )
        statement = (
            select(ChunkRecord.id, ChunkRecord.content)
            .where(
                ChunkRecord.scope_id == self._scope_id,
                ~indexed,
            )
            .order_by(ChunkRecord.id)
            .limit(limit)
        )
        return tuple(
            PendingChunk(id=chunk_id, content=content)
            for chunk_id, content in self._session.execute(statement)
        )

    def add_embeddings(
        self,
        provider_key: str,
        dimensions: int,
        chunks: tuple[PendingChunk, ...],
        vectors: tuple[tuple[float, ...], ...],
    ) -> None:
        self._session.add_all(
            ChunkEmbeddingRecord(
                id=uuid4(),
                scope_id=self._scope_id,
                chunk_id=chunk.id,
                provider_key=provider_key,
                dimensions=dimensions,
                embedding=list(vector),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        )

"""Store chunk embeddings as provider-specific derived indexes."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SCOPE_SETTING = "NULLIF(current_setting('perseo.scope_id', true), '')::uuid"


def upgrade() -> None:
    op.drop_column("chunks", "embedding")
    op.create_unique_constraint(
        "uq_chunks_scope_id_id",
        "chunks",
        ["scope_id", "id"],
    )

    op.create_table(
        "chunk_embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("scope_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("provider_key", sa.String(length=120), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(), nullable=False),
        sa.ForeignKeyConstraint(
            ["scope_id", "chunk_id"],
            ["chunks.scope_id", "chunks.id"],
            name="fk_chunk_embeddings_scope_chunk",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chunk_embeddings"),
        sa.UniqueConstraint(
            "scope_id",
            "chunk_id",
            "provider_key",
            name="uq_chunk_embeddings_provider",
        ),
    )
    op.create_index(
        "ix_chunk_embeddings_scope_provider",
        "chunk_embeddings",
        ["scope_id", "provider_key"],
    )

    predicate = f"scope_id = {_SCOPE_SETTING}"
    op.execute('ALTER TABLE "chunk_embeddings" ENABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE "chunk_embeddings" FORCE ROW LEVEL SECURITY')
    op.execute(
        'CREATE POLICY scope_isolation ON "chunk_embeddings" '
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_table("chunk_embeddings")
    op.drop_constraint("uq_chunks_scope_id_id", "chunks", type_="unique")
    op.add_column("chunks", sa.Column("embedding", Vector(), nullable=True))

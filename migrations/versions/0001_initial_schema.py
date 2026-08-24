"""Create the initial scoped document schema."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SCOPE_SETTING = "NULLIF(current_setting('perseo.scope_id', true), '')::uuid"


def _enable_scope_policy(table: str, column: str) -> None:
    predicate = f"{column} = {_SCOPE_SETTING}"
    op.execute(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY')
    op.execute(f'ALTER TABLE "{table}" FORCE ROW LEVEL SECURITY')
    op.execute(
        f'CREATE POLICY scope_isolation ON "{table}" USING ({predicate}) WITH CHECK ({predicate})'
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "scopes",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("slug", sa.String(length=100), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_scopes"),
        sa.UniqueConstraint("slug", name="uq_scopes_slug"),
    )

    op.create_table(
        "collections",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("scope_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.ForeignKeyConstraint(
            ["scope_id"],
            ["scopes.id"],
            name="fk_collections_scope_id_scopes",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_collections"),
        sa.UniqueConstraint("scope_id", "id", name="uq_collections_scope_id_id"),
        sa.UniqueConstraint("scope_id", "name", name="uq_collections_scope_id_name"),
    )
    op.create_index("ix_collections_scope_id", "collections", ["scope_id"])

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("scope_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("source_ref", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["scope_id", "collection_id"],
            ["collections.scope_id", "collections.id"],
            name="fk_documents_scope_collection",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_documents"),
        sa.UniqueConstraint("scope_id", "id", name="uq_documents_scope_id_id"),
        sa.UniqueConstraint(
            "scope_id",
            "collection_id",
            "source_type",
            "source_ref",
            name="uq_documents_source",
        ),
    )
    op.create_index("ix_documents_scope_id", "documents", ["scope_id"])

    op.create_table(
        "document_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("scope_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["scope_id", "document_id"],
            ["documents.scope_id", "documents.id"],
            name="fk_document_versions_scope_document",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_document_versions"),
        sa.UniqueConstraint(
            "scope_id",
            "id",
            name="uq_document_versions_scope_id_id",
        ),
        sa.UniqueConstraint(
            "scope_id",
            "document_id",
            "content_hash",
            name="uq_document_versions_content",
        ),
    )
    op.create_index(
        "ix_document_versions_scope_id",
        "document_versions",
        ["scope_id"],
    )

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("scope_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(), nullable=True),
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('simple'::regconfig, content)",
                persisted=True,
            ),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["scope_id", "document_version_id"],
            ["document_versions.scope_id", "document_versions.id"],
            name="fk_chunks_scope_document_version",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chunks"),
        sa.UniqueConstraint(
            "scope_id",
            "document_version_id",
            "position",
            name="uq_chunks_position",
        ),
    )
    op.create_index("ix_chunks_scope_id", "chunks", ["scope_id"])
    op.create_index(
        "ix_chunks_search_vector",
        "chunks",
        ["search_vector"],
        postgresql_using="gin",
    )

    _enable_scope_policy("scopes", "id")
    _enable_scope_policy("collections", "scope_id")
    _enable_scope_policy("documents", "scope_id")
    _enable_scope_policy("document_versions", "scope_id")
    _enable_scope_policy("chunks", "scope_id")


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("document_versions")
    op.drop_table("documents")
    op.drop_table("collections")
    op.drop_table("scopes")

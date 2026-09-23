"""Track the current version of each document."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SCOPE_SETTING = "NULLIF(current_setting('perseo.scope_id', true), '')::uuid"


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_document_versions_scope_document_id",
        "document_versions",
        ["scope_id", "document_id", "id"],
    )

    op.create_table(
        "document_heads",
        sa.Column("scope_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["scope_id", "document_id", "version_id"],
            ["document_versions.scope_id", "document_versions.document_id", "document_versions.id"],
            name="fk_document_heads_scope_document_version",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("scope_id", "document_id", name="pk_document_heads"),
    )
    op.create_index("ix_document_heads_scope_id", "document_heads", ["scope_id"])

    op.execute(
        """
        INSERT INTO document_heads (scope_id, document_id, version_id)
        SELECT scope_id, document_id, id
        FROM (
            SELECT
                scope_id,
                document_id,
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY scope_id, document_id
                    ORDER BY created_at DESC, id DESC
                ) AS row_number
            FROM document_versions
        ) AS ranked
        WHERE row_number = 1
        """
    )

    predicate = f"scope_id = {_SCOPE_SETTING}"
    op.execute('ALTER TABLE "document_heads" ENABLE ROW LEVEL SECURITY')
    op.execute('ALTER TABLE "document_heads" FORCE ROW LEVEL SECURITY')
    op.execute(
        'CREATE POLICY scope_isolation ON "document_heads" '
        f"USING ({predicate}) WITH CHECK ({predicate})"
    )


def downgrade() -> None:
    op.drop_table("document_heads")
    op.drop_constraint(
        "uq_document_versions_scope_document_id",
        "document_versions",
        type_="unique",
    )

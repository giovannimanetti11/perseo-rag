from datetime import datetime
from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class ScopeRecord(Base):
    __tablename__ = "scopes"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)


class CollectionRecord(Base):
    __tablename__ = "collections"
    __table_args__ = (
        UniqueConstraint("scope_id", "id", name="uq_collections_scope_id_id"),
        UniqueConstraint("scope_id", "name", name="uq_collections_scope_id_name"),
        Index("ix_collections_scope_id", "scope_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    scope_id: Mapped[UUID] = mapped_column(
        ForeignKey("scopes.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)


class DocumentRecord(Base):
    __tablename__ = "documents"
    __table_args__ = (
        ForeignKeyConstraint(
            ["scope_id", "collection_id"],
            ["collections.scope_id", "collections.id"],
            name="fk_documents_scope_collection",
            ondelete="CASCADE",
        ),
        UniqueConstraint("scope_id", "id", name="uq_documents_scope_id_id"),
        UniqueConstraint(
            "scope_id",
            "collection_id",
            "source_type",
            "source_ref",
            name="uq_documents_source",
        ),
        Index("ix_documents_scope_id", "scope_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    scope_id: Mapped[UUID] = mapped_column(nullable=False)
    collection_id: Mapped[UUID] = mapped_column(nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source_ref: Mapped[str] = mapped_column(Text, nullable=False)


class DocumentVersionRecord(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["scope_id", "document_id"],
            ["documents.scope_id", "documents.id"],
            name="fk_document_versions_scope_document",
            ondelete="CASCADE",
        ),
        UniqueConstraint("scope_id", "id", name="uq_document_versions_scope_id_id"),
        UniqueConstraint(
            "scope_id",
            "document_id",
            "content_hash",
            name="uq_document_versions_content",
        ),
        Index("ix_document_versions_scope_id", "scope_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    scope_id: Mapped[UUID] = mapped_column(nullable=False)
    document_id: Mapped[UUID] = mapped_column(nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )


class ChunkRecord(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        ForeignKeyConstraint(
            ["scope_id", "document_version_id"],
            ["document_versions.scope_id", "document_versions.id"],
            name="fk_chunks_scope_document_version",
            ondelete="CASCADE",
        ),
        UniqueConstraint("scope_id", "id", name="uq_chunks_scope_id_id"),
        UniqueConstraint(
            "scope_id",
            "document_version_id",
            "position",
            name="uq_chunks_position",
        ),
        Index("ix_chunks_scope_id", "scope_id"),
        Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    scope_id: Mapped[UUID] = mapped_column(nullable=False)
    document_version_id: Mapped[UUID] = mapped_column(nullable=False)
    position: Mapped[int] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple'::regconfig, content)", persisted=True),
    )


class ChunkEmbeddingRecord(Base):
    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        ForeignKeyConstraint(
            ["scope_id", "chunk_id"],
            ["chunks.scope_id", "chunks.id"],
            name="fk_chunk_embeddings_scope_chunk",
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "scope_id",
            "chunk_id",
            "provider_key",
            name="uq_chunk_embeddings_provider",
        ),
        Index(
            "ix_chunk_embeddings_scope_provider",
            "scope_id",
            "provider_key",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    scope_id: Mapped[UUID] = mapped_column(nullable=False)
    chunk_id: Mapped[UUID] = mapped_column(nullable=False)
    provider_key: Mapped[str] = mapped_column(String(120), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)

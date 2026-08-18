from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Tenant:
    id: UUID
    slug: str
    name: str


@dataclass(frozen=True, slots=True)
class Collection:
    id: UUID
    tenant_id: UUID
    name: str


@dataclass(frozen=True, slots=True)
class Document:
    id: UUID
    tenant_id: UUID
    collection_id: UUID
    source_type: str
    source_ref: str


@dataclass(frozen=True, slots=True)
class DocumentVersion:
    id: UUID
    tenant_id: UUID
    document_id: UUID
    content_hash: str
    created_at: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    id: UUID
    tenant_id: UUID
    document_version_id: UUID
    position: int
    content: str

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass(frozen=True, slots=True)
class DocumentInput:
    collection_id: UUID
    source_type: str
    source_ref: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NormalizedDocument:
    collection_id: UUID
    source_type: str
    source_ref: str
    content: str
    metadata: dict[str, Any]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document_id: UUID
    version_id: UUID
    created: bool
    chunk_count: int

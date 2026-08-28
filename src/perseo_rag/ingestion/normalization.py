import hashlib
import json
from typing import Any

from perseo_rag.ingestion.models import DocumentInput, NormalizedDocument


class InvalidDocument(ValueError):
    pass


def normalize_document(document: DocumentInput) -> NormalizedDocument:
    source_type = document.source_type.strip().lower()
    source_ref = document.source_ref.strip()
    content = normalize_text(document.content)
    metadata = _normalize_metadata(document.metadata)

    if not source_type:
        raise InvalidDocument("source_type must not be empty")

    if not source_ref:
        raise InvalidDocument("source_ref must not be empty")

    if not content:
        raise InvalidDocument("content must not be empty")

    fingerprint = _fingerprint(content, metadata)

    return NormalizedDocument(
        collection_id=document.collection_id,
        source_type=source_type,
        source_ref=source_ref,
        content=content,
        metadata=metadata,
        fingerprint=fingerprint,
    )


def normalize_text(content: str) -> str:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = (line.rstrip() for line in normalized.split("\n"))
    return "\n".join(lines).strip()


def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        serialized = json.dumps(
            metadata,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise InvalidDocument("metadata must be JSON serializable") from exc

    value = json.loads(serialized)
    if not isinstance(value, dict):
        raise InvalidDocument("metadata must be a JSON object")

    return value


def _fingerprint(content: str, metadata: dict[str, Any]) -> str:
    payload = json.dumps(
        {"content": content, "metadata": metadata},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()

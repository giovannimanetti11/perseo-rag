from uuid import uuid4

import pytest

from perseo_rag.ingestion import DocumentInput, InvalidDocument, normalize_document


def test_equivalent_content_and_metadata_have_the_same_fingerprint() -> None:
    collection_id = uuid4()
    first = normalize_document(
        DocumentInput(
            collection_id=collection_id,
            source_type=" Web ",
            source_ref=" https://example.test/page ",
            content="Heading  \r\n\r\nBody\r\n",
            metadata={"language": "en", "status": 200},
        )
    )
    second = normalize_document(
        DocumentInput(
            collection_id=collection_id,
            source_type="web",
            source_ref="https://example.test/page",
            content="Heading\n\nBody",
            metadata={"status": 200, "language": "en"},
        )
    )

    assert first.content == "Heading\n\nBody"
    assert first.source_type == "web"
    assert first.source_ref == "https://example.test/page"
    assert first.fingerprint == second.fingerprint


def test_metadata_changes_the_fingerprint() -> None:
    collection_id = uuid4()
    first = normalize_document(
        DocumentInput(
            collection_id=collection_id,
            source_type="web",
            source_ref="https://example.test/",
            content="Same content",
            metadata={"status": 200},
        )
    )
    second = normalize_document(
        DocumentInput(
            collection_id=collection_id,
            source_type="web",
            source_ref="https://example.test/",
            content="Same content",
            metadata={"status": 304},
        )
    )

    assert first.fingerprint != second.fingerprint


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_type", " "),
        ("source_ref", " "),
        ("content", "\r\n  "),
    ],
)
def test_required_fields_cannot_be_empty(field: str, value: str) -> None:
    data = {
        "collection_id": uuid4(),
        "source_type": "web",
        "source_ref": "https://example.test/",
        "content": "content",
    }
    data[field] = value

    with pytest.raises(InvalidDocument):
        normalize_document(DocumentInput(**data))  # type: ignore[arg-type]


def test_metadata_must_be_json_serializable() -> None:
    with pytest.raises(InvalidDocument):
        normalize_document(
            DocumentInput(
                collection_id=uuid4(),
                source_type="web",
                source_ref="https://example.test/",
                content="content",
                metadata={"unsupported": object()},
            )
        )

import pytest

from perseo_rag.ingestion import TextChunker


def test_chunker_prefers_natural_breaks() -> None:
    chunker = TextChunker(max_chars=24, min_break_ratio=0.5)
    chunks = chunker.split("First paragraph.\n\nSecond paragraph is longer.")

    assert chunks == ("First paragraph.", "Second paragraph is", "longer.")


def test_chunker_hard_splits_long_unbroken_content() -> None:
    chunker = TextChunker(max_chars=5)

    assert chunker.split("abcdefghijk") == ("abcde", "fghij", "k")


def test_empty_content_produces_no_chunks() -> None:
    assert TextChunker().split("") == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_chars": 0},
        {"min_break_ratio": 0},
        {"min_break_ratio": 1.1},
    ],
)
def test_invalid_chunker_configuration_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        TextChunker(**kwargs)  # type: ignore[arg-type]

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from perseo_rag.embeddings import EmbeddingService, InvalidEmbedding
from perseo_rag.embeddings.validation import validate_provider, validate_vectors


@dataclass(frozen=True, slots=True)
class Provider:
    key: str = "test-v1"
    dimensions: int = 2

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[float(len(text)), 1.0] for text in texts]


@dataclass(frozen=True, slots=True)
class InvalidProvider:
    key: str = "test-v1"
    dimensions: int = 2

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[1.0] for _ in texts]


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError):
        EmbeddingService(None, batch_size=0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "provider",
    [
        Provider(key=" "),
        Provider(dimensions=0),
    ],
)
def test_invalid_provider_configuration_is_rejected(provider: Provider) -> None:
    with pytest.raises(InvalidEmbedding):
        validate_provider(provider)


def test_invalid_dimensions_are_rejected() -> None:
    with pytest.raises(InvalidEmbedding):
        validate_vectors(
            InvalidProvider().embed(["content"]),
            expected_count=1,
            dimensions=2,
        )


def test_non_finite_embeddings_are_rejected() -> None:
    with pytest.raises(InvalidEmbedding):
        validate_vectors(
            [[float("nan"), 1.0]],
            expected_count=1,
            dimensions=2,
        )


def test_embedding_count_must_match_input_count() -> None:
    with pytest.raises(InvalidEmbedding):
        validate_vectors([], expected_count=1, dimensions=2)

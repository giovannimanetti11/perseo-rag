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


class AsymmetricProvider:
    key = "asymmetric-v1"
    dimensions = 2

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        raise AssertionError("legacy embed must not be used when asymmetric methods exist")

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[2.0, float(len(text))] for text in texts]

    def embed_query(self, query: str) -> Sequence[float]:
        return [3.0, float(len(query))]


def test_retrieval_aware_provider_uses_document_embedding_method() -> None:
    from perseo_rag.embeddings.validation import embed_documents

    vectors = embed_documents(AsymmetricProvider(), ["one", "two"])

    assert vectors == ((2.0, 3.0), (2.0, 3.0))


def test_retrieval_aware_provider_uses_query_embedding_method() -> None:
    from perseo_rag.embeddings.validation import embed_query

    vector = embed_query(AsymmetricProvider(), "query")

    assert vector == (3.0, 5.0)

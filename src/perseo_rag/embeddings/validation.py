import math
from collections.abc import Sequence

from perseo_rag.embeddings.provider import EmbeddingProvider


class InvalidEmbedding(ValueError):
    pass


def validate_provider(provider: EmbeddingProvider) -> None:
    if not provider.key.strip():
        raise InvalidEmbedding("provider key must not be empty")

    if provider.dimensions < 1:
        raise InvalidEmbedding("provider dimensions must be positive")


def validate_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
    dimensions: int,
) -> tuple[tuple[float, ...], ...]:
    if len(vectors) != expected_count:
        raise InvalidEmbedding("provider returned an unexpected number of embeddings")

    validated: list[tuple[float, ...]] = []
    for vector in vectors:
        values = tuple(float(value) for value in vector)
        if len(values) != dimensions:
            raise InvalidEmbedding("provider returned an embedding with invalid dimensions")

        if not all(math.isfinite(value) for value in values):
            raise InvalidEmbedding("provider returned a non-finite embedding")

        validated.append(values)

    return tuple(validated)


def embed_query(provider: EmbeddingProvider, query: str) -> tuple[float, ...]:
    validate_provider(provider)
    vectors = validate_vectors(
        provider.embed([query]),
        expected_count=1,
        dimensions=provider.dimensions,
    )
    return vectors[0]

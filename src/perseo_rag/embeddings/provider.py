from collections.abc import Sequence
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Backward-compatible embedding provider contract."""

    @property
    def key(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class AsymmetricEmbeddingProvider(Protocol):
    """Optional retrieval-aware embedding capability.

    Providers can expose distinct document/query embedding methods while legacy
    providers continue to work through ``EmbeddingProvider.embed``.
    """

    @property
    def key(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    def embed_query(self, query: str) -> Sequence[float]: ...
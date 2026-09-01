from collections.abc import Sequence
from typing import Protocol


class EmbeddingProvider(Protocol):
    @property
    def key(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingIndexResult:
    provider_key: str
    indexed_chunks: int

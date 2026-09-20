from collections.abc import Sequence
from dataclasses import dataclass

from perseo_rag.generation import GenerationOutput, GenerationRequest
from perseo_rag.providers import (
    ProviderTimeout,
    RetryPolicy,
    RetryingEmbeddingProvider,
    RetryingGenerationProvider,
)


@dataclass
class EmbeddingProviderFixture:
    key: str = "embedding-fixture"
    dimensions: int = 2
    calls: int = 0

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls += 1
        if self.calls == 1:
            raise ProviderTimeout("timeout")
        return [[1.0, 0.0] for _ in texts]


@dataclass
class GenerationProviderFixture:
    key: str = "generation-fixture"
    calls: int = 0

    def generate(self, request: GenerationRequest) -> GenerationOutput:
        self.calls += 1
        if self.calls == 1:
            raise ProviderTimeout("timeout")
        return GenerationOutput(text="answer", citation_ids=("S1",))


def test_retrying_embedding_provider_preserves_provider_identity() -> None:
    provider = EmbeddingProviderFixture()
    wrapped = RetryingEmbeddingProvider(
        provider,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay=0),
        sleeper=lambda _: None,
    )

    vectors = wrapped.embed(["one", "two"])

    assert wrapped.key == provider.key
    assert wrapped.dimensions == provider.dimensions
    assert vectors == [[1.0, 0.0], [1.0, 0.0]]
    assert provider.calls == 2


def test_retrying_generation_provider_retries_transient_failure() -> None:
    provider = GenerationProviderFixture()
    wrapped = RetryingGenerationProvider(
        provider,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay=0),
        sleeper=lambda _: None,
    )
    request = GenerationRequest(question="question", evidence=())

    output = wrapped.generate(request)

    assert wrapped.key == provider.key
    assert output.text == "answer"
    assert provider.calls == 2

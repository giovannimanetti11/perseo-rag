from collections.abc import Sequence

from perseo_rag.embeddings.provider import EmbeddingProvider
from perseo_rag.generation.models import GenerationOutput, GenerationRequest
from perseo_rag.generation.provider import GenerationProvider
from perseo_rag.providers.retry import RetryPolicy, Sleeper, call_with_retry


class RetryingEmbeddingProvider:
    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        retry_policy: RetryPolicy | None = None,
        sleeper: Sleeper | None = None,
    ) -> None:
        self._provider = provider
        self._retry_policy = retry_policy or RetryPolicy()
        self._sleeper = sleeper

    @property
    def key(self) -> str:
        return self._provider.key

    @property
    def dimensions(self) -> int:
        return self._provider.dimensions

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        operation = lambda: self._provider.embed(texts)
        if self._sleeper is None:
            return call_with_retry(operation, policy=self._retry_policy)

        return call_with_retry(
            operation,
            policy=self._retry_policy,
            sleeper=self._sleeper,
        )


class RetryingGenerationProvider:
    def __init__(
        self,
        provider: GenerationProvider,
        *,
        retry_policy: RetryPolicy | None = None,
        sleeper: Sleeper | None = None,
    ) -> None:
        self._provider = provider
        self._retry_policy = retry_policy or RetryPolicy()
        self._sleeper = sleeper

    @property
    def key(self) -> str:
        return self._provider.key

    def generate(self, request: GenerationRequest) -> GenerationOutput:
        operation = lambda: self._provider.generate(request)
        if self._sleeper is None:
            return call_with_retry(operation, policy=self._retry_policy)

        return call_with_retry(
            operation,
            policy=self._retry_policy,
            sleeper=self._sleeper,
        )

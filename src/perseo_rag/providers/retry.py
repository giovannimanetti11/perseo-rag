import time
from collections.abc import Callable
from dataclasses import dataclass

from perseo_rag.providers.errors import (
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)

Sleeper = Callable[[float], None]
TransientProviderError = ProviderTimeout | ProviderUnavailable | ProviderRateLimited


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_delay: float = 0.25
    multiplier: float = 2.0
    max_delay: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.initial_delay < 0:
            raise ValueError("initial_delay must not be negative")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least 1")
        if self.max_delay < 0:
            raise ValueError("max_delay must not be negative")

    def delay_for_retry(self, retry_number: int) -> float:
        if retry_number < 1:
            raise ValueError("retry_number must be positive")

        delay = self.initial_delay * (self.multiplier ** (retry_number - 1))
        return min(delay, self.max_delay)


def call_with_retry[T](
    operation: Callable[[], T],
    *,
    policy: RetryPolicy,
    sleeper: Sleeper = time.sleep,
) -> T:
    attempt = 1

    while True:
        try:
            return operation()
        except (ProviderTimeout, ProviderUnavailable, ProviderRateLimited):
            if attempt >= policy.max_attempts:
                raise

            sleeper(policy.delay_for_retry(attempt))
            attempt += 1

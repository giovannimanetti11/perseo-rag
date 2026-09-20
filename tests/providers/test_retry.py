from dataclasses import dataclass

import pytest

from perseo_rag.providers import (
    ProviderProtocolError,
    ProviderTimeout,
    ProviderUnavailable,
    RetryPolicy,
    call_with_retry,
)


@dataclass
class Operation:
    failures: list[Exception]
    calls: int = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return "ok"


def test_retry_succeeds_after_transient_failures() -> None:
    delays: list[float] = []
    operation = Operation(failures=[ProviderTimeout("timeout"), ProviderUnavailable("unavailable")])

    result = call_with_retry(
        operation,
        policy=RetryPolicy(
            max_attempts=3,
            initial_delay=0.1,
            multiplier=2,
            max_delay=1,
        ),
        sleeper=delays.append,
    )

    assert result == "ok"
    assert operation.calls == 3
    assert delays == [0.1, 0.2]


def test_retry_does_not_retry_protocol_errors() -> None:
    delays: list[float] = []
    operation = Operation(failures=[ProviderProtocolError("invalid response")])

    with pytest.raises(ProviderProtocolError):
        call_with_retry(
            operation,
            policy=RetryPolicy(max_attempts=3),
            sleeper=delays.append,
        )

    assert operation.calls == 1
    assert delays == []


def test_retry_stops_after_max_attempts() -> None:
    operation = Operation(
        failures=[
            ProviderTimeout("timeout"),
            ProviderTimeout("timeout"),
            ProviderTimeout("timeout"),
        ]
    )

    with pytest.raises(ProviderTimeout):
        call_with_retry(
            operation,
            policy=RetryPolicy(max_attempts=2, initial_delay=0),
            sleeper=lambda _: None,
        )

    assert operation.calls == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"initial_delay": -1},
        {"multiplier": 0.5},
        {"max_delay": -1},
    ],
)
def test_retry_policy_rejects_invalid_configuration(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)  # type: ignore[arg-type]


def test_retry_number_must_be_positive() -> None:
    with pytest.raises(ValueError):
        RetryPolicy().delay_for_retry(0)

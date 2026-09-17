"""Client-side pacing against provider per-minute limits (architecture §7.2, plan Appendix B).

Two token buckets per model — one for requests per minute, one for tokens per minute — refilled
continuously. `acquire()` blocks (via the injected `sleep`) until both buckets can cover the call
instead of letting the provider answer 429. The clock and sleep are injectable so tests can
simulate bursts without waiting.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from rag.core.config import Pacing


class PacingError(RuntimeError):
    """A single call can never fit under the configured per-minute ceiling."""


class TokenBucket:
    def __init__(self, capacity: float, refill_per_second: float, clock: Callable[[], float]):
        if capacity <= 0 or refill_per_second <= 0:
            raise ValueError("capacity and refill rate must be positive")
        self.capacity = float(capacity)
        self.refill_per_second = float(refill_per_second)
        self._clock = clock
        self._tokens = float(capacity)
        self._last = clock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_second)

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens

    def seconds_until(self, amount: float) -> float:
        """How long to wait before `amount` can be consumed (0 if it can be consumed now)."""
        if amount > self.capacity:
            raise PacingError(
                f"requested {amount:.0f} exceeds bucket capacity {self.capacity:.0f} per minute"
            )
        self._refill()
        if self._tokens >= amount:
            return 0.0
        return (amount - self._tokens) / self.refill_per_second

    def consume(self, amount: float) -> None:
        self._refill()
        self._tokens -= amount  # may dip slightly negative if an actual usage exceeded the estimate


class RateLimiter:
    """Combined RPM + TPM limiter for one model id."""

    def __init__(
        self,
        pacing: Pacing,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        max_wait_seconds: float = 120.0,
    ):
        self.pacing = pacing
        self._sleep = sleep
        self._lock = threading.Lock()
        self.max_wait_seconds = max_wait_seconds
        self.requests = TokenBucket(pacing.rpm, pacing.rpm / 60.0, clock)
        self.tokens = TokenBucket(pacing.tpm, pacing.tpm / 60.0, clock)
        self.output_tokens = (
            TokenBucket(pacing.otpm, pacing.otpm / 60.0, clock) if pacing.otpm else None
        )
        self.total_wait_seconds = 0.0
        self.waits = 0

    def acquire(self, estimated_tokens: int, requested_output_tokens: int = 0) -> float:
        """Block until one request, `estimated_tokens` and the requested output budget fit.

        Returns seconds waited.
        """
        waited = 0.0
        with self._lock:
            while True:
                wait = max(
                    self.requests.seconds_until(1), self.tokens.seconds_until(estimated_tokens)
                )
                if self.output_tokens is not None and requested_output_tokens:
                    wait = max(wait, self.output_tokens.seconds_until(requested_output_tokens))
                if wait <= 0:
                    break
                if waited + wait > self.max_wait_seconds:
                    raise PacingError(
                        f"pacing wait would exceed {self.max_wait_seconds:.0f}s "
                        f"(rpm={self.pacing.rpm}, tpm={self.pacing.tpm}, est={estimated_tokens})"
                    )
                self._sleep(wait)
                waited += wait
            self.requests.consume(1)
            self.tokens.consume(estimated_tokens)
            if self.output_tokens is not None and requested_output_tokens:
                self.output_tokens.consume(requested_output_tokens)
        if waited > 0:
            self.waits += 1
            self.total_wait_seconds += waited
        return waited

    def reconcile(self, estimated_tokens: int, actual_tokens: int) -> None:
        """Charge the difference once the provider reports real usage."""
        delta = actual_tokens - estimated_tokens
        if delta > 0:
            with self._lock:
                self.tokens.consume(delta)

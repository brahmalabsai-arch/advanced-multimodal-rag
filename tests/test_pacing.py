"""Phase 0 tests — pacing: a burst beyond the RPM limit waits instead of calling."""

from __future__ import annotations

import pytest

from rag.core.config import Pacing
from rag.core.pacing import PacingError, RateLimiter, TokenBucket


def test_burst_beyond_rpm_waits(fake_clock) -> None:
    limiter = RateLimiter(Pacing(rpm=3, tpm=100_000), clock=fake_clock, sleep=fake_clock.sleep)

    for _ in range(3):
        assert limiter.acquire(100) == 0.0
    assert fake_clock.sleeps == []

    waited = limiter.acquire(100)  # 4th request within the same minute
    assert waited > 0
    assert fake_clock.sleeps == [pytest.approx(waited)]
    # Refill rate is rpm/60 per second, so one request costs 20 s at rpm=3.
    assert waited == pytest.approx(20.0)
    assert limiter.waits == 1


def test_tokens_per_minute_limit_waits(fake_clock) -> None:
    limiter = RateLimiter(Pacing(rpm=1000, tpm=6000), clock=fake_clock, sleep=fake_clock.sleep)
    assert limiter.acquire(4000) == 0.0
    waited = limiter.acquire(4000)  # only 2000 left; need 2000 more at 100 tokens/s
    assert waited == pytest.approx(20.0)


def test_bucket_refills_over_time(fake_clock) -> None:
    limiter = RateLimiter(Pacing(rpm=3, tpm=100_000), clock=fake_clock, sleep=fake_clock.sleep)
    for _ in range(3):
        limiter.acquire(10)
    fake_clock.now += 60  # a full minute passes with no calls
    assert limiter.acquire(10) == 0.0
    assert fake_clock.sleeps == []


def test_single_call_over_tpm_ceiling_is_an_error(fake_clock) -> None:
    limiter = RateLimiter(Pacing(rpm=30, tpm=8000), clock=fake_clock, sleep=fake_clock.sleep)
    with pytest.raises(PacingError, match="exceeds"):
        limiter.acquire(9000)


def test_wait_cap(fake_clock) -> None:
    limiter = RateLimiter(
        Pacing(rpm=1, tpm=100_000), clock=fake_clock, sleep=fake_clock.sleep, max_wait_seconds=30
    )
    limiter.acquire(1)
    with pytest.raises(PacingError, match="would exceed"):
        limiter.acquire(1)  # would need 60 s


def test_reconcile_charges_underestimate(fake_clock) -> None:
    limiter = RateLimiter(Pacing(rpm=1000, tpm=1000), clock=fake_clock, sleep=fake_clock.sleep)
    limiter.acquire(100)
    limiter.reconcile(estimated_tokens=100, actual_tokens=600)
    assert limiter.tokens.available == pytest.approx(400)


def test_token_bucket_rejects_bad_config(fake_clock) -> None:
    with pytest.raises(ValueError):
        TokenBucket(0, 1, fake_clock)

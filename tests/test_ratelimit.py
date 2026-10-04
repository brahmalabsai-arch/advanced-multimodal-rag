"""Per-address rate limit and in-flight cap for the public deploy (D-70)."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.api.ratelimit import (
    ASK_PATH,
    KEY_TEST_PATH,
    RateLimiter,
    RateLimitMiddleware,
    client_address,
)
from rag.core.config import RateLimitConfig, load_app_config


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _limiter(**kw) -> tuple[RateLimiter, Clock]:  # noqa: ANN003
    clock = Clock()
    return RateLimiter(RateLimitConfig(**kw), clock=clock), clock


# ------------------------------------------------------------------------------- windows


def test_the_minute_window_refuses_the_seventh_question_and_says_when_to_retry() -> None:
    limiter, clock = _limiter(ask_per_minute=6, ask_per_hour=60)
    for _ in range(6):
        assert limiter.check("1.2.3.4", ASK_PATH) is None
        clock.t += 5
    refused = limiter.check("1.2.3.4", ASK_PATH)
    assert refused is not None
    rule, retry = refused
    # the first question was 30 s ago, so the window frees a slot in 30 s
    assert rule.name == "ask_minute" and retry == 30
    clock.t += 30
    assert limiter.check("1.2.3.4", ASK_PATH) is None


def test_a_refused_request_does_not_extend_the_wait() -> None:
    limiter, clock = _limiter(ask_per_minute=2, ask_per_hour=0)
    limiter.check("a", ASK_PATH)
    limiter.check("a", ASK_PATH)
    for _ in range(10):  # hammering while refused must not push the window out
        assert limiter.check("a", ASK_PATH) is not None
    clock.t += 60
    assert limiter.check("a", ASK_PATH) is None


def test_the_hour_window_holds_when_the_minute_window_is_clear() -> None:
    limiter, clock = _limiter(ask_per_minute=6, ask_per_hour=10)
    for _ in range(10):
        assert limiter.check("a", ASK_PATH) is None
        clock.t += 61  # always under the minute limit
    rule, retry = limiter.check("a", ASK_PATH)  # type: ignore[misc]
    assert rule.name == "ask_hour" and 0 < retry <= 3600


def test_addresses_and_routes_are_counted_separately() -> None:
    limiter, _ = _limiter(ask_per_minute=1, ask_per_hour=0, key_test_per_minute=1)
    assert limiter.check("a", ASK_PATH) is None
    assert limiter.check("a", ASK_PATH) is not None
    assert limiter.check("b", ASK_PATH) is None  # another visitor is unaffected
    assert limiter.check("a", KEY_TEST_PATH) is None  # the key check has its own budget
    assert limiter.check("a", KEY_TEST_PATH) is not None
    assert limiter.check("a", "/api/pages/3") is None  # unlimited routes stay unlimited


def test_memory_is_bounded_and_idle_addresses_are_forgotten() -> None:
    limiter, clock = _limiter(max_clients=10)
    for i in range(25):
        limiter.check(f"10.0.0.{i}", ASK_PATH)
    assert limiter.tracked == 10
    clock.t += 3601
    limiter.prune()
    assert limiter.tracked == 0


# ------------------------------------------------------------------------- client address


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"cf-connecting-ip": "203.0.113.7"}, "203.0.113.7"),
        ({"true-client-ip": "203.0.113.8"}, "203.0.113.8"),
        ({"cf-connecting-ip": "203.0.113.7", "true-client-ip": "198.51.100.1"}, "203.0.113.7"),
        # X-Forwarded-For is spoofable on Render (it appends, never replaces): ignored
        ({"x-forwarded-for": "1.1.1.1, 203.0.113.9"}, "testclient"),
        ({}, "testclient"),
    ],
)
def test_client_address_trusts_only_the_proxy_set_headers(
    headers: dict[str, str], expected: str
) -> None:
    from starlette.requests import Request

    scope = {
        "type": "http",
        "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
        "client": ("testclient", 50000),
    }
    assert client_address(Request(scope), ["cf-connecting-ip", "true-client-ip"]) == expected


# ------------------------------------------------------------------------------- middleware


def _app(limiter: RateLimiter | None, gate: asyncio.Event | None = None) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)
    app.state.rate_limiter = limiter

    @app.post(ASK_PATH)
    async def ask() -> dict:
        if gate is not None:
            await gate.wait()
        return {"ok": True}

    @app.post(KEY_TEST_PATH)
    async def key_test() -> dict:
        return {"ok": True}

    @app.get("/api/pages/{n}")
    async def page(n: int) -> dict:
        return {"n": n}

    return app


def test_the_middleware_answers_429_with_retry_after_and_a_reason() -> None:
    limiter, _ = _limiter(ask_per_minute=2, ask_per_hour=0)
    with TestClient(_app(limiter)) as c:
        visitor = {"cf-connecting-ip": "203.0.113.7"}
        assert c.post(ASK_PATH, headers=visitor).status_code == 200
        assert c.post(ASK_PATH, headers=visitor).status_code == 200
        r = c.post(ASK_PATH, headers=visitor)
        assert r.status_code == 429
        body = r.json()
        assert body["limit"] == "rate" and body["retry_after"] == 60
        assert r.headers["retry-after"] == "60"
        assert "2 questions a minute" in body["detail"]
        # a different visitor behind the same proxy is still served
        assert c.post(ASK_PATH, headers={"cf-connecting-ip": "203.0.113.8"}).status_code == 200
        # reads are never limited
        assert all(c.get("/api/pages/3").status_code == 200 for _ in range(20))


def test_without_a_limiter_the_middleware_is_a_pass_through() -> None:
    with TestClient(_app(None)) as c:
        assert all(c.post(ASK_PATH).status_code == 200 for _ in range(30))


def test_the_in_flight_cap_refuses_with_503_busy() -> None:
    limiter, _ = _limiter(max_in_flight=1, ask_per_minute=0, ask_per_hour=0)
    limiter.in_flight = 1  # one question already running
    with TestClient(_app(limiter)) as c:
        r = c.post(ASK_PATH)
        assert r.status_code == 503 and r.json()["limit"] == "busy"
        assert r.headers["retry-after"] == "5"
    limiter.in_flight = 0
    with TestClient(_app(limiter)) as c:
        assert c.post(ASK_PATH).status_code == 200
    assert limiter.in_flight == 0  # released after the response


def test_the_shipped_limits_are_the_documented_ones() -> None:
    cfg = load_app_config().server.rate_limit
    assert cfg.enabled
    assert (cfg.ask_per_minute, cfg.ask_per_hour, cfg.key_test_per_minute) == (6, 60, 5)
    assert cfg.max_in_flight == 2
    assert "x-forwarded-for" not in cfg.client_ip_headers

"""Per-visitor rate limit and an in-flight cap for the public deploy (D-43 → D-70).

The deployed instance has a tenth of one CPU. A question spends 1–4 s of that CPU on the
question embedding, hybrid search and context assembly before any model is called, so a single
client looping on `/api/ask` — or a handful of them — keeps it permanently busy and every other
visitor waits behind them. Bring-your-own-key means nobody can spend *our* quota; it does not stop
anyone from spending our CPU. Two limits close that:

1. **Per client address**, sliding windows: `/api/ask` 6 a minute and 60 an hour, `/api/key/test`
   5 a minute (it is also a free oracle for whether a leaked key is live). Over a limit → 429 with
   `Retry-After` and `{"limit": "rate", "retry_after": s}` so the page can say how long to wait.
2. **In flight**, all clients together: at most `max_in_flight` questions run at once. Over it →
   503 `{"limit": "busy"}`. Per-address limits cannot see many addresses at once; this can.

The visitor's address comes from the first trusted proxy header present. On Render the request
passes through Cloudflare, which *sets* `CF-Connecting-IP` / `True-Client-IP` (a client-supplied
value is overwritten), whereas Render only appends to `X-Forwarded-For`, so its first entry is
whatever the client wrote — never trusted here. Without those headers (localhost, CI) the socket
peer is used.

Active only when the app runs `PUBLIC_DEPLOY` with `server.rate_limit.enabled`; state is held in
memory, which is correct for the one worker on one instance this deploy runs.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from rag.core.config import RateLimitConfig
from rag.core.logging import get_logger

log = get_logger(__name__)

ASK_PATH = "/api/ask"
KEY_TEST_PATH = "/api/key/test"


@dataclass(frozen=True)
class Rule:
    name: str
    path: str
    limit: int
    window_s: float
    label: str  # "a minute" / "an hour", for the message


def rules_from(cfg: RateLimitConfig) -> list[Rule]:
    rules = [
        Rule("ask_minute", ASK_PATH, cfg.ask_per_minute, 60.0, "a minute"),
        Rule("ask_hour", ASK_PATH, cfg.ask_per_hour, 3600.0, "an hour"),
        Rule("key_minute", KEY_TEST_PATH, cfg.key_test_per_minute, 60.0, "a minute"),
    ]
    return [r for r in rules if r.limit > 0]


def client_address(request: Request, trusted_headers: list[str]) -> str:
    for header in trusted_headers:
        value = request.headers.get(header)
        if value:
            return value.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """Sliding windows per (address, rule), plus a global in-flight counter.

    Memory is bounded: at most `max_clients` addresses are tracked, least recently seen dropped
    first, and an address whose windows have all emptied is forgotten.
    """

    def __init__(
        self,
        cfg: RateLimitConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cfg = cfg
        self.rules = rules_from(cfg)
        self.clock = clock
        self.in_flight = 0
        self._hits: OrderedDict[str, dict[str, deque[float]]] = OrderedDict()

    def check(self, address: str, path: str) -> tuple[Rule, int] | None:
        """Record a request; return (rule, retry_after_seconds) when it must be refused.

        A refused request is not recorded, so a client that waits as told gets through.
        """
        rules = [r for r in self.rules if r.path == path]
        if not rules:
            return None
        now = self.clock()
        windows = self._hits.get(address)
        if windows is None:
            windows = self._hits[address] = {}
            while len(self._hits) > self.cfg.max_clients:
                self._hits.popitem(last=False)
        else:
            self._hits.move_to_end(address)
        for rule in rules:
            hits = windows.setdefault(rule.name, deque())
            while hits and now - hits[0] >= rule.window_s:
                hits.popleft()
            if len(hits) >= rule.limit:
                retry = max(1, math.ceil(rule.window_s - (now - hits[0])))
                return rule, retry
        for rule in rules:
            windows[rule.name].append(now)
        return None

    def prune(self) -> None:
        """Forget addresses with no hits left in any window (called opportunistically)."""
        now = self.clock()
        longest = max((r.window_s for r in self.rules), default=0.0)
        stale = [
            a
            for a, w in self._hits.items()
            if all(not d or now - d[-1] >= longest for d in w.values())
        ]
        for a in stale:
            del self._hits[a]

    @property
    def tracked(self) -> int:
        return len(self._hits)


def _refusal(status: int, detail: str, limit: str, retry_after: int) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"detail": detail, "limit": limit, "retry_after": retry_after},
        headers={"Retry-After": str(retry_after)},
    )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Applies `app.state.rate_limiter` when one is installed (public deploy); otherwise a
    pass-through, so localhost and the test suite are unaffected."""

    async def dispatch(self, request: Request, call_next) -> Response:  # noqa: ANN001
        limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
        if limiter is None or request.method != "POST":
            return await call_next(request)
        path = request.url.path
        if path not in {ASK_PATH, KEY_TEST_PATH}:
            return await call_next(request)

        address = client_address(request, limiter.cfg.client_ip_headers)
        refused = limiter.check(address, path)
        if refused is not None:
            rule, retry = refused
            what = "questions" if rule.path == ASK_PATH else "key checks"
            log.info("rate limit: %s refused (%s), retry in %ss", rule.name, what, retry)
            return _refusal(
                429,
                f"That is more than {rule.limit} {what} {rule.label} from your connection. "
                f"Try again in {retry} s.",
                "rate",
                retry,
            )
        if path != ASK_PATH:
            return await call_next(request)

        if limiter.in_flight >= limiter.cfg.max_in_flight:
            return _refusal(
                503,
                "The demo is answering other visitors' questions right now. "
                "Try again in a few seconds.",
                "busy",
                5,
            )
        limiter.in_flight += 1
        try:
            return await call_next(request)
        finally:
            limiter.in_flight -= 1
            if limiter.tracked > limiter.cfg.max_clients // 2:
                limiter.prune()


__all__ = ["RateLimitMiddleware", "RateLimiter", "Rule", "client_address", "rules_from"]

"""Bring-your-own-key: the caller's provider and key, per request (F2 deployment).

The localhost build reads `GROQ_API_KEY` from `.env` and every answer is paid for by whoever
runs the server. A public demo cannot work that way, so the deployed build takes the credential
from two request headers and builds a model client for that one request:

    X-Provider      groq | anthropic | gemini   (see `rag.llm.PROVIDERS`)
    X-Provider-Key  the caller's API key

Nothing about the key is stored. `LLMClient.for_request` builds a `Settings` copy whose only
credential is the supplied key — the environment's keys are blanked in that copy, so a caller
cannot fall back onto the server's quota by omitting theirs. The client, its rate limiters and
its chat-model objects live for the duration of the request and are then garbage.

**The key never reaches a log, a trace or the cache.** Three mechanisms together:

* `core/logging.py` masks the three provider key shapes in everything that goes through the
  logging handlers, registered secrets or not;
* `use_key()` puts the live key in a context variable for the duration of the request, and
  `redact()` — called by the trace writer and the usage ledger before they write a line —
  masks that exact string as well as the shapes, so a key of an unexpected shape is caught too;
* the cache stores the answer, never the request's credentials, and `versions.generator_model`
  includes the provider so one visitor's Groq answer is never served to another's Anthropic
  request (§5.7).

`POST /api/key/test` is the gate the UI calls before letting anyone in: one `max_tokens=1` call,
the cheapest proof that a key works, returning the model id it would answer with.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from rag.core.logging import get_logger, redact, use_key
from rag.llm import (
    DAILY_LIMIT_MESSAGE,
    PROVIDERS,
    LLMClient,
    LLMError,
    ProviderNotInstalledError,
    UnknownProviderError,
)

log = get_logger(__name__)
router = APIRouter(prefix="/api")

PROVIDER_HEADER = "X-Provider"
KEY_HEADER = "X-Provider-Key"
MIN_KEY_CHARS = 16
MAX_KEY_CHARS = 512


@dataclass(frozen=True)
class Credentials:
    provider: str
    key: str

    def client(self, **kwargs: Any) -> LLMClient:
        return LLMClient.for_request(self.provider, self.key, **kwargs)


def _clean_key(raw: str) -> str:
    """Reject a malformed key before spending a network round trip on it."""
    key = raw.strip()
    if len(key) < MIN_KEY_CHARS:
        raise HTTPException(status_code=401, detail="That key looks too short to be valid.")
    if len(key) > MAX_KEY_CHARS:
        raise HTTPException(status_code=401, detail="That key is longer than any key we accept.")
    if any(ch.isspace() or unicodedata.category(ch) == "Cc" for ch in key):
        raise HTTPException(
            status_code=401, detail="That key contains spaces or control characters."
        )
    if not re.fullmatch(r"[A-Za-z0-9._\-]+", key):
        raise HTTPException(
            status_code=401, detail="That key contains characters no provider uses."
        )
    return key


def credentials_from(request: Request) -> Credentials:
    """Read and validate the two headers, or raise the 4xx the UI knows how to render."""
    provider = (request.headers.get(PROVIDER_HEADER) or "").strip().lower()
    raw_key = request.headers.get(KEY_HEADER) or ""
    if not provider and not raw_key:
        raise HTTPException(
            status_code=401,
            detail="Add your model key to ask a question — it is used for this request only.",
        )
    if provider not in PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider {provider!r}. This demo serves "
            f"{', '.join(sorted(PROVIDERS))}.",
        )
    return Credentials(provider=provider, key=_clean_key(raw_key))


def http_error(exc: LLMError) -> HTTPException:
    """Map a model-layer failure onto the status codes the UI already handles.

    The provider's own message is passed through *redacted*, because some SDKs echo the request
    — including its Authorization header — back in the error text.
    """
    if isinstance(exc, UnknownProviderError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, ProviderNotInstalledError):
        return HTTPException(
            status_code=503,
            detail="This server cannot reach that provider. Try another one.",
        )
    status = getattr(exc, "status_code", None)
    detail = redact(str(exc))
    if status in {401, 403}:
        return HTTPException(status_code=401, detail="The provider rejected that key.")
    if status == 402:
        return HTTPException(
            status_code=402, detail="That key has no credit left with the provider."
        )
    if status == 429 and getattr(exc, "daily_quota", False):
        return HTTPException(
            status_code=429, detail=DAILY_LIMIT_MESSAGE, headers={"X-Limit": "daily"}
        )
    if status == 429:
        return HTTPException(
            status_code=429,
            detail="The provider is rate-limiting this key. Wait a minute and ask again.",
        )
    if status == 404:
        return HTTPException(
            status_code=502,
            detail="The provider does not offer the model this demo asks for on that key.",
        )
    log.warning("provider call failed: %s", detail[:300])
    return HTTPException(status_code=502, detail="The provider could not be reached.")


class KeyTestResponse(BaseModel):
    model: str
    provider: str


@router.post("/key/test", response_model=KeyTestResponse)
async def key_test(request: Request) -> KeyTestResponse:
    """One `max_tokens=1` call. 200 means the key works and names the model it answers with."""
    from fastapi.concurrency import run_in_threadpool

    creds = credentials_from(request)
    with use_key(creds.key):
        try:
            client = creds.client(pacing_enabled=False, max_attempts=2)
            model = await run_in_threadpool(client.smoke_test, request_id="key-test")
        except LLMError as exc:
            raise http_error(exc) from None
        except Exception as exc:  # never leak an internal traceback to a public endpoint
            log.error("key test failed unexpectedly: %s", redact(str(exc))[:300])
            raise HTTPException(status_code=502, detail="Could not check that key.") from None
    return KeyTestResponse(model=model, provider=creds.provider)


__all__ = [
    "KEY_HEADER",
    "PROVIDER_HEADER",
    "Credentials",
    "credentials_from",
    "http_error",
    "redact",
    "router",
    "use_key",
]

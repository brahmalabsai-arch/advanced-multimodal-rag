"""Swappable, paced, logged model layer (architecture §7, P2, P8; plan Phase 0).

Graph nodes ask for a *role* (`small`, `large`, `vision`) — never a provider or a model id.
Everything provider-specific stays inside this module:

* lazy provider imports — only `langchain-groq` is needed for `groq_build`;
* JSON handling — Groq JSON mode + Pydantic validation + one corrective retry (§7.2);
* client-side pacing — RPM/TPM token buckets sized from `models.yaml` (`core/pacing.py`);
* retries — exponential backoff with jitter on 429/5xx, honouring `retry-after`;
* usage ledger — one line per attempt in `data/logs/llm_usage.jsonl` (§9.4).
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, SecretStr, ValidationError
from tenacity import RetryCallState, Retrying, retry_if_exception, stop_after_attempt
from tenacity.wait import wait_exponential_jitter

from rag.core.config import ModelsConfig, Profile, Role, RoleModel, load_models_config
from rag.core.ledger import UsageLedger, UsageRecord
from rag.core.logging import get_logger
from rag.core.pacing import RateLimiter
from rag.core.settings import PROVIDER_KEY_FIELDS, Settings, get_settings
from rag.core.tokens import estimate_messages_tokens

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

# Bring-your-own-key (F2): the names the UI sends in `X-Provider` → the provider id used by
# `Settings.api_key_for` → the `models.yaml` profile that serves it. Adding a provider means
# adding a profile, not a code path.
PROVIDERS: dict[str, str] = {"groq": "groq", "anthropic": "anthropic", "gemini": "google"}
PROVIDER_PROFILES: dict[str, str] = {
    "groq": "groq_build",
    "anthropic": "anthropic",
    "gemini": "gemini",
}
# A retry-after longer than this (daily-quota 429s say "try again in 19m") is not worth blocking
# a request for; the call fails fast and a later cache-driven re-run picks the item up.
MAX_RETRY_AFTER_SECONDS = 120.0
DEFAULT_EXPECTED_OUTPUT_TOKENS = 512
_TRY_AGAIN_IN = re.compile(r"try again in\s+([0-9.]+)\s*(ms|s|m)\b", re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


# ------------------------------------------------------------------------------ errors


class LLMError(RuntimeError):
    """Base class for model-layer failures."""


class ProviderNotInstalledError(LLMError):
    """The active profile needs a provider package that is not installed."""


class LLMJSONError(LLMError):
    """The model failed to return JSON matching the schema, even after the corrective retry."""

    def __init__(self, message: str, raw: str, errors: str):
        super().__init__(message)
        self.raw = raw
        self.errors = errors


class UnknownProviderError(LLMError):
    """The caller asked for a provider this build does not serve."""


class LLMCallError(LLMError):
    """The provider call failed after all retries. `status_code` carries the provider's HTTP
    status when there was one, so the degrade path can say *why* (429 vs 400 vs outage)."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code

    @property
    def rate_limited(self) -> bool:
        return self.status_code == 429


# ----------------------------------------------------------------------------- helpers


def profile_for_provider(provider: str, models: ModelsConfig) -> str:
    """The `models.yaml` profile that serves a UI provider name, or `UnknownProviderError`."""
    name = PROVIDER_PROFILES.get(provider)
    if name is None:
        raise UnknownProviderError(
            f"Unknown provider {provider!r}; this build serves {sorted(PROVIDER_PROFILES)}"
        )
    if name not in models.profiles:
        raise UnknownProviderError(
            f"Provider {provider!r} needs profile {name!r}, which is not in models.yaml"
        )
    return name


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if code is None:
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
    return int(code) if isinstance(code, int) else None


def is_retryable(exc: BaseException) -> bool:
    code = _status_code(exc)
    if code is not None:
        if code == 429:
            hinted = retry_after_seconds(exc)
            return hinted is None or hinted <= MAX_RETRY_AFTER_SECONDS
        return code in RETRYABLE_STATUS
    name = type(exc).__name__
    return any(marker in name for marker in ("Connection", "Timeout"))


def retry_after_seconds(exc: BaseException) -> float | None:
    """`retry-after` header if present, else Groq's "Please try again in 7.66s" hint."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        raw = headers.get("retry-after") if hasattr(headers, "get") else None
        if raw:
            try:
                return float(raw)
            except ValueError:
                pass
    match = _TRY_AGAIN_IN.search(str(exc))
    if match:
        value, unit = float(match.group(1)), match.group(2).lower()
        return value / 1000 if unit == "ms" else value * 60 if unit == "m" else value
    return None


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _strip_fences(text: str) -> str:
    text = text.strip()
    return _FENCE.sub("", text).strip()


# Groq charges image tokens by pixel area; figure crops rasterised at 200 DPI (up to ~3300 px
# wide) cost ~3K input tokens raw against a 7K ITPM bucket, versus ~1.1K once bounded to this
# side length. The same bound is applied at ingestion, so descriptions and answers see the
# same picture.
VISION_MAX_SIDE_PX = 1600


def _image_to_data_url(image: Path | bytes | str, max_side: int = VISION_MAX_SIDE_PX) -> str:
    """Encode an image for a vision request, bounding its longer side to `max_side` px (JPEG).
    Pre-built `data:` URLs pass through untouched."""
    if isinstance(image, str) and image.startswith("data:"):
        return image
    if isinstance(image, bytes):
        data, mime = image, "image/png"
    else:
        path = Path(image)
        data = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            if max(im.size) > max_side:
                im = im.convert("RGB")
                im.thumbnail((max_side, max_side))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=85)
                data, mime = buf.getvalue(), "image/jpeg"
    except ImportError:  # Pillow is a serve dependency; keep working without it in tests
        pass
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _schema_instruction(schema: type[BaseModel]) -> str:
    return (
        "Respond with a single JSON object and nothing else — no prose, no code fences. "
        "The JSON must validate against this JSON Schema:\n"
        f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
    )


@dataclass
class LLMResponse:
    text: str
    model: str
    role: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    retries: int
    finish_reason: str | None = None
    raw: AIMessage | None = None


ChatFactory = Callable[[RoleModel, str], BaseChatModel]


# ------------------------------------------------------------------------------ client


class LLMClient:
    def __init__(
        self,
        settings: Settings | None = None,
        models_config: ModelsConfig | None = None,
        *,
        ledger: UsageLedger | None = None,
        chat_factory: ChatFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        max_attempts: int = 5,
        pacing_enabled: bool = True,
    ):
        self.settings = settings or get_settings()
        self.config = models_config or load_models_config(settings=self.settings)
        self.profile_name = self.config.active_profile
        self.profile: Profile = self.config.active()
        self._ledger = ledger or UsageLedger(self.settings.logs_dir / "llm_usage.jsonl")
        self._chat_factory = chat_factory or self._build_chat_model
        self._clock = clock
        self._sleep = sleep
        self._max_attempts = max_attempts
        self._pacing_enabled = pacing_enabled
        self._models: dict[Role, BaseChatModel] = {}
        self._limiters: dict[str, RateLimiter] = {}

        # Fail fast (plan Phase 0): every provider the active profile uses must have a key.
        for provider in sorted(self.profile.providers()):
            self.settings.api_key_for(provider)

    # -- bring-your-own-key -------------------------------------------------------------

    @classmethod
    def for_request(
        cls,
        provider: str,
        api_key: str,
        *,
        settings: Settings | None = None,
        models_config: ModelsConfig | None = None,
        **kwargs: Any,
    ) -> LLMClient:
        """A client that uses the caller's key, for one request (F2, bring-your-own-key).

        The environment is not consulted: the returned client carries a `Settings` copy whose
        only credential is `api_key`, and a `ModelsConfig` copy whose active profile is the one
        that serves `provider`. Nothing is cached process-wide, so two visitors never share a
        key, a rate limiter or a chat-model object.
        """
        profile_name = profile_for_provider(provider, models_config or load_models_config())
        base = settings or get_settings()
        field = PROVIDER_KEY_FIELDS[PROVIDERS[provider]]
        blanked = dict.fromkeys(PROVIDER_KEY_FIELDS.values())  # drop any key from .env
        scrubbed = base.model_copy(
            update={**blanked, field: SecretStr(api_key), "model_profile": profile_name}
        )
        models = (models_config or load_models_config(settings=scrubbed)).model_copy(
            update={"active_profile": profile_name}
        )
        return cls(scrubbed, models, **kwargs)

    def smoke_test(self, role: Role = "small", *, request_id: str | None = None) -> str:
        """Cheapest possible call that proves the key works: one token out. Returns the model id.

        `max_tokens=1` is the point — a rejected key costs the caller nothing, and an accepted
        one costs a single output token. Reasoning models spend hidden tokens before any visible
        output, so an empty reply is still a success: the call was authorised, which is all this
        asks. Provider errors propagate as `LLMError` for the caller to map to a status code.
        """
        self._invoke(
            self._messages("ping", None),
            role,
            json_mode=False,
            request_id=request_id,
            max_tokens=1,
        )
        return self.model_id(role)

    # -- public API ---------------------------------------------------------------------

    def text(
        self,
        prompt: str,
        role: Role = "small",
        *,
        system: str | None = None,
        request_id: str | None = None,
        ingestion_job: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = self._messages(prompt, system)
        return self._invoke(
            messages,
            role,
            json_mode=False,
            request_id=request_id,
            ingestion_job=ingestion_job,
            max_tokens=max_tokens,
        ).text

    def json(
        self,
        prompt: str,
        schema: type[T],
        role: Role = "large",
        *,
        system: str | None = None,
        request_id: str | None = None,
        ingestion_job: str | None = None,
        max_tokens: int | None = None,
    ) -> T:
        messages = self._messages(prompt, system, schema=schema)
        return self._json_call(
            messages,
            schema,
            role,
            request_id=request_id,
            ingestion_job=ingestion_job,
            max_tokens=max_tokens,
        )

    def vision_json(
        self,
        prompt: str,
        images: Sequence[Path | bytes | str],
        schema: type[T],
        role: Role = "vision",
        *,
        system: str | None = None,
        request_id: str | None = None,
        ingestion_job: str | None = None,
        max_tokens: int | None = None,
    ) -> T:
        if not images:
            raise ValueError("vision_json needs at least one image")
        limit = self.profile.role(role).max_images_per_request
        if limit is not None and len(images) > limit:
            raise ValueError(
                f"{len(images)} images exceed the {limit}-image limit of role {role!r}"
            )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_to_data_url(img)}} for img in images
        )
        messages = self._messages(content, system, schema=schema)
        return self._json_call(
            messages,
            schema,
            role,
            request_id=request_id,
            ingestion_job=ingestion_job,
            max_tokens=max_tokens,
        )

    def model_id(self, role: Role) -> str:
        return self.profile.role(role).model

    def max_output_tokens(self, role: Role) -> int | None:
        """Largest `max_tokens` the role's pacing allows in one request (its OTPM bucket), or
        None when the role has no output-token limit. Groq charges the *requested* budget (D-52),
        so callers cap their request here instead of tripping `PacingError`."""
        pacing = self.profile.pacing_for(role)
        return pacing.otpm if pacing and pacing.otpm else None

    def max_input_tokens(self, role: Role, output_budget: int = 0) -> int | None:
        """Largest input (prompt + images) one request may carry: the role's ITPM gate when the
        provider has one, else whatever the TPM bucket leaves after the output budget."""
        pacing = self.profile.pacing_for(role)
        if pacing is None:
            return None
        if pacing.itpm:
            return pacing.itpm
        return max(0, pacing.tpm - output_budget)

    # -- internals ----------------------------------------------------------------------

    @staticmethod
    def _messages(
        content: str | list[dict[str, Any]],
        system: str | None,
        schema: type[BaseModel] | None = None,
    ) -> list[BaseMessage]:
        system_parts = [p for p in (system, _schema_instruction(schema) if schema else None) if p]
        messages: list[BaseMessage] = []
        if system_parts:
            messages.append(SystemMessage(content="\n\n".join(system_parts)))
        messages.append(HumanMessage(content=content))
        return messages

    def _json_call(
        self,
        messages: list[BaseMessage],
        schema: type[T],
        role: Role,
        **call_kwargs: Any,
    ) -> T:
        """JSON mode → validate; on failure retry once with the error appended; then raise.

        Groq answers 400 `json_validate_failed` when hidden reasoning exhausts `max_tokens`
        before the JSON closes; that is retried once with double the output budget.
        """
        try:
            response = self._invoke(messages, role, json_mode=True, **call_kwargs)
        except LLMCallError as exc:
            if "json_validate_failed" not in str(exc) or not call_kwargs.get("max_tokens"):
                raise
            call_kwargs = {**call_kwargs, "max_tokens": call_kwargs["max_tokens"] * 2}
            log.warning(
                "JSON truncated before completion (role=%s); retrying with max_tokens=%d",
                role,
                call_kwargs["max_tokens"],
            )
            response = self._invoke(messages, role, json_mode=True, **call_kwargs)
        parsed, error = self._parse(response.text, schema)
        if parsed is not None:
            return parsed

        self._ledger.append(
            self._record(
                role, status="invalid_json", error=(error or "")[:500], retries=0, **call_kwargs
            )
        )
        log.warning("Invalid JSON from role=%s model=%s; retrying once", role, response.model)
        retry_messages = [
            *messages,
            AIMessage(content=response.text),
            HumanMessage(
                content=(
                    "Your previous response was not valid JSON for the required schema. "
                    f"Validation error:\n{error}\n\n"
                    "Respond again with only the corrected JSON object."
                )
            ),
        ]
        response = self._invoke(retry_messages, role, json_mode=True, **call_kwargs)
        parsed, error = self._parse(response.text, schema)
        if parsed is not None:
            return parsed
        raise LLMJSONError(
            f"role={role} model={response.model} returned invalid JSON twice: {error}",
            raw=response.text,
            errors=error or "",
        )

    @staticmethod
    def _parse(text: str, schema: type[T]) -> tuple[T | None, str | None]:
        try:
            return schema.model_validate_json(_strip_fences(text)), None
        except (ValidationError, ValueError) as exc:
            return None, str(exc)

    def _record(self, role: Role, **fields: Any) -> UsageRecord:
        cfg = self.profile.role(role)
        fields.pop("max_tokens", None)
        return UsageRecord(role=role, provider=cfg.provider, model=cfg.model, **fields)

    def _model(self, role: Role) -> BaseChatModel:
        if role not in self._models:
            self._models[role] = self._chat_factory(self.profile.role(role), role)
        return self._models[role]

    def _limiter(self, role: Role) -> RateLimiter | None:
        if not self._pacing_enabled:
            return None
        cfg = self.profile.role(role)
        pacing = self.profile.pacing_for(role)
        if pacing is None:
            return None
        if cfg.model not in self._limiters:
            self._limiters[cfg.model] = RateLimiter(pacing, clock=self._clock, sleep=self._sleep)
        return self._limiters[cfg.model]

    def _build_chat_model(self, cfg: RoleModel, role: str) -> BaseChatModel:
        """Lazy provider import: only the active profile's package must be installed."""
        api_key = self.settings.api_key_for(cfg.provider)
        if cfg.provider == "groq":
            try:
                from langchain_groq import ChatGroq
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ProviderNotInstalledError("pip install langchain-groq") from exc
            return ChatGroq(
                model=cfg.model,
                api_key=api_key,
                temperature=0,
                max_retries=0,  # retries and pacing are handled here, with ledger lines
                **cfg.provider_kwargs,
            )
        if cfg.provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError as exc:  # pragma: no cover
                raise ProviderNotInstalledError(
                    "pip install -r requirements-future.txt (langchain-anthropic)"
                ) from exc
            # No sampling parameters: Claude Sonnet 5 / Opus 5 reject `temperature` (400,
            # "deprecated for this model"); determinism comes from the prompt contract and the
            # verifier instead. Thinking / effort settings go through `provider_kwargs`.
            return ChatAnthropic(
                model=cfg.model,
                api_key=api_key,
                max_retries=0,
                **cfg.provider_kwargs,
            )
        if cfg.provider == "google":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as exc:  # pragma: no cover
                raise ProviderNotInstalledError(
                    "pip install -r requirements-future.txt (langchain-google-genai)"
                ) from exc
            return ChatGoogleGenerativeAI(
                model=cfg.model, google_api_key=api_key, temperature=0, **cfg.provider_kwargs
            )
        raise LLMError(f"Unsupported provider {cfg.provider!r}")  # pragma: no cover

    def _bind(self, model: BaseChatModel, role: Role, json_mode: bool, max_tokens: int | None):
        """Per-call binding. JSON mode is a Groq request option; native providers rely on the
        schema instruction + validation in Phase 0 (structured output adopted at F1)."""
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode and self.profile.role(role).provider == "groq":
            kwargs["response_format"] = {"type": "json_object"}
        return model.bind(**kwargs) if kwargs else model

    def _wait(self, retry_state: RetryCallState) -> float:
        """Exponential backoff with jitter, but never shorter than the provider's retry-after."""
        base = wait_exponential_jitter(initial=1, max=30, jitter=1)(retry_state)
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        hinted = retry_after_seconds(exc) if exc is not None else None
        return max(base, hinted or 0.0)

    def _invoke(
        self,
        messages: list[BaseMessage],
        role: Role,
        *,
        json_mode: bool,
        request_id: str | None = None,
        ingestion_job: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        cfg = self.profile.role(role)
        runnable = self._bind(self._model(role), role, json_mode, max_tokens)
        limiter = self._limiter(role)
        estimate = estimate_messages_tokens(messages) + (
            max_tokens or DEFAULT_EXPECTED_OUTPUT_TOKENS
        )
        meta = {"request_id": request_id, "ingestion_job": ingestion_job}
        attempts_made = 0

        retrying = Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._wait,
            retry=retry_if_exception(is_retryable),
            sleep=self._sleep,
            reraise=True,
        )
        try:
            for attempt in retrying:
                with attempt:
                    retries = attempt.retry_state.attempt_number - 1
                    attempts_made = retries + 1
                    waited = (
                        limiter.acquire(estimate, max_tokens or DEFAULT_EXPECTED_OUTPUT_TOKENS)
                        if limiter
                        else 0.0
                    )
                    started = self._clock()
                    try:
                        result = runnable.invoke(messages)
                    except Exception as exc:
                        latency_ms = int((self._clock() - started) * 1000)
                        status = "rate_limited" if _status_code(exc) == 429 else "error"
                        self._ledger.append(
                            self._record(
                                role,
                                status=status,
                                retries=retries,
                                latency_ms=latency_ms,
                                pacing_wait_ms=int(waited * 1000),
                                error=f"{type(exc).__name__}: {exc}"[:500],
                                **meta,
                            )
                        )
                        log.warning(
                            "LLM call failed role=%s model=%s attempt=%d status=%s",
                            role,
                            cfg.model,
                            retries + 1,
                            status,
                        )
                        raise

                    latency_ms = int((self._clock() - started) * 1000)
                    usage = getattr(result, "usage_metadata", None) or {}
                    tokens_in = int(usage.get("input_tokens", 0) or 0)
                    tokens_out = int(usage.get("output_tokens", 0) or 0)
                    if limiter:
                        limiter.reconcile(estimate, tokens_in + tokens_out)
                    finish_reason = (getattr(result, "response_metadata", None) or {}).get(
                        "finish_reason"
                    )
                    text = _content_to_text(result.content)
                    if finish_reason == "length":
                        # Reasoning models spend hidden tokens first; a tight max_tokens can
                        # leave no room for the visible answer.
                        log.warning(
                            "Output truncated (finish_reason=length) role=%s model=%s "
                            "max_tokens=%s content_chars=%d",
                            role,
                            cfg.model,
                            max_tokens,
                            len(text),
                        )
                    self._ledger.append(
                        self._record(
                            role,
                            status="ok",
                            retries=retries,
                            tokens_in=tokens_in,
                            tokens_out=tokens_out,
                            latency_ms=latency_ms,
                            pacing_wait_ms=int(waited * 1000),
                            **meta,
                        )
                    )
                    return LLMResponse(
                        text=text,
                        model=cfg.model,
                        role=role,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        latency_ms=latency_ms,
                        retries=retries,
                        finish_reason=finish_reason,
                        raw=result if isinstance(result, AIMessage) else None,
                    )
        except LLMError:
            raise
        except Exception as exc:
            status = _status_code(exc)
            raise LLMCallError(
                f"role={role} model={cfg.model} failed after {attempts_made} attempt(s)"
                f"{f' (HTTP {status})' if status else ''}: {type(exc).__name__}: {exc}",
                status_code=status,
            ) from exc
        raise LLMCallError("unreachable")  # pragma: no cover

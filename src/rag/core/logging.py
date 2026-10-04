"""Structured logging with API-key redaction (architecture §13: "API key leaked into logs").

Three layers of defence:
1. Known secret values (from `Settings.secret_values()`) are replaced wherever they appear.
2. Provider key *shapes* (`gsk_…`, `sk-ant-…`, `AIza…`) are masked even if a key was never
   registered — e.g. one pasted into a prompt by mistake.
3. The key of the request in flight (F2 bring-your-own-key), held in a context variable by
   `use_key()` and applied by `redact()`. The trace writer and the usage ledger call `redact()`
   before writing, so a visitor's key cannot reach disk even if its shape is unknown to layer 2.
   Context variables follow the request into a worker thread, which is where the pipeline runs.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import IO, Any

REDACTED = "***REDACTED***"

# Shapes of the provider keys this project can hold. Deliberately loose on the tail length.
SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"gsk_[A-Za-z0-9_\-]{8,}"),  # Groq
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),  # Anthropic
    re.compile(r"AIza[0-9A-Za-z_\-]{16,}"),  # Google
)

_HANDLER_MARK = "_rag_configured_handler"


class Redactor:
    """Masks secrets in strings and, recursively, in dicts / lists / tuples."""

    def __init__(self, secrets: Iterable[str] = ()):
        # Longest first so a key that is a prefix of another is not left half-masked.
        self._secrets = sorted({s for s in secrets if s}, key=len, reverse=True)

    def redact_text(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        for pattern in SECRET_PATTERNS:
            text = pattern.sub(REDACTED, text)
        return text

    def redact(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, Mapping):
            return {k: self.redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self.redact(v) for v in value)
        if isinstance(value, BaseException):
            return self.redact_text(str(value))
        return value


_SHAPE_REDACTOR = Redactor()
_CURRENT_KEY: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "rag_current_provider_key", default=None
)


@contextmanager
def use_key(key: str | None) -> Iterator[None]:
    """Make `key` redactable for the duration of the block, and in threads it hands off to."""
    token = _CURRENT_KEY.set(key or None)
    try:
        yield
    finally:
        _CURRENT_KEY.reset(token)


def redact(value: Any) -> Any:
    """Mask provider keys in anything about to be written to disk or handed to a caller.

    Masks the in-flight request's key whatever its shape, plus the known provider key shapes,
    recursing through dicts, lists, tuples and exceptions. Anything with no strings in it comes
    back unchanged, so this is safe to call on any value.
    """
    key = _CURRENT_KEY.get()
    return (Redactor([key]) if key else _SHAPE_REDACTOR).redact(value)


class RedactingFilter(logging.Filter):
    """Applies a `Redactor` to the message, its args, and any `extra` fields."""

    _STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message"}

    def __init__(self, redactor: Redactor):
        super().__init__()
        self.redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.redactor.redact(record.msg)
        if record.args:
            record.args = self.redactor.redact(record.args)
        for key in list(vars(record)):
            if key not in self._STANDARD_ATTRS:
                setattr(record, key, self.redactor.redact(getattr(record, key)))
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line; extra fields are included at top level."""

    _SKIP = RedactingFilter._STANDARD_ATTRS

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in self._SKIP:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(
    level: str | int = "INFO",
    secrets: Iterable[str] = (),
    *,
    json_output: bool = False,
    stream: IO[str] | None = None,
) -> logging.Logger:
    """Install one redacting handler on the root logger. Safe to call repeatedly."""
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            root.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stderr)
    setattr(handler, _HANDLER_MARK, True)
    handler.addFilter(RedactingFilter(Redactor(secrets)))
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root.addHandler(handler)
    # HTTP client libraries log every request at INFO; keep them to warnings.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

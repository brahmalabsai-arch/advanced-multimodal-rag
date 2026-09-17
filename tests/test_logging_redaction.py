"""Phase 0 tests — log redaction: a logged dict containing the API key is masked."""

from __future__ import annotations

import io
import json
import logging

from rag.core.logging import REDACTED, Redactor, configure_logging, get_logger

KEY = "gsk_abcdefghijklmnopqrstuvwxyz0123456789"


def _capture(secrets=(), json_output=False) -> tuple[io.StringIO, logging.Logger]:
    stream = io.StringIO()
    configure_logging("DEBUG", secrets=secrets, json_output=json_output, stream=stream)
    return stream, get_logger("test.redaction")


def test_logged_dict_with_registered_key_is_masked() -> None:
    stream, logger = _capture(secrets=[KEY])
    logger.info("settings loaded: %s", {"groq_api_key": KEY, "profile": "groq_build"})
    out = stream.getvalue()
    assert KEY not in out
    assert REDACTED in out
    assert "groq_build" in out


def test_unregistered_key_shape_is_still_masked() -> None:
    stream, logger = _capture(secrets=[])
    logger.warning("header: Authorization: Bearer %s", "gsk_Zz9999999999999999999999")
    logger.warning(
        "anthropic %s google %s", "sk-ant-api03-abcdefghijklmnop", "AIzaSyABCDEFGHIJKLMNOPQRSTUV"
    )
    out = stream.getvalue()
    assert "gsk_" not in out
    assert "sk-ant-" not in out
    assert "AIzaSy" not in out
    assert out.count(REDACTED) == 3


def test_extra_fields_are_masked_in_json_output() -> None:
    stream, logger = _capture(secrets=[KEY], json_output=True)
    logger.info("call", extra={"payload": {"headers": {"Authorization": f"Bearer {KEY}"}}})
    record = json.loads(stream.getvalue().strip())
    assert record["msg"] == "call"
    assert record["payload"]["headers"]["Authorization"] == f"Bearer {REDACTED}"


def test_redactor_handles_nested_structures_and_exceptions() -> None:
    r = Redactor([KEY])
    nested = {"a": [KEY, ("x", KEY)], "b": {"c": f"key={KEY}"}, "n": 3}
    out = r.redact(nested)
    assert out == {"a": [REDACTED, ("x", REDACTED)], "b": {"c": f"key={REDACTED}"}, "n": 3}
    assert r.redact(RuntimeError(f"boom {KEY}")) == f"boom {REDACTED}"


def test_configure_logging_is_idempotent() -> None:
    stream1, _ = _capture()
    stream2, logger = _capture()
    logger.info("once")
    assert stream1.getvalue() == ""
    assert stream2.getvalue().count("once") == 1

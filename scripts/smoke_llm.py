"""Phase 0 smoke test: one call per role through the active profile (plan §Phase 0).

    make smoke      (or)      .venv/Scripts/python scripts/smoke_llm.py

Exit criterion: all three roles answer and exactly three `ok` lines are appended to the ledger.
Costs 3 Groq calls per run.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, Field

from rag.core.console import utf8_console
from rag.core.ledger import UsageLedger
from rag.core.logging import configure_logging, get_logger
from rag.core.settings import PROJECT_ROOT, SettingsError, get_settings
from rag.llm import LLMClient, LLMError

log = get_logger("smoke")

IMAGE = PROJECT_ROOT / "tests" / "fixtures" / "smoke_image.png"


class Arithmetic(BaseModel):
    answer: int = Field(description="the numeric result")
    reasoning: str = Field(description="one short sentence")


class ImageDescription(BaseModel):
    dominant_colors: list[str] = Field(description="colour names, most prominent first")
    shapes: list[str] = Field(description="simple shape names visible in the image")
    description: str = Field(description="one sentence describing the image")


def main() -> int:
    utf8_console()
    try:
        settings = get_settings()
        configure_logging(settings.log_level, secrets=settings.secret_values())
        client = LLMClient(settings)
    except SettingsError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2

    ledger = UsageLedger(settings.logs_dir / "llm_usage.jsonl")
    before = ledger.count()
    print(f"profile={client.profile_name}  ledger={ledger.path}  lines_before={before}")

    failures = 0
    try:
        reply = client.text(
            "Reply with exactly one word: PONG",
            role="small",
            request_id="smoke-text",
            max_tokens=64,  # reasoning models spend hidden tokens first; 16 is too tight
        )
        print(f"[small ] {client.model_id('small')} -> {reply.strip()!r}")
        if not reply.strip():
            failures += 1
            print("[small ] FAILED: empty reply", file=sys.stderr)
    except LLMError as exc:
        failures += 1
        print(f"[small ] FAILED: {exc}", file=sys.stderr)

    try:
        result = client.json(
            "What is 17 + 25? Return the answer and one sentence of reasoning.",
            Arithmetic,
            role="large",
            request_id="smoke-json",
            max_tokens=128,
        )
        ok = result.answer == 42
        verdict = "OK" if ok else "WRONG"
        print(f"[large ] {client.model_id('large')} -> {result.model_dump()}  {verdict}")
        failures += 0 if ok else 1
    except LLMError as exc:
        failures += 1
        print(f"[large ] FAILED: {exc}", file=sys.stderr)

    try:
        if not IMAGE.exists():
            raise FileNotFoundError(IMAGE)
        desc = client.vision_json(
            "Describe this image.",
            [Path(IMAGE)],
            ImageDescription,
            request_id="smoke-vision",
            max_tokens=256,
        )
        print(f"[vision] {client.model_id('vision')} -> {desc.model_dump()}")
    except (LLMError, FileNotFoundError) as exc:
        failures += 1
        print(f"[vision] FAILED: {exc}", file=sys.stderr)

    records = ledger.read_all()[before:]
    ok_lines = [r for r in records if r.status == "ok"]
    tokens_in = sum(r.tokens_in for r in ok_lines)
    tokens_out = sum(r.tokens_out for r in ok_lines)
    print(
        f"ledger: +{len(records)} lines ({len(ok_lines)} ok); "
        f"tokens_in={tokens_in} tokens_out={tokens_out}"
    )
    if failures or len(ok_lines) != 3:
        print("SMOKE FAILED", file=sys.stderr)
        return 1
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

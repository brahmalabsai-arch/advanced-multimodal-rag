"""Ask one question from the terminal (plan Phase 3).

    .venv/Scripts/python scripts/ask_cli.py "What is the current ratio as of Jan 25, 2026?"
    .venv/Scripts/python scripts/ask_cli.py --api "..."      # go through a running server
    .venv/Scripts/python scripts/ask_cli.py --json "..."     # dump the full result

Runs the pipeline in-process by default (loads the index, ~15 s on first call).
"""

from __future__ import annotations

import argparse
import json
import sys

from rag.core.console import utf8_console


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("question")
    ap.add_argument("--api", action="store_true", help="go through the running server")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--json", action="store_true", help="print the full JSON response")
    ap.add_argument("--no-bypass", action="store_true", help="allow cache (default: bypass)")
    args = ap.parse_args(argv)

    if args.api:
        import httpx

        r = httpx.post(
            f"{args.url}/api/ask",
            json={"question": args.question, "bypass_cache": not args.no_bypass},
            timeout=180,
        )
        r.raise_for_status()
        data = r.json()
        if args.json:
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return 0
        answer, debug = data["answer"], data["debug"]
        citations = [f"{c['block_id']} {c['label']}" for c in data["citations"]]
    else:
        from rag.core.logging import configure_logging
        from rag.core.settings import get_settings
        from rag.graph import Pipeline

        s = get_settings()
        configure_logging("WARNING", secrets=s.secret_values())
        result = Pipeline().ask(args.question, bypass_cache=not args.no_bypass)
        if args.json:
            print(result.model_dump_json(indent=2))
            return 0
        answer = result.answer.model_dump()
        debug = {
            "intent": result.intent,
            "total_latency_ms": result.total_latency_ms,
            "tokens_by_model": result.tokens_by_model,
            "generation_attempts": result.generation_attempts,
            "verify": result.verify.model_dump(),
            "calculations": [c.model_dump() for c in result.calculations],
        }
        citations = [
            f"{b.block_id} {b.header.strip('[]')}"
            for b in result.context.blocks
            if b.block_id in result.verify.citations_found
        ]
        if result.warning:
            print(f"WARNING: {result.warning}\n")

    print(answer["answer_markdown"])
    print(
        f"\n[{debug['intent']} · confidence {answer['confidence']} · {answer['answer_class']} · "
        f"{debug['total_latency_ms']} ms · attempts {debug['generation_attempts']} · verify {debug['verify']['passed']}]"
    )
    for c in debug.get("calculations", []):
        print(f"calc {c['formula']}: {c.get('rounded')} ({c['status']})")
    for c in citations:
        print(f"  {c}")
    print("tokens:", debug["tokens_by_model"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

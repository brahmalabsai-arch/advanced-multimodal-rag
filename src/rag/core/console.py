"""Console encoding for the CLI scripts (Phase 8, found by the fresh-clone rehearsal).

Windows consoles default to the ANSI code page (cp1252), and model answers routinely contain
characters outside it — a narrow no-break space in "$206,803 million", en dashes, arrows — so
`print(answer)` raised `UnicodeEncodeError` on a fresh machine even though the request had
succeeded. Every script calls `utf8_console()` first; on a UTF-8 terminal it is a no-op.
"""

from __future__ import annotations

import contextlib
import sys


def utf8_console() -> None:
    """Switch stdout/stderr to UTF-8 with replacement, when the streams support it."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # a closed or exotic stream is left alone
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


__all__ = ["utf8_console"]

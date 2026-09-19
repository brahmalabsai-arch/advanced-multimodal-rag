"""Single source of "now" for TTL expiry, LFU decay, the sweeper and time-anchored rules
(architecture §14.3, D-46).

`Clock.now()` returns epoch seconds. In dev mode an offset can be applied (`POST /api/admin/clock`
or the UI buttons) so 1-, 7- and 30-day expiry is testable in seconds; outside dev mode
`set_offset` refuses. Log timestamps (`ts` fields in traces, ledger, decision log) stay on the real
wall clock — they record when something happened — and every trace carries `clock_offset_s`, so
a shifted clock is always visible in the record.

Pacing and latency measurements use `time.perf_counter()`/monotonic clocks and are unaffected.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from functools import lru_cache

from rag.core.settings import get_settings


class ClockError(RuntimeError):
    """Raised when the dev offset is set outside dev mode or beyond the allowed range."""


class Clock:
    def __init__(self, *, allow_offset: bool, max_offset_days: int = 400):
        self.allow_offset = allow_offset
        self.max_offset_s = max_offset_days * 86_400
        self._offset_s = 0
        self._lock = threading.Lock()

    @property
    def offset_s(self) -> int:
        return self._offset_s

    def now(self) -> int:
        """Epoch seconds, with the dev offset applied."""
        return int(time.time()) + self._offset_s

    def now_iso(self) -> str:
        return datetime.fromtimestamp(self.now(), UTC).isoformat(timespec="seconds")

    def set_offset(self, seconds: int) -> int:
        if not self.allow_offset:
            raise ClockError("the dev clock offset is only available when APP_ENV=dev")
        if abs(int(seconds)) > self.max_offset_s:
            raise ClockError(
                f"offset {seconds}s exceeds the configured maximum of {self.max_offset_s}s"
            )
        with self._lock:
            self._offset_s = int(seconds)
        return self._offset_s

    def advance(self, seconds: int) -> int:
        return self.set_offset(self._offset_s + int(seconds))

    def reset(self) -> int:
        with self._lock:
            self._offset_s = 0
        return 0


@lru_cache(maxsize=1)
def get_clock() -> Clock:
    """Process-wide clock; the offset is allowed only in dev mode (`Settings.is_dev`)."""
    settings = get_settings()
    return Clock(allow_offset=settings.is_dev)


def now() -> int:
    return get_clock().now()

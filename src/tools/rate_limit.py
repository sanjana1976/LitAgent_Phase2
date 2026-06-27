"""
Simple per-host rate limiting for courteous API usage.

Uses a minimum interval between calls per logical host key (not thread-safe for
high concurrency; adequate for an interactive agent).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class RateLimiter:
    """Enforce a minimum delay between successive calls per key."""

    def __init__(self, default_interval_sec: float = 1.0) -> None:
        self._default = default_interval_sec
        self._last: dict[str, float] = {}
        self._intervals: dict[str, float] = defaultdict(lambda: self._default)
        self._lock = threading.Lock()

    def set_interval(self, key: str, seconds: float) -> None:
        """Override minimum spacing for a logical key (e.g. api.semanticscholar.org)."""
        self._intervals[key] = max(0.0, seconds)

    def wait(self, key: str) -> None:
        """Sleep if needed so the previous call for ``key`` is at least ``interval`` ago."""
        with self._lock:
            interval = self._intervals[key]
            now = time.monotonic()
            last = self._last.get(key)
            if last is not None:
                elapsed = now - last
                if elapsed < interval:
                    sleep_for = interval - elapsed
                    logger.debug("Rate limit sleep %.3fs for %s", sleep_for, key)
                    time.sleep(sleep_for)
            self._last[key] = time.monotonic()


_global_limiter = RateLimiter(default_interval_sec=1.0)


def global_rate_limiter() -> RateLimiter:
    return _global_limiter

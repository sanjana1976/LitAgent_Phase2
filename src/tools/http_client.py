"""Shared httpx wrapper with rate limiting, timeouts, and structured errors."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from tools.rate_limit import global_rate_limiter

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=15.0)


def rate_limited_get(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    host_key: str | None = None,
) -> httpx.Response:
    """
    Perform a GET with polite spacing per host.

    Args:
        url: Full URL.
        params: Query parameters.
        headers: Optional extra headers (User-Agent set if missing).
        host_key: Rate-limiter bucket; defaults to URL host.
    """
    lim = global_rate_limiter()
    key = host_key or httpx.URL(url).host
    lim.wait(key)
    merged_headers = {
        "User-Agent": "ResearchPaperAnalyzer/0.1 (mailto:example@example.edu)",
        **(headers or {}),
    }
    with httpx.Client(timeout=_DEFAULT_TIMEOUT, follow_redirects=True) as client:
        response = client.get(url, params=params, headers=merged_headers)
    response.raise_for_status()
    return response

"""
Utility helpers for async execution, network retries, and API rate limiting.

- run_async(coro): Runs async coroutines in sync contexts safely.
- retry_on_network_error(max_retries=5, delay=3): Decorator to retry functions on
  network/ServerError (503) and quota (429 RESOURCE_EXHAUSTED) failures.
- gemini_rate_limit(): Decorator that throttles calls so we never exceed the
  configured requests-per-minute budget for the Gemini API (avoids hitting 429s
  in the first place instead of just reacting to them).
"""
import asyncio
import os
import re
import threading
import time
from collections import deque
from functools import wraps

import httpx
import httpcore
from google.genai.errors import ServerError, ClientError


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _extract_retry_delay(exc, default=10.0):
    """In case of a 429 error, Google usually returns 'retry after X seconds' information 
    either in the RetryInfo block inside error.details or within the error message. 
    If we can find it, we use that; otherwise, we fall back to a reasonable default."""
    try:
        details = getattr(exc, "details", None) or {}
        if isinstance(details, dict):
            for item in details.get("error", {}).get("details", []) or []:
                if item.get("@type", "").endswith("RetryInfo"):
                    delay_str = item.get("retryDelay", "")
                    match = re.match(r"([\d.]+)s?", delay_str)
                    if match:
                        return float(match.group(1))
    except Exception:
        pass

    match = re.search(r"retry in ([\d.]+)s", str(exc), re.IGNORECASE)
    if match:
        return float(match.group(1))

    return default


class _RateLimiter:
    """Simple, thread-safe 'sliding window' rate limiter. Pools all Gemini calls
    under a single global budget to keep the number of requests per minute under the
    limit; waits when the limit is reached instead of waiting for an error."""

    def __init__(self, max_calls_per_minute=4):
        self.max_calls = max_calls_per_minute
        self.window = 60.0
        self._calls = deque()
        self._lock = threading.Lock()

    def acquire(self):
        wait_time = 0
        with self._lock:
            now = time.monotonic()
            while self._calls and now - self._calls[0] > self.window:
                self._calls.popleft()

            if len(self._calls) >= self.max_calls:
                wait_time = self.window - (now - self._calls[0]) + 0.1

        if wait_time > 0:
            time.sleep(wait_time)
            return self.acquire()

        with self._lock:
            self._calls.append(time.monotonic())


_gemini_limiter = _RateLimiter(max_calls_per_minute=int(os.getenv("GEMINI_RPM_LIMIT", "4")))


def gemini_rate_limit():
    """Allocates space from the global request budget before the function runs.
    Waits BEFORE hitting a 429, not after."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _gemini_limiter.acquire()
            return func(*args, **kwargs)

        return wrapper

    return decorator


def retry_on_network_error(max_retries=5, delay=3):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except (httpx.ConnectError, httpcore.ConnectError, ServerError, ClientError) as e:
                    wait = delay

                    if isinstance(e, ClientError):
                        status = getattr(e, "status_code", None) or getattr(e, "code", None)
                        if status != 429:
                            raise
                        wait = _extract_retry_delay(e, default=delay)
                    elif isinstance(e, ServerError):
                        if not getattr(e, "status_code", None) == 503:
                            raise
                    elif isinstance(e, (httpx.ConnectError, httpcore.ConnectError)):
                        pass

                    attempts += 1
                    if attempts > max_retries:
                        raise
                    time.sleep(wait)

        return wrapper

    return decorator
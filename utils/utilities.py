"""
Utility helpers for async execution and network retries.

- run_async(coro): Runs async coroutines in sync contexts safely.
- retry_on_network_error(max_retries=5, delay=3): Decorator to retry functions on network/ServerError (503) failures.
"""
import asyncio
import time
from functools import wraps
import httpx
import httpcore
from google.genai.errors import ServerError

def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def retry_on_network_error(max_retries=5, delay=3):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except (httpx.ConnectError, httpcore.ConnectError, ServerError) as e:
                    if isinstance(e, ServerError):
                        if not getattr(e, 'status_code', None) == 503:
                            raise
                    elif isinstance(e, (httpx.ConnectError, httpcore.ConnectError)):
                        pass
                    attempts += 1
                    if attempts > max_retries:
                        raise
                    time.sleep(delay)

        return wrapper

    return decorator
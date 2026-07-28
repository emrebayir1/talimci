"""
Model management center. Provides unified access to different LLM
models via Google GenAI, Groq, and OpenRouter APIs.

Usage:
    from models import get_chat_model

    chat = get_chat_model()
    response = chat.send_message("Hello!")

Environment variables (.env file in the project root):
    GEMINI_API                - Google GenAI API key
    GROQ_API                  - Groq API key
    OPENROUTER_API_KEY        - OpenRouter API key (no credit card required for free ':free' models)
    LLM_PROVIDER              - "gemini", "groq", or "openrouter" (default: gemini)
    CHAT_MODEL                - Name of the chat model to use
    GEMINI_RPM_LIMIT          - Requests per minute limit for Gemini (default: 4)
    GEMINI_TPM_LIMIT          - Tokens per minute limit for Gemini (default: disabled)
    GROQ_RPM_LIMIT            - Requests per minute limit for Groq (default: 28)
    GROQ_TPM_LIMIT            - Tokens per minute limit for Groq (default: 12000)
    OPENROUTER_RPM_LIMIT      - Requests per minute limit for OpenRouter (default: 18)
    OPENROUTER_TPM_LIMIT      - Tokens per minute limit for OpenRouter (default: disabled)
    OPENROUTER_SITE_URL       - (optional) OpenRouter ranking header
    OPENROUTER_APP_NAME       - (optional) OpenRouter app name header
"""
import os
import json
import time
import threading
import asyncio
import re
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union
from dataclasses import dataclass
from collections import deque
from functools import wraps
from urllib.parse import urlparse, parse_qs

import requests


def _sanitize_model_name(name: Optional[str], hint: str = "model") -> Optional[str]:
    """Sometimes, instead of a model name, the URL from the provider's website is
    mistakenly copied and pasted into .env
    (e.g. 'https://jina.ai?sui=embeddings&model=jina-embeddings-v5-text-small'
    when it should just be 'jina-embeddings-v5-text-small').
    If such a URL is detected, this tries to extract the real model name from the
    query parameters; if it can't find one, it prints a warning and returns the
    value as-is (so the error at least comes back more clearly from the API)."""
    if not name or not name.lower().startswith("http"):
        return name
    try:
        parsed = urlparse(name)
        qs = parse_qs(parsed.query)
        if hint in qs and qs[hint]:
            extracted = qs[hint][0]
            print(
                f"[models] Warning: model name looks like a URL ('{name}'). "
                f"Using just '{extracted}' instead. We recommend fixing this "
                "value in your .env file (only the model name, not the page link)."
            )
            return extracted
    except Exception:
        pass
    print(
        f"[models] Warning: model name looks like a URL ('{name}') but the model "
        "name could not be extracted from it. Please check the CHAT_MODEL value "
        "in your .env file."
    )
    return name


# ═══════════════════════════════════════════════════════════════
# LOADING FROM THE .env FILE
# ═══════════════════════════════════════════════════════════════

def _load_env_file(filepath: str = ".env") -> None:
    """
    Manually reads the .env file and loads it into os.environ.
    Works without any external library dependency.
    """
    if not os.path.exists(filepath):
        return

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comment lines and empty lines
            if not line or line.startswith("#"):
                continue
            # Parse the KEY=VALUE format
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Strip surrounding quotes
                if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                # Only load if not already defined (don't override system env vars)
                if key and key not in os.environ:
                    os.environ[key] = value


# Load the .env file on application startup
_load_env_file()


# ═══════════════════════════════════════════════════════════════
# ENVIRONMENT VARIABLES (from .env file or system env)
# ═══════════════════════════════════════════════════════════════

GEMINI_API_KEY = os.getenv("GEMINI_API")
GROQ_API_KEY = os.getenv("GROQ_API")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# OpenRouter recommends these two headers for ranking/analytics purposes
# (not required; omitted if left empty). https://openrouter.ai/docs
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "").strip()
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "Talimci").strip()

# Default provider: gemini
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower().strip()

# Default model names
DEFAULT_GEMINI_CHAT_MODEL = "gemini-2.5-flash"
DEFAULT_GROQ_CHAT_MODEL = "llama3-8b-8192"
# NOTE: OpenRouter's ":free" model list changes/rotates over time
# (see openrouter.ai/models?max_price=0). The one below is a free model that
# was commonly offered as of when these lines were written (2026-07); if it
# gets removed in the future you may need to update CHAT_MODEL via .env to
# another ":free" model.
DEFAULT_OPENROUTER_CHAT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

# Can be overridden by the user via .env
CHAT_MODEL = os.getenv("CHAT_MODEL")


# ═══════════════════════════════════════════════════════════════
# RATE LIMITING
# ═══════════════════════════════════════════════════════════════
#
# Behavior differs by provider: Gemini generally limits requests per minute
# (RPM), while Groq limits tokens per minute (TPM). The limiter therefore
# tracks both request count and (as a rough estimate) token count, with each
# provider tracked separately against its own budget.

def estimate_tokens(text: str) -> int:
    """A rough token estimate that doesn't depend on a tokenizer.
    Assumes ~4 characters ≈ 1 token; the goal isn't exact measurement, just
    safely throttling against the TPM limit before hitting a 429."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class _RateLimiter:
    """Thread-safe 'sliding window' rate limiter. Enforces both requests/min
    (RPM) and an optional tokens/min (TPM) limit together.

    IMPORTANT: the 'check if budget allows it' and 'record the call against
    the budget' steps happen ATOMICALLY inside a SINGLE lock block. These two
    steps used to be in separate `with self._lock` blocks; this meant multiple
    concurrent requests (e.g. multiple user messages running on different
    threads in Chainlit) could each see a "budget available" result at the
    same time and both proceed before either one finished recording its call
    (a classic TOCTOU race condition) — as a result, the total number of
    requests the local limiter allowed could exceed the provider's actual
    limit, and 429s could still occur."""

    def __init__(self, max_calls_per_minute=4, max_tokens_per_minute=None):
        self.max_calls = max_calls_per_minute
        self.max_tokens = max_tokens_per_minute
        self.window = 60.0
        self._calls = deque()  # [(timestamp, token_estimate), ...]
        self._lock = threading.Lock()

    def acquire(self, token_estimate: int = 0):
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0][0] > self.window:
                    self._calls.popleft()

                over_calls = len(self._calls) >= self.max_calls
                current_tokens = sum(t for _, t in self._calls)
                over_tokens = (
                    self.max_tokens is not None
                    and current_tokens + token_estimate > self.max_tokens
                )

                if not over_calls and not over_tokens:
                    self._calls.append((time.monotonic(), token_estimate))
                    return

                if not self._calls:
                    # The deque is empty (no prior call to wait out) but this
                    # message alone exceeds the budget — meaning the message
                    # itself is larger than the limit (e.g. a very large
                    # prompt). This can't be resolved by waiting (it will
                    # never fit), so we print a warning and send the request
                    # as-is.
                    if self.max_tokens is not None and over_tokens:
                        print(
                            f"[rate limiter] Warning: a single request (~{token_estimate} estimated tokens) "
                            f"exceeds the per-minute budget ({self.max_tokens} tokens). Skipping the wait and "
                            "sending the request as-is — consider reducing the prompt size."
                        )
                    self._calls.append((time.monotonic(), token_estimate))
                    return

                wait_time = self.window - (now - self._calls[0][0]) + 0.1

            time.sleep(wait_time)


# Default per-provider limits — can be overridden via .env.
# The GROQ_TPM_LIMIT default (12000) matches the actual account limit
# observed for llama-3.3-70b-versatile; if you're using a different model or
# your account's limit has changed, update it in .env (this is also now
# automatically corrected from the error message when a 429 is received,
# see retry_on_network_error).
#
# The OPENROUTER_RPM_LIMIT default (18) leaves a small safety margin against
# OpenRouter's fixed "20 requests/minute" limit for free (":free") models.
# OpenRouter also has a daily quota (50 or 1000 requests/day depending on
# the account), but that's not something a per-minute limiter can prevent —
# if the daily quota is exhausted you'll get a 429, which
# retry_on_network_error will also retry a few times (with the default delay).
_PROVIDER_LIMITS = {
    "gemini": {
        "rpm": int(os.getenv("GEMINI_RPM_LIMIT", "4")),
        "tpm": int(os.getenv("GEMINI_TPM_LIMIT", "0")) or None,
    },
    "groq": {
        "rpm": int(os.getenv("GROQ_RPM_LIMIT", "28")),
        "tpm": int(os.getenv("GROQ_TPM_LIMIT", "12000")) or None,
    },
    "openrouter": {
        "rpm": int(os.getenv("OPENROUTER_RPM_LIMIT", "18")),
        "tpm": int(os.getenv("OPENROUTER_TPM_LIMIT", "0")) or None,
    },
}

_llm_limiters = {
    provider: _RateLimiter(max_calls_per_minute=cfg["rpm"], max_tokens_per_minute=cfg["tpm"])
    for provider, cfg in _PROVIDER_LIMITS.items()
}


def _get_limiter(provider: str) -> _RateLimiter:
    """Creates a reasonable default limiter (instead of a KeyError) for any
    new provider added in the future."""
    if provider not in _llm_limiters:
        _llm_limiters[provider] = _RateLimiter(max_calls_per_minute=15, max_tokens_per_minute=None)
    return _llm_limiters[provider]


def llm_rate_limit(provider: str, reserve_completion_tokens: int = 700):
    """Reserves space from the given provider's request/token budget before
    the function runs.

    IMPORTANT: it is NOT enough to only estimate the tokens of the current
    `message` parameter. In stateful conversations (create_chat +
    send_chat_message), the entire prior conversation history (self._messages)
    and the system prompt are resent with every request; if these aren't
    included in the estimate, the limiter will significantly undercount the
    real payload size and say "budget is fine" while sending a request that
    the provider still rejects with a 429. That's why self._messages and
    self._system_prompt (if present) are also factored into the estimate.

    Since response (completion) tokens also count against the TPM limit
    (Groq TPM = prompt + completion), a `reserve_completion_tokens` margin is
    also reserved — not the model's full requested max_tokens, but a
    reasonable/conservative estimate."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, message: str = "", *args, **kwargs):
            limiter = _get_limiter(provider)

            parts = [message]
            history = getattr(self, "_messages", None) or []
            parts.extend(m.get("content", "") for m in history)
            system_prompt = getattr(self, "_system_prompt", None)
            if system_prompt:
                parts.append(system_prompt)

            token_estimate = estimate_tokens(" ".join(parts)) + reserve_completion_tokens
            limiter.acquire(token_estimate)
            return func(self, message, *args, **kwargs)
        return wrapper
    return decorator


def _extract_retry_seconds(exc: requests.HTTPError, default: float) -> float:
    """Tries to figure out how long to wait based on a 429 response.
    First checks the standard 'Retry-After' header; since providers like Groq
    don't send that header and instead only embed the duration in the error
    message text (e.g. '...try again in 17.189999999s...'), it also tries to
    parse that duration from the message text. Falls back to `default` if
    neither is found."""
    response = getattr(exc, "response", None)
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass

    match = re.search(r"try again in\s*([\d.]+)\s*s", str(exc), re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    return default


def _extract_tpm_limit(exc: requests.HTTPError) -> Optional[int]:
    """Groq's TPM (tokens-per-minute) 429 error message also includes the
    account's/model's ACTUAL limit (e.g. '...on tokens per minute (TPM): Limit
    8000, Used 7098, Requested 1059...'). Different Groq models can have
    different TPM limits (see the GROQ_TPM_LIMIT description), and the user
    may not have manually adjusted this in .env to match whichever model
    they're using. So if we can capture the actual limit from the error
    message, we automatically correct the local rate limiter accordingly —
    this way, even if the configured value is wrong/stale, the same 429 loop
    doesn't keep happening over and over."""
    match = re.search(r"tokens per minute \(TPM\):\s*Limit\s*(\d+)", str(exc), re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return None


def retry_on_network_error(max_retries=5, delay=3, max_rate_limit_wait=600, provider: Optional[str] = None):
    """Decorator that retries on network errors.

    429 (rate limit) errors are handled specially: instead of retrying a
    limited number of times and then giving up, it waits for the duration
    specified in the error message itself (e.g. Groq's 'Please try again in
    17.19s' message) and resends the exact same request - and keeps doing so
    for as long as the provider keeps saying "not yet", since the quota is
    guaranteed to free up eventually. This wait does not consume the
    `max_retries` counter (that one is only for connection errors/503s).
    As a safety net against a truly unrecoverable situation (e.g. a single
    request that is permanently larger than the account's limit), the total
    time spent waiting on 429s is capped at `max_rate_limit_wait` seconds
    (default 10 minutes); only past that point does it give up and raise.

    If `provider` is given (e.g. "groq") and the actual TPM limit can be
    parsed from the 429 message (see `_extract_tpm_limit`), the relevant
    provider's local rate limiter is automatically lowered to that real
    value. This prevents the same 429 error from recurring over and over due
    to GROQ_TPM_LIMIT in .env staying higher than the real/current account
    limit (since different Groq models can have different TPM limits, e.g.
    one model at 12000, another at 8000) — it both fixes the issue and lets
    the user know they should update .env for a permanent fix.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            rate_limit_attempts = 0
            total_rate_limit_wait = 0.0
            while True:
                try:
                    return func(*args, **kwargs)
                except (requests.ConnectionError, requests.Timeout) as e:
                    wait = delay
                    attempts += 1
                    if attempts > max_retries:
                        raise
                    time.sleep(wait)
                except requests.HTTPError as e:
                    status = e.response.status_code if e.response else 0
                    if status == 429:
                        if provider:
                            real_limit = _extract_tpm_limit(e)
                            if real_limit is not None:
                                limiter = _get_limiter(provider)
                                if limiter.max_tokens is None or real_limit < limiter.max_tokens:
                                    print(
                                        f"[rate limiter] Your '{provider}' account's actual TPM limit "
                                        f"({real_limit}) is lower than the configured value ({limiter.max_tokens}); "
                                        f"the local limiter has been automatically adjusted to {real_limit} tokens/min. "
                                        "To fix this permanently, we recommend adding "
                                        f"{provider.upper()}_TPM_LIMIT={real_limit} to your .env file."
                                    )
                                    limiter.max_tokens = real_limit
                        wait = _extract_retry_seconds(e, default=delay) + 0.5
                        if total_rate_limit_wait + wait > max_rate_limit_wait:
                            print(
                                f"[retry] 429 rate limit persisted past the {max_rate_limit_wait}s safety cap "
                                f"({rate_limit_attempts} attempts); giving up."
                            )
                            raise
                        rate_limit_attempts += 1
                        total_rate_limit_wait += wait
                        print(
                            f"[retry] Received 429 rate limit; waiting {wait:.2f}s "
                            f"and resending the same request (attempt {rate_limit_attempts}, "
                            f"{total_rate_limit_wait:.1f}s/{max_rate_limit_wait}s total wait so far)..."
                        )
                        # Wait slightly longer than the duration specified by
                        # the provider, to avoid hitting another 429 right at
                        # the edge of the window.
                        time.sleep(wait)
                    elif status == 503:
                        attempts += 1
                        if attempts > max_retries:
                            raise
                        time.sleep(delay)
                    else:
                        raise
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class ChatResponse:
    """Data class that standardizes model responses."""
    text: str
    raw_response: Any = None
    model_name: str = ""
    provider: str = ""


# ═══════════════════════════════════════════════════════════════
# ABSTRACT BASE CLASS
# ═══════════════════════════════════════════════════════════════

class BaseChatModel(ABC):
    """Abstract base class for all chat models."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    def send_message(self, message: str) -> ChatResponse:
        """Sends a single message and returns the response."""
        pass

    @abstractmethod
    def create_chat(self, system_prompt: Optional[str] = None) -> Any:
        """Creates a chat session (stateful conversation)."""
        pass


# ═══════════════════════════════════════════════════════════════
# GEMINI IMPLEMENTATION (via requests)
# ═══════════════════════════════════════════════════════════════

class GeminiChatModel(BaseChatModel):
    """Google GenAI (Gemini) chat model — implemented with requests."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self._api_key = api_key or GEMINI_API_KEY
        if not self._api_key:
            raise ValueError("The GEMINI_API environment variable is not set.")

        self.model_name = _sanitize_model_name(model_name or CHAT_MODEL or DEFAULT_GEMINI_CHAT_MODEL)
        self._messages: List[dict] = []
        self._system_prompt: Optional[str] = None
        super().__init__(self.model_name, self._api_key)

    def _build_request_body(self, message: str, stream: bool = False) -> dict:
        """Builds the Gemini API request body."""
        contents = []

        for msg in self._messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        if self._system_prompt and not self._messages:
            contents.append({
                "role": "user",
                "parts": [{"text": f"System: {self._system_prompt}"}]
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "Understood."}]
            })

        contents.append({
            "role": "user",
            "parts": [{"text": message}]
        })

        body = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 4096,
            }
        }

        if stream:
            body["generationConfig"]["responseMimeType"] = "text/plain"

        return body

    @llm_rate_limit("gemini")
    @retry_on_network_error(max_retries=5, delay=3, provider="gemini")
    def send_message(self, message: str) -> ChatResponse:
        """Sends a single, stateless message."""
        url = f"{self.BASE_URL}/models/{self.model_name}:generateContent"
        params = {"key": self._api_key}
        body = self._build_request_body(message)

        response = requests.post(url, params=params, json=body, timeout=60)
        response.raise_for_status()

        data = response.json()
        text = self._extract_text(data)

        return ChatResponse(
            text=text,
            raw_response=data,
            model_name=self.model_name,
            provider="gemini"
        )

    def create_chat(self, system_prompt: Optional[str] = None):
        """Creates a stateful chat session."""
        self._messages = []
        self._system_prompt = system_prompt
        return self

    @llm_rate_limit("gemini")
    @retry_on_network_error(max_retries=5, delay=3, provider="gemini")
    def send_chat_message(self, message: str) -> ChatResponse:
        """Sends a message to the active chat session."""
        url = f"{self.BASE_URL}/models/{self.model_name}:generateContent"
        params = {"key": self._api_key}
        body = self._build_request_body(message)

        response = requests.post(url, params=params, json=body, timeout=60)
        response.raise_for_status()

        data = response.json()
        text = self._extract_text(data)

        self._messages.append({"role": "user", "content": message})
        self._messages.append({"role": "assistant", "content": text})

        return ChatResponse(
            text=text,
            raw_response=data,
            model_name=self.model_name,
            provider="gemini"
        )

    @staticmethod
    def _extract_text(data: dict) -> str:
        """Extracts text from the API response."""
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            return ""
        return parts[0].get("text", "")


# ═══════════════════════════════════════════════════════════════
# GROQ IMPLEMENTATION (via requests)
# ═══════════════════════════════════════════════════════════════

class GroqChatModel(BaseChatModel):
    """Groq API chat model — implemented with requests."""

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self._api_key = api_key or GROQ_API_KEY
        if not self._api_key:
            raise ValueError("The GROQ_API environment variable is not set.")

        self.model_name = _sanitize_model_name(model_name or CHAT_MODEL or DEFAULT_GROQ_CHAT_MODEL)
        self._messages: List[dict] = []
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        super().__init__(self.model_name, self._api_key)

    def _build_messages(self, user_message: str) -> List[dict]:
        """Builds a message list in Groq API format."""
        messages = self._messages.copy()
        messages.append({"role": "user", "content": user_message})
        return messages

    @llm_rate_limit("groq")
    @retry_on_network_error(max_retries=5, delay=3, provider="groq")
    def send_message(self, message: str) -> ChatResponse:
        """Sends a single, stateless message."""
        url = f"{self.BASE_URL}/chat/completions"
        messages = [{"role": "user", "content": message}]

        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048
        }

        response = requests.post(url, headers=self._headers, json=body, timeout=60)
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        return ChatResponse(
            text=content,
            raw_response=data,
            model_name=self.model_name,
            provider="groq"
        )

    def create_chat(self, system_prompt: Optional[str] = None):
        """Starts a stateful chat session."""
        self._messages = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})
        return self

    @llm_rate_limit("groq")
    @retry_on_network_error(max_retries=5, delay=3, provider="groq")
    def send_chat_message(self, message: str) -> ChatResponse:
        """Sends a message to the active chat session."""
        url = f"{self.BASE_URL}/chat/completions"
        messages = self._build_messages(message)

        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048
        }

        response = requests.post(url, headers=self._headers, json=body, timeout=60)

        if response.status_code != 200:
            try:
                error_json = response.json()
                error_msg = error_json.get('error', {}).get('message', response.text)
                error_type = error_json.get('error', {}).get('type', 'unknown')
            except:
                error_msg = response.text
                error_type = 'unknown'

            print(f"\n{'='*60}")
            print(f"GROQ API ERROR {response.status_code}")
            print(f"Error Type: {error_type}")
            print(f"Error Message: {error_msg}")
            print(f"Request Model: {self.model_name}")
            print(f"Messages Count: {len(messages)}")
            print(f"{'='*60}\n")

            raise requests.HTTPError(
                f"Groq API {response.status_code}: {error_msg}",
                response=response
            )

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        self._messages.append({"role": "user", "content": message})
        self._messages.append({"role": "assistant", "content": content})

        return ChatResponse(
            text=content,
            raw_response=data,
            model_name=self.model_name,
            provider="groq"
        )


# ═══════════════════════════════════════════════════════════════
# OPENROUTER IMPLEMENTATION (via requests, OpenAI-compatible)
# ═══════════════════════════════════════════════════════════════

class OpenRouterChatModel(BaseChatModel):
    """OpenRouter API chat model — implemented with requests.

    OpenRouter provides access to dozens of providers' models (Meta, Google,
    Alibaba, etc.) through a single endpoint compatible with OpenAI's
    /chat/completions format. Model names suffixed with ':free' (e.g.
    'meta-llama/llama-3.3-70b-instruct:free') are entirely free and require
    no credit card; in exchange, a fixed "20 requests/minute" limit and a
    "50 or 1000 requests/day" limit (depending on the account) apply (see
    OPENROUTER_RPM_LIMIT).

    Difference from Groq: Groq limited by tokens per minute (TPM) (a limit
    that varied by model and couldn't be known in advance); OpenRouter's free
    tier instead uses a fixed, known-in-advance requests (RPM) limit, which
    makes it much easier/more reliable to configure the local limiter
    correctly ahead of time.

    Setup (environment variables):
        OPENROUTER_API_KEY  - API key obtained without a credit card via
                               https://openrouter.ai/settings/keys
        CHAT_MODEL           - e.g. 'meta-llama/llama-3.3-70b-instruct:free'
                               (if left empty, DEFAULT_OPENROUTER_CHAT_MODEL
                               is used - since the free model list changes
                               over time, we recommend checking
                               openrouter.ai/models for a current ':free'
                               model)
        OPENROUTER_SITE_URL  - (optional) header OpenRouter recommends for
                               ranking/traffic-source tracking
        OPENROUTER_APP_NAME  - (optional) app name for the same purpose
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self._api_key = api_key or OPENROUTER_API_KEY
        if not self._api_key:
            raise ValueError("The OPENROUTER_API_KEY environment variable is not set.")

        self.model_name = _sanitize_model_name(model_name or CHAT_MODEL or DEFAULT_OPENROUTER_CHAT_MODEL)
        self._messages: List[dict] = []
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        # OpenRouter's recommended (optional) ranking headers - only added
        # if a value is present.
        if OPENROUTER_SITE_URL:
            self._headers["HTTP-Referer"] = OPENROUTER_SITE_URL
        if OPENROUTER_APP_NAME:
            self._headers["X-Title"] = OPENROUTER_APP_NAME

        super().__init__(self.model_name, self._api_key)

    def _build_messages(self, user_message: str) -> List[dict]:
        messages = self._messages.copy()
        messages.append({"role": "user", "content": user_message})
        return messages

    def _post(self, messages: List[dict]) -> ChatResponse:
        url = f"{self.BASE_URL}/chat/completions"
        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        response = requests.post(url, headers=self._headers, json=body, timeout=60)

        if response.status_code != 200:
            try:
                error_json = response.json()
                error_msg = error_json.get("error", {}).get("message", response.text)
            except Exception:
                error_msg = response.text

            print(f"\n{'='*60}")
            print(f"OPENROUTER API ERROR {response.status_code}")
            print(f"Error Message: {error_msg}")
            print(f"Request Model: {self.model_name}")
            print(f"{'='*60}\n")

            raise requests.HTTPError(
                f"OpenRouter API {response.status_code}: {error_msg}",
                response=response,
            )

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return ChatResponse(
            text=content,
            raw_response=data,
            model_name=self.model_name,
            provider="openrouter",
        )

    @llm_rate_limit("openrouter")
    @retry_on_network_error(max_retries=5, delay=3, provider="openrouter")
    def send_message(self, message: str) -> ChatResponse:
        """Sends a single, stateless message."""
        return self._post([{"role": "user", "content": message}])

    def create_chat(self, system_prompt: Optional[str] = None):
        """Starts a stateful chat session."""
        self._messages = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})
        return self

    @llm_rate_limit("openrouter")
    @retry_on_network_error(max_retries=5, delay=3, provider="openrouter")
    def send_chat_message(self, message: str) -> ChatResponse:
        """Sends a message to the active chat session."""
        messages = self._build_messages(message)
        result = self._post(messages)

        self._messages.append({"role": "user", "content": message})
        self._messages.append({"role": "assistant", "content": result.text})

        return result


# ═══════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_chat_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> BaseChatModel:
    """
    Returns a chat model based on the configuration.

    Args:
        provider: "gemini", "groq", or "openrouter". If None, the LLM_PROVIDER env variable is used.
        model_name: A specific model name. If None, the CHAT_MODEL env variable is used.
        api_key: A custom API key. If None, the relevant env variable is used.

    Returns:
        BaseChatModel: The configured chat model.
    """
    prov = (provider or LLM_PROVIDER).lower().strip()

    if prov == "groq":
        return GroqChatModel(model_name=model_name, api_key=api_key)
    elif prov == "gemini":
        return GeminiChatModel(model_name=model_name, api_key=api_key)
    elif prov == "openrouter":
        return OpenRouterChatModel(model_name=model_name, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: '{prov}'. Use 'gemini', 'groq', or 'openrouter'.")


def get_provider_name() -> str:
    """Returns the name of the active LLM provider."""
    return LLM_PROVIDER


def get_active_model_name() -> str:
    """Returns the name of the active chat model."""
    if LLM_PROVIDER == "groq":
        return CHAT_MODEL or DEFAULT_GROQ_CHAT_MODEL
    if LLM_PROVIDER == "openrouter":
        return CHAT_MODEL or DEFAULT_OPENROUTER_CHAT_MODEL
    return CHAT_MODEL or DEFAULT_GEMINI_CHAT_MODEL
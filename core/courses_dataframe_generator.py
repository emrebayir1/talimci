"""
Provides asynchronous retrieval, cleaning, and aggregation of online course data
from Udemy, Coursera, and YouTube, returning a structured pandas DataFrame of
course information.
"""
import os
import asyncio
import threading
import time
import logging
from dotenv import load_dotenv
import pandas as pd
from typing import List, Optional, Any
from ddgs import DDGS
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from lingua import LanguageDetectorBuilder
import re

logger = logging.getLogger("courses_dataframe_generator")

# Utility functions
def to_float(val, default=0):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.replace(',', ''))
        except ValueError:
            return default
    return default

def clean_text(text: str) -> str:
    if not text:
        return ""

    # Remove URLs (http/https links)
    text = re.sub(r'https?://[^\s]+', '', text)

    # Remove www links
    text = re.sub(r'www\.[^\s]+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove emojis and special Unicode characters
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  
        "\U0001F300-\U0001F5FF"  
        "\U0001F680-\U0001F6FF"  
        "\U0001F1E0-\U0001F1FF"  
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  
        "\u3030"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)

    # Remove extra whitespaces and newlines
    text = re.sub(r'\s+', ' ', text)

    # Remove common social media handles and hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)

    # Remove HTML entities that might have been missed
    text = re.sub(r'&[a-zA-Z0-9]+;', '', text)

    # Remove brackets with links or references like [1], (link), etc.
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\([^)]*link[^)]*\)', '', text, flags=re.IGNORECASE)

    # Remove excessive punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)

    # Clean up and return
    return text.strip()

class _DuckDuckGoRateLimiter:
    """Since DuckDuckGo does not have an official API, no strict request limit
    is specified, but we leave a small safety margin to avoid making requests too quickly
    back-to-back when multiple search terms are triggered simultaneously (the semaphore
    inside generate_courses_dataframe) and getting temporarily banned/rate-limited.
    This is not a strict API requirement, but a precautionary throttle."""

    def __init__(self, min_interval: float = 1.0):
        self.min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()


# Single limiter shared across the process - kept at the module level
# so that every new DuckDuckGoSearch/Udemy/Coursera instance doesn't create its own limiter.
_ddg_rate_limiter = _DuckDuckGoRateLimiter()


class DuckDuckGoSearch:
    """
    Common client that fetches DuckDuckGo search results restricted to a single domain
    via the `site:` operator using the `ddgs` library (formerly `duckduckgo_search`).
    Both Udemy and Coursera classes use this client.

    Why DuckDuckGo (instead of Serper/Brave/Google CSE):
    - Udemy's official affiliate API was discontinued; previously, Google Custom Search
      was used instead, but its free quota is limited to only 100 queries per day.
    - Coursera's unofficial endpoints like courses.v1/onDemandCourses.v1 could also be
      silently changed and restricted by Coursera.
    - Brave Search API was tried, but Brave now requires a payment method (credit card)
      for new accounts.
    - Serper.dev's free quota (2500 queries) was a one-time starter quota, and upon exhaustion,
      switching to paid credits was required.
    - `ddgs` returns public search results from DuckDuckGo without any API key, registration,
      or quota; the `site:` operator allows using the same mechanism for both Udemy and Coursera.
      (Note: Since this is not an official API, temporary rate limits/blocks may occur due to
      changes on DuckDuckGo's side or aggressive usage; therefore, requests are throttled with
      `_ddg_rate_limiter`.)

    Setup: Not required - `pip install ddgs` is enough, no API key needed.
    `is_configured` therefore always returns True; this field is kept solely to maintain
    the same interface as SerperSearch and ensure calling code (Udemy/Coursera) works unchanged.
    """

    def __init__(self):
        self._is_configured = True

    @property
    def is_configured(self) -> bool:
        return self._is_configured

    async def search_site(self, query: str, site: str, max_results: int = 10) -> List[dict]:
        """Searches with the `site:{site} {query}` query and returns a list of
        [{"title", "url", "description"}, ...]."""
        await _ddg_rate_limiter.acquire()

        search_query = f"site:{site} {query}"

        try:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                None,
                lambda: DDGS().text(search_query, max_results=max_results),
            )
        except Exception:
            logger.exception("DuckDuckGo Search request failed: '%s'", search_query)
            return []

        return [
            {
                "title": item.get("title", ""),
                "url": item.get("href", ""),
                "description": item.get("body", ""),
            }
            for item in (results or [])
        ]


class Udemy:
    """
    DOES NOT USE Udemy's own API. The official Affiliate API was discontinued
    (even with an affiliate account, API client approval was a separate process and usually
    resulted in a 403 Forbidden). Instead, public web searches restricted to `site:udemy.com/course`
    are performed using the common `DuckDuckGoSearch` client (Google Custom Search was tried first,
    then Brave, then Serper).

    IMPORTANT LIMITATIONS:
    - This only searches public Udemy pages returned by DuckDuckGo; it is not a direct query to
      Udemy's own catalog. Therefore, structured fields like price, instructor name, and language
      DO NOT COME THROUGH — only title, link, and search result snippet are obtained.
      Since `is_paid` is unknown, it returns None here; the calling function
      (fetch_course_dataframe) applies a reasonable assumption (courses on Udemy are mostly paid).

    Setup: Not required - DuckDuckGoSearch does not require an API key, so
    `is_configured` always returns True.
    """

    def __init__(self, ddg: Optional["DuckDuckGoSearch"] = None):
        self.ddg = ddg or DuckDuckGoSearch()
        self._course_cache = {}

    @property
    def is_configured(self) -> bool:
        return self.ddg.is_configured

    async def retrieve_udemy_courses(self, search_term: str, max_results: int = 10) -> List[dict]:
        if not self.is_configured:
            return []

        if search_term in self._course_cache:
            return self._course_cache[search_term]

        results = await self.ddg.search_site(search_term, site="udemy.com/course", max_results=max_results)

        courses = []
        for item in results:
            link = item.get("url", "")
            if "udemy.com/course" not in link:
                continue
            courses.append({
                "id": link,
                "title": clean_text(item.get("title", "")),
                "instructors": "",
                "url": link,
                "language": "",
                "price": None,
                "price_currency": "",
                "description": clean_text(item.get("description", "")),
                "content_info": "",
                "is_paid": None,  # unknown - calling side applies assumption
            })

        logger.info(
            "DuckDuckGo Search (Udemy) '%s': %d results found.",
            search_term, len(courses),
        )
        self._course_cache[search_term] = courses
        return courses


class Coursera:
    """
    DOES NOT USE Coursera's own API. Previously, unofficial endpoints like `courses.v1` /
    `onDemandCourses.v1` (which do not require OAuth but could be changed and restricted by
    Coursera at any time) were tried; as of 2026-07, their `search` finder started returning
    405 with "Routing error: finder 'search' not implemented".
    Instead, public web searches restricted to `site:coursera.org/learn` are performed
    using the common `DuckDuckGoSearch` client.

    IMPORTANT LIMITATIONS:
    - This only searches public Coursera pages returned by DuckDuckGo; it is not a direct query to
      Coursera's own catalog. Since some fields like instructor name do not come from search results,
      `instructors` is left blank. `is_paid` is assumed to be False by default since most courses on Coursera
      have a free "audit" mode (only the certificate is paid) — this is not definitive information, but a reasonable assumption.

    Setup: Not required - DuckDuckGoSearch does not require an API key, so
    `is_configured` always returns True.
    """

    def __init__(self, ddg: Optional["DuckDuckGoSearch"] = None):
        self.ddg = ddg or DuckDuckGoSearch()
        self._course_cache = {}

    @property
    def is_configured(self) -> bool:
        return self.ddg.is_configured

    async def retrieve_courses(self, search_term: str, max_results: int = 15) -> List[dict]:
        if not self.is_configured:
            return []

        if search_term in self._course_cache:
            return self._course_cache[search_term]

        results = await self.ddg.search_site(search_term, site="coursera.org/learn", max_results=max_results)

        courses = []
        for item in results:
            link = item.get("url", "")
            if "coursera.org/learn" not in link:
                continue
            courses.append({
                "id": link,
                "title": clean_text(item.get("title", "")),
                "instructors": "",
                "url": link,
                "language": "",
                "price": None,
                "price_currency": "",
                "description": clean_text(item.get("description", "")),
                "content_info": "",
                "is_paid": False,
            })

        logger.info(
            "DuckDuckGo Search (Coursera) '%s': %d results found.",
            search_term, len(courses),
        )
        self._course_cache[search_term] = courses
        return courses


class YouTube:
    """
    YouTube API wrapper.

    Same approach as youtube_manager.py in the Hey DJ project: multiple
    YouTube API keys/accounts can be defined, searches are distributed among them
    round-robin style (other accounts don't sit idle until a single account is exhausted),
    and when an account exhausts its daily quota, it automatically switches to the next account.
    """

    _exhausted_keys: set = set()

    _rr_index: int = 0
    _rr_lock = threading.Lock()

    def __init__(self, api_keys: Optional[List[str]] = None):
        load_dotenv()
        self.api_keys = api_keys or self._load_api_keys()
        if not self.api_keys:
            raise ValueError("YouTube API key required")

        self.language_detector = LanguageDetectorBuilder.from_all_languages().build()
        self._transcript_cache = {}
        self._clients = {}  # key -> googleapiclient service (lazily built, reused)

    @staticmethod
    def _load_api_keys() -> List[str]:
        """YOUTUBE_API_KEYS (comma-separated, multiple accounts) takes priority;
        otherwise falls back to singular YOUTUBE_API for backwards compatibility."""
        raw = os.getenv("YOUTUBE_API_KEYS", "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if keys:
            return keys
        single = os.getenv("YOUTUBE_API", "").strip()
        return [single] if single else []

    def _keys_to_try(self) -> List[str]:
        """Returns configured keys starting from a rotating starting point (round-robin)
        across calls, so searches start spread across all accounts instead of always
        using the same account. Keys known to be quota-exhausted are moved to the end
        of the list (though tried as a last resort just in case quotas reset daily)."""
        n = len(self.api_keys)
        if n == 0:
            return []

        with self._rr_lock:
            start = YouTube._rr_index % n
            YouTube._rr_index += 1

        rotated = self.api_keys[start:] + self.api_keys[:start]
        fresh = [k for k in rotated if k not in self._exhausted_keys]
        exhausted = [k for k in rotated if k in self._exhausted_keys]
        return fresh + exhausted

    def _client_for(self, key: str):
        client = self._clients.get(key)
        if client is None:
            client = build('youtube', 'v3', developerKey=key)
            self._clients[key] = client
        return client

    @staticmethod
    def _mask(key: str) -> str:
        return f"...{key[-6:]}" if len(key) > 6 else "***"

    @staticmethod
    def _is_quota_error(e: HttpError) -> bool:
        """Recognizes 403 quotaExceeded/dailyLimitExceeded or 429 as a quota error;
        distinguishes from other 403 reasons (e.g. invalid/restricted key) so the next
        account is still tried in those cases."""
        status = getattr(getattr(e, "resp", None), "status", None)
        if status == 429:
            return True
        if status != 403:
            return False
        try:
            content = e.content.decode("utf-8", errors="ignore") if isinstance(e.content, bytes) else str(e.content)
        except Exception:
            content = str(e)
        return "quota" in content.lower()

    def get_language(self, text: str) -> str:
        if len(text) < 10:
            return 'en'

        try:
            language = self.language_detector.detect_language_of(text[:200])
            return str(language.iso_code_639_1.name.lower())
        except:
            return 'en'


    async def search_videos(self, keyword: str, max_results: int = 15) -> List[dict]:
        request_params = {
            'q': keyword,
            'part': 'snippet',
            'type': 'video',
            'maxResults': max_results,
            'order': 'relevance',
            'videoDuration': 'long'
        }

        for key in self._keys_to_try():
            videos = []
            video_ids = set()
            try:
                youtube = self._client_for(key)
                response = youtube.search().list(**request_params).execute()

                for item in response['items']:
                    vid_id = item['id']['videoId']
                    if vid_id not in video_ids:
                        video_ids.add(vid_id)
                        videos.append({
                            'videoId': vid_id,
                            'title': clean_text(item['snippet']['title']),
                            'channel': clean_text(item['snippet']['channelTitle']),
                            'url': f"https://www.youtube.com/watch?v={vid_id}"
                        })

                if video_ids:
                    details = youtube.videos().list(
                        part='snippet,statistics,contentDetails',
                        id=','.join(video_ids)
                    ).execute()

                    id_to_details = {item['id']: item for item in details['items']}
                    tasks = [self._process_video_details(video, id_to_details) for video in videos]
                    videos = await asyncio.gather(*tasks)

                return videos

            except HttpError as e:
                if self._is_quota_error(e):
                    self._exhausted_keys.add(key)
                    print(f"YouTube account {self._mask(key)} exhausted its daily quota; trying the next account.")
                else:
                    print(f"YouTube search error (account {self._mask(key)}): {e}")
                continue
            except Exception as e:
                print(f"YouTube search error (account {self._mask(key)}): {e}")
                continue

        print(f"All configured YouTube accounts were tried, no results obtained for '{keyword}'.")
        return []

    async def _process_video_details(self, video: dict, id_to_details: dict) -> dict:
        vid_id = video['videoId']
        stats = id_to_details.get(vid_id, {})

        # Clean description
        raw_description = stats.get('snippet', {}).get('description', '')
        video['description'] = clean_text(raw_description)

        lang = (stats.get('snippet', {}).get('defaultAudioLanguage') or
                stats.get('snippet', {}).get('defaultLanguage'))
        if not lang:
            lang = self.get_language(video['title'])
        video['language'] = lang

        return video

async def fetch_course_dataframe(search_term: str) -> pd.DataFrame:
    # Udemy and Coursera now share the same DuckDuckGo (ddgs) client
    # (no API key required, uses a single rate limiter).
    ddg = DuckDuckGoSearch()
    udemy = Udemy(ddg)
    coursera = Coursera(ddg)

    try:
        youtube = YouTube()
    except ValueError:
        youtube = None

    tasks = []

    if udemy.is_configured:
        udemy_task = asyncio.create_task(
            udemy.retrieve_udemy_courses(search_term, max_results=15)
        )
        tasks.append(('udemy', udemy_task))

    if coursera.is_configured:
        coursera_task = asyncio.create_task(
            coursera.retrieve_courses(search_term, max_results=15)
        )
        tasks.append(('coursera', coursera_task))

    if youtube:
        youtube_task = asyncio.create_task(
            youtube.search_videos(search_term, max_results=10)
        )
        tasks.append(('youtube', youtube_task))

    rows = []

    for platform, task in tasks:
        try:
            courses = await task

            for c in courses:
                if not c:
                    continue

                title = c.get('title', '')
                if not title:
                    continue

                if platform in ('udemy', 'coursera'):
                    # Create clean text for Udemy/Coursera courses
                    content_text = f"{c.get('description', '')} {c.get('content_info', '')}"
                    clean_content = clean_text(content_text)

                    is_paid = c.get('is_paid')
                    if is_paid is None:
                        # Google CSE fallback path (Udemy) does not provide price info;
                        # since a large portion of the Udemy catalog is paid, we accept True
                        # as a reasonable assumption — this is not definitive information.
                        is_paid = to_float(c.get('price', 0)) > 0 if c.get('price') is not None else True

                    row = {
                        'platform': platform,
                        'title': title,
                        'channel_or_instructor': c.get('instructors', ''),
                        'url': c.get('url', ''),
                        'is_paid': is_paid,
                        'language': c.get('language', ''),
                        'description': c.get('description',''),
                        'text': f"Title: {title}, Content: {clean_content}",
                    }

                elif platform == 'youtube':
                    # Create clean text for YouTube videos
                    content_text = f"{c.get('description', '')}"
                    clean_content = clean_text(content_text)

                    row = {
                        'platform': 'youtube',
                        'title': title,
                        'channel_or_instructor': c.get('channel', ''),
                        'url': c.get('url', ''),
                        'is_paid': False,
                        'language': c.get('language', ''),
                        'description': clean_content,
                        'text': f"Title: {title}, Content: {clean_content}",
                    }

                if row:
                    rows.append(row)

        except Exception as e:
            print(f"Error processing {platform}: {e}")
            continue

    return pd.DataFrame(rows)

async def generate_courses_dataframe(search_terms: List[str]) -> pd.DataFrame:

    semaphore = asyncio.Semaphore(3)

    async def fetch_with_semaphore(term):
        async with semaphore:
            return await fetch_course_dataframe(term)

    tasks = [fetch_with_semaphore(term) for term in search_terms]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_dfs = [df for df in results if isinstance(df, pd.DataFrame) and not df.empty]

    if not valid_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(valid_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['url'], keep='first')

    return combined_df
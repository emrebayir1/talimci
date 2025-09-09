"""
Provides asynchronous retrieval, cleaning, and aggregation of online course data
from Udemy and YouTube, returning a structured pandas DataFrame of course information.
"""
import os
import asyncio
import aiohttp
import random
from dotenv import load_dotenv
import pandas as pd
from typing import List, Optional, Any
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from lingua import LanguageDetectorBuilder
import re

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

class Udemy:

    def __init__(self, base_url: str = "https://www.udemy.com/api-2.0/"):
        self.base_url = base_url
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.udemy.com/",
            "Origin": "https://www.udemy.com",
            "DNT": "1",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin"
        }
        # Cache for avoiding repeated requests
        self._search_cache = {}
        self._course_cache = {}

    @staticmethod
    def safe_get(data: dict, keys: list, default: Optional[Any] = '') -> Any:
        if data is None:
            return default

        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    async def _make_request_with_circuit_breaker(self, session: aiohttp.ClientSession, url: str,
                                                 max_retries: int = 2) -> dict:
        """
        Request method with circuit breaker pattern.
        """
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = random.uniform(0.5, 1.5) * attempt
                    await asyncio.sleep(delay)

                async with session.get(url=url, headers=self.headers) as response:
                    if response.status == 429:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(random.uniform(2, 5))
                            continue
                        else:
                            return {}

                    response.raise_for_status()

                    content_type = response.headers.get('content-type', '').lower()
                    if 'application/json' not in content_type:
                        return {}

                    data = await response.json()
                    return data

            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(random.uniform(0.5, 2))
                    continue
                else:
                    return {}

        return {}

    async def search_cached(self, session: aiohttp.ClientSession, search_term: str) -> dict:
        if search_term in self._search_cache:
            return self._search_cache[search_term]

        query = search_term.replace(" ", "+")
        search_url = self.base_url + f"search-suggestions/?q={query}"

        result = await self._make_request_with_circuit_breaker(session, search_url)
        self._search_cache[search_term] = result
        return result

    async def get_alternative_queries(self, session: aiohttp.ClientSession, search_term: str) -> List[str]:
        try:
            query = search_term.replace(" ", "+")
            alternative_query_url = self.base_url + f"related-searches/?q={query}"

            data = await self._make_request_with_circuit_breaker(session, alternative_query_url)
            return [item['phrase'] for item in data.get('related_searches', [])[:3]]
        except:
            return []

    async def filter_search_results_batch(self, session: aiohttp.ClientSession, search_terms: List[str]) -> List[dict]:
        tasks = [self.search_cached(session, term) for term in search_terms]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        filtered_results = []
        for result in all_results:
            if isinstance(result, dict) and 'results' in result:
                filtered = [
                    {'id': item['id'], '_class': item['_class']}
                    for item in result['results']
                    if item.get('id', 0) != 0
                ]
                filtered_results.extend(filtered)

        return filtered_results

    async def get_user_courses(self, session: aiohttp.ClientSession, user_id: int) -> List[int]:
        course_ids = []
        url = self.base_url + f"users/{user_id}/taught-profile-courses/"
        page_count = 0
        max_pages = 3

        while url and page_count < max_pages:
            data = await self._make_request_with_circuit_breaker(session, url)
            if not data:
                break

            results = data.get("results", [])
            ids = [item['id'] for item in results if item.get('_class') == 'course']
            course_ids.extend(ids)

            url = data.get("next")
            page_count += 1

        return course_ids

    async def get_course_info(self, session: aiohttp.ClientSession, course_id: int) -> dict:
        if course_id in self._course_cache:
            return self._course_cache[course_id]

        course_url = (
            f"{self.base_url}courses/{course_id}/?fields[course]="
            "title,headline,url,price_detail,visible_instructors,locale,description,content_info"
        )

        course_data = await self._make_request_with_circuit_breaker(session, course_url)
        if not course_data:
            return {}

        # Clean description text
        raw_description = self.safe_get(course_data, ['description'], '')
        clean_description = clean_text(
            BeautifulSoup(raw_description, "html.parser").get_text(separator=" ", strip=True))

        course = {
            'id': self.safe_get(course_data, ['id']),
            'title': clean_text(self.safe_get(course_data, ['title'])),
            'instructors': ', '.join([
                clean_text(self.safe_get(item, ['title']))
                for item in self.safe_get(course_data, ['visible_instructors'], default=[])
            ]),
            'url': 'https://www.udemy.com' + self.safe_get(course_data, ['url']),
            'language': self.safe_get(course_data, ['locale', 'locale']),
            'price': self.safe_get(course_data, ['price_detail', 'amount']),
            'price_currency': self.safe_get(course_data, ['price_detail', 'currency']),
            'description': clean_description,
            'content_info': clean_text(str(self.safe_get(course_data, ['content_info'], ''))),
        }

        self._course_cache[course_id] = course
        return course

    async def get_udemy_data(self, session: aiohttp.ClientSession, search_term: str) -> List[int]:
        alternative_queries = await self.get_alternative_queries(session, search_term)
        queries = [search_term] + alternative_queries[:2]

        all_results = await self.filter_search_results_batch(session, queries)

        course_ids = set()
        user_ids = []

        for result in all_results:
            if result['_class'] == 'course':
                course_ids.add(result['id'])
            elif result['_class'] == 'user':
                user_ids.append(result['id'])

        if user_ids:
            user_ids = user_ids[:5]
            user_tasks = [self.get_user_courses(session, user_id) for user_id in user_ids]
            user_results = await asyncio.gather(*user_tasks, return_exceptions=True)

            for result in user_results:
                if isinstance(result, list):
                    course_ids.update(result)

        return list(course_ids)

    async def retrieve_udemy_courses(self, search_term: str, max_concurrent: int = 10) -> List[dict]:
        connector = aiohttp.TCPConnector(
            limit=max_concurrent * 2,
            limit_per_host=max_concurrent,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
            ttl_dns_cache=300
        )
        timeout = aiohttp.ClientTimeout(total=30, connect=5)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            try:
                course_ids = await self.get_udemy_data(session, search_term)

                if not course_ids:
                    return []

                course_ids = list(set(course_ids))[:50]

                semaphore = asyncio.Semaphore(max_concurrent)

                async def get_course_with_semaphore(course_id):
                    async with semaphore:
                        return await self.get_course_info(session, course_id)

                tasks = [get_course_with_semaphore(course_id) for course_id in course_ids]
                courses = await asyncio.gather(*tasks, return_exceptions=True)

                valid_courses = [
                    course for course in courses
                    if isinstance(course, dict) and course.get('id')
                ]

                return valid_courses

            except Exception as e:
                print(f"Udemy retrieval error: {e}")
                return []

class YouTube:
    """
    YouTube API wrapper.
    """

    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.getenv("YOUTUBE_API")
        if not self.api_key:
            raise ValueError("YouTube API key required")

        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.language_detector = LanguageDetectorBuilder.from_all_languages().build()
        self._transcript_cache = {}

    def get_language(self, text: str) -> str:
        if len(text) < 10:
            return 'en'

        try:
            language = self.language_detector.detect_language_of(text[:200])
            return str(language.iso_code_639_1.name.lower())
        except:
            return 'en'


    async def search_videos(self, keyword: str, max_results: int = 15) -> List[dict]:
        videos = []
        video_ids = set()

        request_params = {
            'q': keyword,
            'part': 'snippet',
            'type': 'video',
            'maxResults': max_results,
            'order': 'relevance',
            'videoDuration': 'long'
        }

        try:
            response = self.youtube.search().list(**request_params).execute()

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
                details = self.youtube.videos().list(
                    part='snippet,statistics,contentDetails',
                    id=','.join(video_ids)
                ).execute()

                id_to_details = {item['id']: item for item in details['items']}
                tasks = [self._process_video_details(video, id_to_details) for video in videos]
                videos = await asyncio.gather(*tasks)

            return videos

        except Exception as e:
            print(f"YouTube search error: {e}")
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
    udemy = Udemy()

    try:
        youtube = YouTube()
    except ValueError:
        youtube = None

    tasks = []

    udemy_task = asyncio.create_task(
        udemy.retrieve_udemy_courses(search_term, max_concurrent=15)
    )
    tasks.append(('udemy', udemy_task))

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

                if platform == 'udemy':
                    # Create clean text for Udemy courses
                    content_text = f"{c.get('description', '')} {c.get('content_info', '')}"
                    clean_content = clean_text(content_text)

                    row = {
                        'platform': 'udemy',
                        'title': title,
                        'channel_or_instructor': c.get('instructors', ''),
                        'url': c.get('url', ''),
                        'is_paid': to_float(c.get('price', 0)) > 0,
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

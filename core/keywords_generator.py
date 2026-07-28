"""
Extracts relevant, platform-optimized search keywords from educational course titles
using a chat model and a robust list-parsing utility.
"""

import os
import ast
import re
from dotenv import load_dotenv
from typing import List
from core.learning_session import LearningSession
from utils.models import get_chat_model

load_dotenv()


def _parse_list_response(text: str) -> list:
    cleaned = text.strip()
    cleaned = re.sub(r'^```(?:python|json)?\s*', '', cleaned)
    cleaned = re.sub(r'```$', '', cleaned).strip()

    try:
        result = ast.literal_eval(cleaned)
        if isinstance(result, list):
            return [str(item).strip() for item in result if str(item).strip()]
    except (ValueError, SyntaxError):
        pass

    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if match:
        try:
            result = ast.literal_eval(match.group(0))
            if isinstance(result, list):
                return [str(item).strip() for item in result if str(item).strip()]
        except (ValueError, SyntaxError):
            pass

    items = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[\-\*\d\.\)\s]+', '', line)
        line = line.strip(' \'",')
        if line:
            items.append(line)
    return items


def generate_keywords(learning_session: LearningSession, max_keywords: int = 10) -> List[str]:
    chat = get_chat_model()
    chat.create_chat()

    titles = learning_session.course_titles_to_search
    titles_text = "\n".join([f'- {title}' for title in titles])

    prompt = f"""
    Extract up to {max_keywords} meaningful keywords from the following educational titles.
    - Keywords should be suitable for searching courses on **YouTube** and **Udemy**.
    - Keywords must be in the same language as the titles.
    - Exclude unnecessary words, stopwords, and duplicates.
    - Only return a Python list of keywords, without explanations or numbers.
    - Output example:
      ["Python for Beginners", "Data Science Fundamentals", "Project Management Basics", "Advanced Excel Techniques"]
    - Do not use code blocks.
    - Do not include any additional explanations or comments.

    Titles:
    {titles_text}
    """

    response = chat.send_chat_message(prompt)
    text = response.text.strip()
    keywords = _parse_list_response(text)
    return keywords

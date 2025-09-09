import os
import ast
from dotenv import load_dotenv
from typing import List
from core.learning_session import LearningSession
from utils.utilities import retry_on_network_error
from google import genai

load_dotenv()
gemini_api = os.getenv("GEMINI_API")
client = genai.Client(api_key=gemini_api)

@retry_on_network_error(max_retries=5, delay=3)
def generate_keywords(learning_session:LearningSession, max_keywords:int=10) -> List[str]:
    chat = client.chats.create(model="gemini-2.5-flash")

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

    response = chat.send_message(prompt)
    text = response.text.strip()
    keywords = ast.literal_eval(text)
    return keywords

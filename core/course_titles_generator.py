import os
import ast
import re
from dotenv import load_dotenv
from core.learning_session import LearningSession
from utils.models import get_chat_model

load_dotenv()


def _parse_list_response(text: str) -> list:
    """The LLM may not always return a strict Python list literal (["A", "B"]);
    sometimes it returns plain lines, bullet points, or inside a code block. Therefore,
    it tries in order: 1) direct literal_eval, 2) finding and literal_eval-ing the [...]
    block within the text, 3) parsing line by line (stripping numbering/bullet points/quotes).
    This prevents it from crashing with a SyntaxError due to a single format assumption."""
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


def generate_course_titles(learning_session: LearningSession) -> LearningSession:
    chat = get_chat_model()
    chat.create_chat()

    prompt = f"""
    You are an AI assistant that suggests course titles based on a user's profile information. 

    - Input: 
        Learning Goal: {learning_session.learning_goal}
        Resume: {learning_session.resume}
        Job Posting: {learning_session.job_posting}

    - Task:
        1. Analyze the user's learning_goal and context from cv_text and job_posting.
        2. Suggest a concise list of **5–10 relevant course titles** that align with the user's learning goal and career objectives.
        3. Do not include platforms, types, reasons, or explanations—only the course titles.
        4. Return the result as a **Python list of strings**.

    - Output example:
    ["Python for Beginners", "Data Science Fundamentals", "Project Management Basics", "Advanced Excel Techniques"]

    - Do not use code blocks.
    - Do not include any additional explanations or comments.
    """
    response = chat.send_chat_message(prompt)
    text = response.text.strip()
    course_titles = _parse_list_response(text)
    learning_session.course_titles_to_search = course_titles
    return learning_session
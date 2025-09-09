import os
import ast
from dotenv import load_dotenv
from core.learning_session import LearningSession
from utils.utilities import retry_on_network_error
from google import genai

load_dotenv()
gemini_api = os.getenv('GEMINI_API')

@retry_on_network_error(max_retries=5, delay=3)
def generate_course_titles(learning_session:LearningSession) -> LearningSession:
    client = genai.Client(api_key=gemini_api)
    chat = client.chats.create(model="gemini-2.5-flash")
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
    response = chat.send_message(prompt)
    text = response.text.strip()
    course_titles = ast.literal_eval(text)
    learning_session.course_titles_to_search = course_titles
    return learning_session
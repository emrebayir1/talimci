"""
Handles course recommendation logic by interacting with the Gemini API,
processing user profiles, conversation history, and generating personalized learning suggestions.
"""
import json
import os
from dotenv import load_dotenv
from core.learning_session import LearningSession
from google import genai
from utils.utilities import retry_on_network_error
from core.keywords_generator import generate_keywords
from core.courses_retreiver import retrieve_courses

load_dotenv()
gemini_api = os.getenv('GEMINI_API')
client = genai.Client(api_key=gemini_api)

def summarize_history(history, max_chars=1000):
    summary = " ".join(msg["content"] for msg in history if msg["role"] == "user")
    return summary[-max_chars:]

@retry_on_network_error(max_retries=5, delay=3)
def recommend_courses(session: LearningSession, chat_history=None, user_input=""):
    chat = client.chats.create(model="gemini-2.5-flash")

    if chat_history is None:
        chat_history = []

    system_prompt = """
    You are Talimci, an AI that specializes in personalized learning recommendations.

    #### Inputs
    You receive three inputs:
    1. **User Profile** – Structured summary with fields:
       - `learning_goal`: User's main learning objective.
       - `resume`: User's resume or 'not_provided'.
       - `job_posting`: User's job posting or 'not_provided'.
    2. **Conversation History** – Past interactions revealing user needs, motivations, and constraints.
    3. **Course Catalog Table** – Table of available courses with columns:
       - `title`: Course name.
       - `platform`: Hosting platform.
       - `channel_or_instructor`: Provider.
       - `description`: Course description.
       - `url`: Direct link.
       - `is_paid`: Free or paid.

    #### Core Task
    - Analyze the **User Profile** and **Conversation History** thoroughly.
    - On the **first relevant user message** (expressing learning needs):
      - Recommend courses from the **Course Catalog Table** only; do not invent new ones.
      - Highlight details: `title`, `platform`, `channel_or_instructor`, `url`, `is_paid`.
      - Prioritize based on needs, skill gaps, and preferences.
      - Suggest a **learning path** (recommended order) if beneficial.
      - Provide explanations for recommendations.
      - End with a **personalized motivational note** tailored to the profile.
    - Always communicate in the **same language as the user**, including headings, labels, and field names.

    #### Handling Subsequent Messages and Special Rules
    - Respond conversationally to general queries and maintain the flow, including answering questions about previously recommended courses (e.g., details, comparisons, or clarifications).
    - Only recommend courses if the user explicitly requests them. Under no circumstances should you suggest courses unprompted. If the user does not ask for a course, behave like a normal assistant.
    - For **recommending new courses:**
      - If suitable options exist in the catalog, recommend them with explanations, prioritization, and a learning path if helpful.
      - If not, output **only plain JSON** (no extra text or code blocks) with:
        - `"learning_goal"`: Updated goal.
        - `"titles"`: List of at least 3 relevant course titles (never empty; generate meaningful ones if needed).
      - Example: `{"learning_goal": "Become proficient in Data Science", "titles": ["Introduction to Python", "Statistics for Data Science", "Machine Learning Basics"]}`
    """

    chat_history.append({"role": "user", "content": user_input})

    user_turn = f"""
    User Profile:
        - Learning Goal: {session.learning_goal}
        - Resume: {session.resume}
        - Job Posting: {session.job_posting}

    Conversation History:
    {session.chat_history}

    Conversation Summary:
    {summarize_history(chat_history)}

    Course Catalog Table:
    {session.recommended_courses}

    New User Message:
    {user_input}
    """

    if len(chat_history) > 10:
        old_messages = chat_history[:-10]
        summary_text = summarize_history(old_messages)
        chat_history = [{"role": "system", "content": "Summary: " + summary_text}] + chat_history[-10:]

    response = chat.send_message(system_prompt + "\n\nUser: " + user_turn)
    message_text = response.text.strip()

    try:
        cleaned = message_text.strip()
        parsed = json.loads(cleaned)
        
        if isinstance(parsed, dict) and "titles" in parsed:
            titles = parsed.get("titles", [])
            if not titles:
                titles = [
                    f"Introduction to {session.learning_goal}",
                    f"Fundamentals of {session.learning_goal}",
                    f"Advanced {session.learning_goal} Concepts"
                ]
            session.learning_goal = parsed['learning_goal']
            session.course_titles_to_search = titles
            
            keywords = generate_keywords(session)
            session = retrieve_courses(keywords, session)
            
            updated_user_turn = f"""
    User Profile:
        - Learning Goal: {session.learning_goal}
        - Resume: {session.resume}
        - Job Posting: {session.job_posting}
    
    Conversation History:
    {session.chat_history}
    
    Conversation Summary:
    {summarize_history(session.chat_history)}
    
    Course Catalog Table:
    {session.recommended_courses}
    
    New User Message:
    {user_input}
    """
            new_response = chat.send_message(system_prompt + "\n\nUser: " + updated_user_turn)
            message_text = new_response.text.strip()
    
    except json.JSONDecodeError as e:
        chat_history.append({"role": "assistant", "content": str(e)})
    
    return message_text, session, chat_history
"""
Handles course recommendation logic by interacting with the LLM API,
processing user profiles, conversation history, and generating personalized learning suggestions.
"""
import json
import os
import requests
from core.learning_session import LearningSession
from utils.models import get_chat_model
from core.courses_retreiver import retrieve_courses

def _trim_catalog_table(markdown_table: str, max_rows: int = 8, max_chars: int = 4000) -> str:
    """Safely trims the course catalog table to a manageable size before embedding
    it into the prompt. Providers like Groq have varying TPM (tokens per minute)
    and request size limits depending on the model/tier (e.g., 12000 for
    llama-3.3-70b-versatile, 8000 for openai/gpt-oss-120b on the same account);
    therefore, instead of assuming a fixed limit, we keep the incoming data
    below a reasonable upper bound upfront. The header + separator line
    is preserved, and the body is trimmed to a maximum of `max_rows` lines."""
    if not markdown_table:
        return markdown_table

    lines = markdown_table.splitlines()
    if len(lines) > 2:
        header, sep, *body = lines
        lines = [header, sep] + body[:max_rows]

    trimmed = "\n".join(lines)
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars] + "\n…"
    return trimmed

 
def _is_too_large_error(e: requests.HTTPError) -> bool:
    """Recognizes the 413 (request/payload size limit exceeded) error."""
    response = getattr(e, "response", None)
    return response is not None and response.status_code == 413
 
 
def _build_too_large_message(user_input: str) -> str:
    """A polite message shown to the user instead of crashing when a 413 occurs."""
    return (
        "The data needed for this request (course catalog and conversation "
        "history) exceeded the model's size limit. Could you rephrase your "
        "question a bit more briefly and try again?"
    )
 
 
def summarize_history(history, max_messages=8, max_chars=3000):
    """Summarizes the conversation history with role tags in chronological order.
 
    IMPORTANT: The previous implementation only took `role == "user"` messages — meaning
    it COMPLETELY forgot the assistant's own recommended courses and given answers. Thus:
    - Follow-up requests like "one source is not enough" didn't know what was lacking
      (because it didn't remember what was recommended),
    - Questions asked about "the courses you recommended" could not be answered,
    - Overall, the flow looked inconsistent (each turn started almost from scratch,
      unaware of previous assistant messages).
 
    We include both user and assistant messages; limiting to the last `max_messages`
    messages and `max_chars` characters to prevent the prompt size (TPM limit)
    from getting out of hand."""
    if not history:
        return ""
 
    recent = history[-max_messages:] if len(history) > max_messages else history
    lines = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {msg.get('content', '')}")
 
    joined = "\n".join(lines)
    if len(joined) > max_chars:
        joined = "…\n" + joined[-max_chars:]
    return joined
 
 
def _build_no_catalog_match_message(session, titles, user_input):
    """Used instead of showing raw JSON to the user when the second model call
    also returns JSON (meaning no matches were found in the catalog yet)."""
    bullet_list = "\n".join(f"- {t}" for t in titles)
 
    return (
        f"I couldn't find a course in our catalog that closely matches your goal of "
        f"'{session.learning_goal}'. In the meantime, here are some topics worth exploring:\n\n"
        f"{bullet_list}\n\n"
        "Would you like to try rephrasing your goal with different keywords? "
        "That would help me find better matching courses."
    )
 
 
SYSTEM_PROMPT = """You are Talimci, a course recommendation AI.
 
Inputs you receive:
1. User Profile (learning_goal, resume, job_posting)
2. Conversation History (includes YOUR OWN previous replies too, e.g. courses you already recommended)
3. Course Catalog (platform, title, url, is_paid, description)
 
IMPORTANT RULES:
- Recommend ONLY from the provided Course Catalog. NEVER invent courses.
- If the user asks about, discusses, or wants more detail on courses you already recommended, answer using the Conversation History — you already listed them there, don't ask the user to repeat their goal.
- If the user explicitly says the current recommendation(s) are not enough, asks for MORE/OTHER/DIFFERENT/ADDITIONAL courses, or otherwise indicates the current Course Catalog does not satisfy them (in any language — e.g. "yetmez", "başka", "daha fazla", "not enough", "more", "other"): return ONLY this exact JSON format: {"learning_goal":"user's goal","titles":["<real course title 1>","<real course title 2>","<real course title 3>"]} to trigger a fresh, broader search. Do this even if the catalog already has some matches — "some" is not "enough" if the user says so. The "titles" array is a FORMAT EXAMPLE ONLY — "Course 1"/"Course 2"/"Course 3" are placeholders, never output those literal strings; always generate real, specific, relevant course titles based on the user's learning goal.
- If user asks for NEW recommendations and catalog has matches: list EVERY SINGLE row from the provided Course Catalog table — do not omit, sample, or summarize any of them, even if there are many (e.g. 20). For each one, include platform, title, url, price. Suggest learning order. End with motivational note.
- If user asks for NEW recommendations but catalog is empty: return ONLY this exact JSON format: {"learning_goal":"user's goal","titles":["<real course title 1>","<real course title 2>","<real course title 3>"]} — again, generate real titles, never the literal placeholders shown above.
- If user asks general questions (not about courses): answer normally, do NOT recommend courses.
- Always respond in the SAME LANGUAGE as the user's most recent message.
- NEVER use markdown code blocks (no ```json)."""
 
 
def recommend_courses(session: LearningSession, chat_history=None, user_input=""):
    chat = get_chat_model()
    chat.create_chat(system_prompt=SYSTEM_PROMPT)
 
    if chat_history is None:
        chat_history = []
 
    chat_history.append({"role": "user", "content": user_input})
 
    conversation_summary = summarize_history(chat_history)
 
    user_turn = f"""User Profile:
- Learning Goal: {session.learning_goal}
- Resume: {session.resume}
- Job Posting: {session.job_posting}
 
Conversation Summary:
{conversation_summary}
 
Course Catalog Table:
{session.recommended_courses}
 
New User Message:
{user_input}"""
 
    try:
        response = chat.send_chat_message(user_turn)
    except requests.HTTPError as e:
        if _is_too_large_error(e):
            return _build_too_large_message(user_input), session, chat_history
        raise
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
 
            session = retrieve_courses(titles, session)
 
            updated_user_turn = f"""User Profile:
- Learning Goal: {session.learning_goal}
- Resume: {session.resume}
- Job Posting: {session.job_posting}
 
Conversation Summary:
{summarize_history(chat_history)}
 
Course Catalog Table:
{session.recommended_courses}
 
New User Message:
{user_input}
 
IMPORTANT: This is already a broadened, retried search — do NOT return JSON again under any circumstances.
- If the Course Catalog above now has matches, list EVERY SINGLE row of them, not a subset — normally as instructed.
- If the Course Catalog above is still empty or has no good matches, do NOT use the JSON format. Instead, reply directly to the user in their own words and in their language: say you couldn't find a matching course in the catalog right now, and suggest 2-3 concrete subtopics or search angles (written by you, based on the user's actual learning goal) they could try. Then ask if they'd like to try different keywords."""
 
            chat.create_chat(system_prompt=SYSTEM_PROMPT)
            try:
                new_response = chat.send_chat_message(updated_user_turn)
            except requests.HTTPError as e:
                if _is_too_large_error(e):
                    return _build_too_large_message(user_input), session, chat_history
                raise
            message_text = new_response.text.strip()
 
            try:
                still_json = json.loads(message_text.strip())
                if isinstance(still_json, dict) and "titles" in still_json:
 
                    forced_prompt = f"""The user's learning goal is: {session.learning_goal}
The user is writing in this language (match it): "{user_input}"
 
Do not return JSON. Write your reply as plain natural-language text explaining
that no matching course was found in the catalog, and suggest 2-3 real
subtopics related to the learning goal above. Then ask if they'd like to try
different keywords."""
                    chat.create_chat(system_prompt=SYSTEM_PROMPT)
                    try:
                        forced_response = chat.send_chat_message(forced_prompt)
                        forced_text = forced_response.text.strip()
                        try:
                            forced_json = json.loads(forced_text)
                        except json.JSONDecodeError:
                            forced_json = None
 
                        if isinstance(forced_json, dict) and "titles" in forced_json:
                            forced_titles = forced_json.get("titles") or titles
                            message_text = _build_no_catalog_match_message(session, forced_titles, user_input)
                        else:
                            message_text = forced_text
                    except requests.HTTPError as e:
                        if _is_too_large_error(e):
                            return _build_too_large_message(user_input), session, chat_history
                        raise
            except json.JSONDecodeError:
                pass
 
    except json.JSONDecodeError as e:
        chat_history.append({"role": "assistant", "content": str(e)})
 
    return message_text, session, chat_history

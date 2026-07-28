"""
Extracts user profile info for personalized course recommendations.
"""
import json
import re
from typing import Optional

from core.learning_session import LearningSession
from utils.models import get_chat_model


SYSTEM_PROMPT = """You are Talimci. You collect the user's profile for course
recommendations by asking ONE question at a time, in this order:
1. learning_goal (required, free text)
2. resume (optional, free text) - first ask if they want to share one
   (closed-choice: share / skip); if they agree, THEN ask them to paste it
   (open-ended) as the next step
3. job_posting (optional, free text) - same optional pattern as resume
4. subscription_preference (required, closed-choice: paid / free / either)

You will be given the conversation so far (your previous questions and the
user's answers, in order). Based on it, figure out what is already known and
ask for the next missing thing. Never ask about something already answered.

Respond with EXACTLY ONE JSON object and NOTHING else (no markdown fences,
no commentary, no extra keys):

A) If there is a next question to ask:
{"question": "<question text>", "options": [<2-4 short button labels>]}
- Include "options" ONLY for genuinely closed-choice questions: subscription
  preference, or "do you want to share your resume/job posting?". For
  example, in English these would be ["Paid","Free","Either"] or
  ["I'll share it","Skip"] - but see the LANGUAGE rule below.
- For open-ended answers (learning_goal, or once the user agreed to paste
  resume/job_posting text) set "options" to an empty list [] - the user will
  type a free-text answer in a normal text box.

B) Once ALL FOUR fields are known:
{"learning_goal": "...", "resume": "... or not_provided", "job_posting": "... or not_provided", "subscription_preference": "free|paid|all"}
- The four values inside this final object must stay as-is (plain text /
  the literal English tokens "free"/"paid"/"all"/"not_provided") - do NOT
  translate these, they are internal data, not something shown to the user.

LANGUAGE RULE (important):
- Detect the language the user is writing in from their messages in the
  conversation so far (if they haven't written anything yet, default to
  English for the very first question).
- Both the "question" text AND every string inside "options" MUST be fully
  translated into that language. Never leave them in English if the user is
  writing in another language - translate the meaning naturally, don't just
  transliterate. This applies to every question, not just the first one.

Other rules:
- Never invent values the user hasn't actually given you.
- If the user declines an optional field (says "skip", "no", "none", or
  clicks a "skip"-equivalent option, in any language), its final value must
  be "not_provided".
- If the user seems confused or asks why you're asking, put a brief
  explanation INSIDE the "question" field (in the user's language), followed
  by the same question again - do not move on to the next field in that
  case.
- Do not use markdown code blocks."""


def _extract_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text.strip())
    except (ValueError, TypeError):
        pass

    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except ValueError:
            pass

    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except ValueError:
            pass

    return None


def _build_context(chat_history: list) -> str:
    if not chat_history:
        return "(The chat has not started yet - ask the first question.)"
    lines = []
    for msg in chat_history:
        role = "Assistant" if msg.get("role") == "assistant" else "User"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


_FIRST_QUESTION = (
    "What would you like to learn, or what's your overall learning goal? "
    "(e.g. \"I want to learn data analysis with Python\")"
)


def get_next_step(chat_history: list) -> dict:
    """Sends the chat history to the LLM and requests the next step.

    Returns one of:
      {"type": "question", "text": str, "options": list[str]}
      {"type": "done", "session": LearningSession}
    """
    if not chat_history:
        return {"type": "question", "text": _FIRST_QUESTION, "options": []}

    chat = get_chat_model()
    chat.create_chat(system_prompt=SYSTEM_PROMPT)

    response = chat.send_chat_message(_build_context(chat_history))
    text = response.text.strip()
    parsed = _extract_json(text)

    if not parsed:
        return {"type": "question", "text": text, "options": []}

    if "question" in parsed:
        options = parsed.get("options") or []
        if not isinstance(options, list):
            options = []
        return {
            "type": "question",
            "text": str(parsed.get("question", "")).strip(),
            "options": [str(o).strip() for o in options if str(o).strip()][:4],
        }

    required = ["learning_goal", "resume", "job_posting", "subscription_preference"]
    if all(k in parsed for k in required):
        session = LearningSession(
            learning_goal=parsed["learning_goal"],
            resume=parsed.get("resume") or "not_provided",
            job_posting=parsed.get("job_posting") or "not_provided",
            subscription_preference=(parsed.get("subscription_preference") or "all"),
            chat_history=chat_history,
        )
        return {"type": "done", "session": session}

    return {"type": "question", "text": text, "options": []}
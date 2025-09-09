"""
Extracts user profile info for personalized course recommendations.
Returns a LearningSession with:
- learning_goal
- resume
- job_posting
- subscription_preference
- chat_history
Ensures JSON-only output and language consistency.
"""
import os
import json
from utils.utilities import retry_on_network_error
from core.learning_session import LearningSession
from dotenv import load_dotenv
from google import genai

load_dotenv()
gemini_api = os.getenv('GEMINI_API')
client = genai.Client(api_key=gemini_api)

@retry_on_network_error(max_retries=5, delay=3)
def get_user_profile(user_input: str, chat_history=None):

    chat = client.chats.create(model="gemini-2.5-flash")

    if chat_history is None:
        chat_history = []

    system_prompt = """
    You are Talimci, an AI assistant that specializes in personalized learning recommendations.  
    Your job is to extract user profile information for an education recommendation system.
    - Respond in the same language the user uses.
    - First, analyze the user's input and **try to fill all fields automatically**:
      - "learning_goal": summarize the user's main learning objective in a short, clear sentence. If only a CV or job posting is provided, attempt to infer a relevant learning goal.
      - "resume": extract if provided; otherwise set internally to 'not_provided'.
      - "job_posting": extract the job posting or role description from the input; if unclear, leave empty.
      - "subscription_preference": extract if user mentioned free/paid/all; if unclear, automatically set to 'all'.
    - After attempting automatic extraction, **ask sequentially only for the missing or unclear fields**:
      - For 'learning_goal', ask **only once** for more details if the answer is too general. If the user does not provide additional details, accept the short or general answer as valid. Only ask follow-ups if the answer is empty, completely nonsensical, or offensive. Follow-ups:
        1. Reminds them politely to provide a real learning objective,
        2. Adapts to the type of input (e.g., nonsense, offensive, irrelevant),
        3. Encourages a clear learning goal,
        4. **Different** from previous ones.
      - For 'resume', 'job_posting' and 'subscription_preference', ask only once each. When asking, provide a short reason why it is useful. If user does not provide an answer, internally set to 'not_provided'. For 'subscription_preference', set to 'all'. **Never phrase this choice to the user. Do not say 'or type not_provided'.**
    - If the user asks "why", "what is the reason", or any similar clarification: 
        1. Respond briefly (e.g., "To provide you with more personalized course recommendations" or the equivalent in the user's language). 
        2. Immediately ask for the missing information again, so the flow of collecting the profile continues without skipping any field.

    - The system should try to infer a meaningful learning_goal from any partial input (CV or job posting). Only if it cannot, it will ask the user for clarification.

    - Always return **only the final JSON** after all information is collected.

    Final JSON structure:
    {
      "learning_goal": "short, clear learning objective based on user's input",
      "resume": "user's resume or 'not_provided'",
      "job_posting": "user's job posting or 'not_provided'",
      "subscription_preference": "all, free, or paid (default 'all' if unknown)"
    }

    JSON Rules:
    - Respond only in the following structure: {"learning_goal": "...", "resume": "...", "job_posting": "...", "subscription_preference": "..."}
    - Both keys and values must be enclosed in double quotes.
    - Do not use code blocks.
    - Do not include any additional explanations or comments.
    """

    chat_history.append({"role": "user", "content": user_input})

    # Chat geçmişini prompt'a dahil et
    conversation_context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )

    response = chat.send_message(system_prompt + "\n\nConversation:\n" + conversation_context)
    message_text = response.text.strip()

    if not message_text:
        return "LLM did not respond", chat_history

    try:
        data = json.loads(message_text)
        session = LearningSession(**data, chat_history=chat_history)
        return session, chat_history
    except json.JSONDecodeError:
        chat_history.append({"role": "assistant", "content": message_text})
        return message_text, chat_history
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"Hata: {e}"})
        return f"Error: {e}", chat_history

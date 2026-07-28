
"""
Chainlit-based application for Talimci - Personalized Course Recommendation Assistant.

To run:
    pip install chainlit
    chainlit run app.py -w
"""

import chainlit as cl

from core.learning_session import LearningSession
from core.course_titles_generator import generate_course_titles
from core.courses_retreiver import retrieve_courses
from core.user_orienter import get_next_step
from core.course_recommender import recommend_courses


async_get_next_step = cl.make_async(get_next_step)
async_generate_course_titles = cl.make_async(generate_course_titles)
async_retrieve_courses = cl.make_async(retrieve_courses)
async_recommend_courses = cl.make_async(recommend_courses)

MAX_PROFILE_STEPS = 20  


AUTHOR = "Talimci"

async def _ask_widget_question(question_text: str, options: list[str]):
    """Displays a widget with buttons if options are available, otherwise shows a free-text box.
    Returns the user's response as text; returns None on timeout."""
    if options:
        actions = [
            cl.Action(name="profile_option", payload={"value": opt}, label=opt)
            for opt in options
        ]
        res = await cl.AskActionMessage(
            content=question_text,
            actions=actions,
            timeout=300,
            author=AUTHOR,
        ).send()
        if not res:
            return None
        payload = res.get("payload") or {}
        return payload.get("value") or res.get("label") or res.get("value")

    res = await cl.AskUserMessage(content=question_text, timeout=300, author=AUTHOR).send()
    if not res:
        return None
    return res.get("output")


async def run_profile_wizard() -> LearningSession | None:
    """A wizard that collects profile information step-by-step using widget-based 
    questions guided by the LLM. Returns a LearningSession upon completion."""
    history: list[dict] = []

    for _ in range(MAX_PROFILE_STEPS):
        step = await async_get_next_step(history)

        if step["type"] == "done":
            return step["session"]

        question_text = step["text"]
        options = step["options"]

        answer = await _ask_widget_question(question_text, options)

        history.append({"role": "assistant", "content": question_text})
        if answer is None:
            history.append({"role": "user", "content": "(yanıt verilmedi)"})
        else:
            history.append({"role": "user", "content": answer})

    await cl.Message(
        content=(
            "An error occurred while completing your profile; "
            "please restart the chat and try again."
        ),
        author=AUTHOR,
    ).send()
    return None


async def process_user_profile_and_retrieve_courses(session: LearningSession) -> LearningSession:
    """Pipeline after profile completion: title generation -> course fetching."""

    session = await async_generate_course_titles(session)

    session = await async_retrieve_courses(
        keywords=session.course_titles_to_search, learning_session=session
    )

    return session


@cl.on_chat_start
async def start():
    cl.user_session.set("session_state", None)
    cl.user_session.set("chat_history", [])

    await cl.Message(
        content=(
            "🎓 **Talimci - Personalized Course Recommendation Assistant**\n\n"
            "Talimci looks at your learning goal - and optionally your resume "
            "and a target job posting - to recommend real courses from our "
            "catalog that actually fit where you are and where you want to go.\n\n"
            "**How this works:**\n"
            "1. I'll ask you a few quick questions: your learning goal, "
            "optionally your resume and a target job posting, and your "
            "paid/free course preference\n"
            "2. I'll build your profile and put together a personalized "
            "course catalog for you\n"
            "3. You'll get your first recommendations - after that, ask for "
            "more detail, request different courses, or refine your "
            "preferences anytime\n\n"
            "Let's get started 👇"
        ),
        author=AUTHOR,
    ).send()

    profile_session = await run_profile_wizard()
    if profile_session is None:
        return

    chat_history = [
        {"role": "user", "content": f"Öğrenme hedefim: {profile_session.learning_goal}"}
    ]

    await cl.Message(content="Finding the best courses for you...", author=AUTHOR).send()
    session_state = await process_user_profile_and_retrieve_courses(profile_session)

    first_recommendation_input = f"""
    Please recommend courses suitable for the following learning goal in English:
    {session_state.learning_goal}
    """

    bot_response, updated_session, _ = await async_recommend_courses(
        session_state, chat_history=chat_history.copy(), user_input=first_recommendation_input
    )
    cl.user_session.set("session_state", updated_session)
    chat_history.append({"role": "assistant", "content": bot_response})
    cl.user_session.set("chat_history", chat_history)
    await cl.Message(content=bot_response, author=AUTHOR).send()


@cl.on_message
async def main(message: cl.Message):
    user_input = message.content

    session_state: LearningSession | None = cl.user_session.get("session_state")
    chat_history = cl.user_session.get("chat_history") or []

    if session_state is None:
        await cl.Message(
            content="Your profile information is not completed yet, please restart the chat.",
            author=AUTHOR,
        ).send()
        return

    chat_history.append({"role": "user", "content": user_input})

    bot_response, updated_session, _ = await async_recommend_courses(
        session_state, chat_history=chat_history.copy(), user_input=user_input
    )
    cl.user_session.set("session_state", updated_session)
    chat_history.append({"role": "assistant", "content": bot_response})
    cl.user_session.set("chat_history", chat_history)

    await cl.Message(content=bot_response, author=AUTHOR).send()
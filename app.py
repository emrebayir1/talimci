"""
A Gradio-based application that collects the user's learning goals,
builds a user profile, and provides personalized course recommendations.
"""

import os
from dotenv import load_dotenv
from google import genai
import gradio as gr

from core.learning_session import LearningSession
from core.course_titles_generator import generate_course_titles
from core.keywords_generator import generate_keywords
from core.courses_retreiver import retrieve_courses
from core.user_orienter import get_user_profile
from core.course_recommender import recommend_courses

load_dotenv()
gemini_api = os.getenv('GEMINI_API')
client = genai.Client(api_key=gemini_api)

def process_user_profile_and_retrieve_courses(session: LearningSession) -> LearningSession:
    session = generate_course_titles(session)
    keywords = generate_keywords(session)
    session = retrieve_courses(keywords=keywords, learning_session=session)
    return session

def gradio_chat(user_input, chat_history, session_state, internal_profile_history):

    if chat_history is None:
        chat_history = []

    if internal_profile_history is None:
        internal_profile_history = []

    if session_state is None:
        result, updated_internal_history = get_user_profile(user_input, internal_profile_history)

        if isinstance(result, str):
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": result})
            return chat_history, session_state, updated_internal_history
        else:
            session_state = process_user_profile_and_retrieve_courses(result)
            chat_history.append({"role": "user", "content": user_input})
            first_recommendation_input = f"""
            Please recommend courses suitable for the following learning goal in the language the user wrote:
            {session_state.learning_goal}
            """
            bot_response, updated_session, _ = recommend_courses(
                session_state, chat_history=chat_history.copy(), user_input=first_recommendation_input
            )

            chat_history.append({"role": "assistant", "content": bot_response})
            return chat_history, updated_session, updated_internal_history

    else:
        chat_history.append({"role": "user", "content": user_input})
        bot_response, updated_session, _ = recommend_courses(
            session_state, chat_history=chat_history.copy(), user_input=user_input
        )
        chat_history.append({"role": "assistant", "content": bot_response})

        return chat_history, updated_session, internal_profile_history


def clear_chat():
    return [], None, []


with gr.Blocks(title="Talimci - Personalized Course Recommendation Assistant") as demo:
    gr.Markdown("# 🎓 Talimci - Personalized Course Recommendation Assistant")
    gr.Markdown("Share your learning goals, and I’ll recommend the courses that suit you best!")

    chatbot = gr.Chatbot(
        type="messages",
        height=500,
        placeholder="Hello! Let’s start by sharing your learning goals… You can also share a brief resume, the job you’re aiming for, or your preference for paid or free courses if you like."
    )

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Type your message...",
            container=False,
            scale=7
        )
        clear_btn = gr.Button("Clear Chat", scale=1)

    session_state = gr.State(value=None)
    internal_profile_history = gr.State(value=[])

    # Event handlers
    def add_user_message(user_input, chat_history):
        if chat_history is None:
            chat_history = []
        if not user_input.strip():
            return chat_history, ""
        chat_history = chat_history + [{"role": "user", "content": user_input}]
        return chat_history, ""

    def bot_reply(chat_history, session_state, internal_history):
        user_input = chat_history[-1]["content"]
        if internal_history is None:
            internal_history = []
        if session_state is None:
            result, updated_internal = get_user_profile(user_input, internal_history)
            if isinstance(result, str):
                chat_history.append({"role": "assistant", "content": result})
                return chat_history, session_state, updated_internal
            else:
                session_state = process_user_profile_and_retrieve_courses(result)
                first_recommendation_input = f"""
                Please recommend courses suitable for the following learning goal in the language the user wrote:
                {session_state.learning_goal}
                """
                bot_response, updated_session, _ = recommend_courses(
                    session_state, chat_history=chat_history.copy(), user_input=first_recommendation_input
                )
                chat_history.append({"role": "assistant", "content": bot_response})
                return chat_history, updated_session, updated_internal
        else:
            bot_response, updated_session, _ = recommend_courses(
                session_state, chat_history=chat_history.copy(), user_input=user_input
            )
            chat_history.append({"role": "assistant", "content": bot_response})
            return chat_history, updated_session, internal_history

    msg_box.submit(
        fn=add_user_message,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, msg_box],
        queue=False
    ).then(
        fn=bot_reply,
        inputs=[chatbot, session_state, internal_profile_history],
        outputs=[chatbot, session_state, internal_profile_history]
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, session_state, internal_profile_history]
    )

if __name__ == "__main__":
    demo.launch()

"""
Performs lexical (non-LLM, non-embedding) matching based on user queries,
and updates a LearningSession with relevant course recommendations.
"""
import pandas as pd
import os
import re
from difflib import SequenceMatcher
from dotenv import load_dotenv
from lingua import LanguageDetectorBuilder
from utils.utilities import run_async
from core.courses_dataframe_generator import generate_courses_dataframe
from core.learning_session import LearningSession

load_dotenv()

# Number of top-matching courses returned per search. Configurable via the
# COURSE_TOP_N environment variable (.env file) so it can be tuned without
# touching the code — e.g. lower it to reduce prompt size (and 429 risk) or
# raise it to show more courses per recommendation.
TOP_N_RESULTS = int(os.getenv("COURSE_TOP_N", "10"))


def _ensure_text_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'text' not in df.columns:
        df['text'] = (
                df.get('title', '').astype(str) + ' ' +
                df.get('description', '').astype(str)
        ).str.strip()
    return df


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> set:
    return set(_WORD_RE.findall((text or "").lower()))


def lexical_similarity(query: str, text: str) -> float:
    """API-free simple similarity score: weighted average of word 
    overlap (Jaccard) and string similarity (SequenceMatcher). Jaccard 
    captures common words in the title/description, while SequenceMatcher 
    tolerates partial/root matches (e.g., 'python' vs 'pythonic')."""
    if not query or not text:
        return 0.0

    q_tokens = _tokenize(query)
    t_tokens = _tokenize(text)
    if not q_tokens or not t_tokens:
        return 0.0

    intersection = len(q_tokens & t_tokens)
    union = len(q_tokens | t_tokens)
    jaccard = intersection / union if union else 0.0

    seq_ratio = SequenceMatcher(None, query.lower(), text.lower()).ratio()

    return 0.7 * jaccard + 0.3 * seq_ratio


async def semantic_search_async(query, df, lang="en", top_n=TOP_N_RESULTS, paid_status="all"):

    if df.empty:
        print("Warning: DataFrame is empty, returning empty result")
        return pd.DataFrame(columns=['platform', 'title', 'url', 'is_paid', 'description'])

    df = _ensure_text_column(df)

    has_similarity = True
    try:
        df["similarity"] = df["text"].apply(lambda t: lexical_similarity(query, t))
    except Exception as e:
        print(f"Lexical similarity error: {e}")
        has_similarity = False

    if 'language' in df.columns:
        lang_str = df["language"].astype(str)
        is_known_language = lang_str.str.strip() != ""
        matches_target_lang = lang_str.str.startswith(lang)
        filtered_df = df[matches_target_lang | ~is_known_language].copy()
        if filtered_df.empty and lang != "en":
            filtered_df = df[df["language"].str.startswith("en") | ~is_known_language].copy()
    else:
        filtered_df = df.copy()

    if paid_status not in ["all", "free", "paid"]:
        paid_status = "all"

    if 'is_paid' in filtered_df.columns:
        if paid_status == "free":
            filtered_df = filtered_df[filtered_df["is_paid"] == False]
        elif paid_status == "paid":
            filtered_df = filtered_df[filtered_df["is_paid"] == True]

    max_len = 200
    if "description" in filtered_df.columns:
        filtered_df["description"] = filtered_df["description"].astype(str).apply(
            lambda x: x[:max_len] + "…" if len(x) > max_len else x
        )

    required_columns = ['platform', 'title', 'url', 'is_paid', 'description']
    for col in required_columns:
        if col not in filtered_df.columns:
            filtered_df[col] = 'N/A'

    if has_similarity:
        filtered_df = filtered_df.sort_values("similarity", ascending=False)

    return filtered_df.head(top_n)[required_columns].reset_index(drop=True)


def get_query_language(text: str) -> str:
    try:
        detector = LanguageDetectorBuilder.from_all_languages().build()
        language = detector.detect_language_of(text)
        return language.iso_code_639_1.name.lower()
    except:
        return "en"


async def retriever(query: str, df: pd.DataFrame, paid_status: str = 'all') -> pd.DataFrame:
    try:
        query_language = get_query_language(query)
        courses_df = _ensure_text_column(df)
        retrieved_courses = await semantic_search_async(query, courses_df, query_language, paid_status=paid_status)
        return retrieved_courses
    except Exception as e:
        print(f"Retriever error: {e}")
        return pd.DataFrame(columns=['platform', 'title', 'url', 'is_paid', 'description'])


def retrieve_courses(keywords: list, learning_session: LearningSession) -> LearningSession:
    try:
        learning_goal = learning_session.learning_goal
        paid_status = learning_session.subscription_preference
        already_shown = set(learning_session.shown_course_titles or [])

        search_terms = list(dict.fromkeys(
            [learning_goal] + list(keywords or [])
        ))

        print(f"Generating courses dataframe for search terms: {search_terms}")
        courses_dataframe = run_async(generate_courses_dataframe(search_terms))

        print(f"Courses dataframe generated. Shape: {courses_dataframe.shape}")
        print(f"Columns: {courses_dataframe.columns.tolist()}")

        if already_shown and not courses_dataframe.empty and 'title' in courses_dataframe.columns:
            before = len(courses_dataframe)
            courses_dataframe = courses_dataframe[
                ~courses_dataframe['title'].isin(already_shown)
            ].reset_index(drop=True)
            print(f"Daha önce gösterilmiş {before - len(courses_dataframe)} kurs aday havuzundan çıkarıldı.")

        results_df = run_async(
            retriever(query=learning_goal, df=courses_dataframe, paid_status=paid_status)
        )

        if not results_df.empty:
            learning_session.recommended_courses = results_df.to_markdown(index=False)
            if 'title' in results_df.columns:
                new_titles = results_df['title'].dropna().tolist()
                learning_session.shown_course_titles = list(dict.fromkeys(
                    (learning_session.shown_course_titles or []) + new_titles
                ))
        elif already_shown:
            learning_session.recommended_courses = (
                "I have already recommended all the matching courses available in our catalog for this topic; "
                "there are no new matches to show right now."
            )
        else:
            learning_session.recommended_courses = "No saved courses found."

    except Exception as e:
        print(f"retrieve_courses error: {e}")
        learning_session.recommended_courses = f"An error occurred while fetching courses: {str(e)}"

    return learning_session
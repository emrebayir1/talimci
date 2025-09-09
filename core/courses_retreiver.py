"""
Generates embeddings for course data, performs semantic search based on user queries,
and updates a LearningSession with relevant course recommendations.
"""
import pandas as pd
import numpy as np
import os
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai
from lingua import LanguageDetectorBuilder
from utils.utilities import run_async
from core.courses_dataframe_generator import generate_courses_dataframe
from core.learning_session import LearningSession

load_dotenv()
gemini_api = os.getenv('GEMINI_API')
genai.configure(api_key=gemini_api)


async def get_embedding_async(text: str):
    result = await genai.embed_content_async(
        model="models/embedding-001",
        content=text
    )
    return result['embedding']


async def generate_embedded_dataframe_async(df: pd.DataFrame) -> pd.DataFrame:
    if 'text' not in df.columns:
        df['text'] = (
                df.get('title', '').astype(str) + ' ' +
                df.get('description', '').astype(str)
        ).str.strip()

    if df.empty:
        df['text_embedding'] = []
        return df

    try:
        tasks = [get_embedding_async(t) for t in df['text']]
        df['text_embedding'] = await asyncio.gather(*tasks)
    except Exception as e:
        print(f"Embedding generation error: {e}")
        df['text_embedding'] = [[] for _ in range(len(df))]

    return df


def cosine_similarity(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.0
    try:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except:
        return 0.0


async def semantic_search_async(query, df, lang="en", top_n=20, paid_status="all"):

    if df.empty:
        print("Warning: DataFrame is empty, returning empty result")
        return pd.DataFrame(columns=['title', 'platform', 'channel_or_instructor', 'description', 'url', 'is_paid'])

    try:
        query_emb = await get_embedding_async(query)
    except Exception as e:
        print(f"Query embedding error: {e}")
        return df.head(top_n)[
            ['title', 'platform', 'channel_or_instructor', 'description', 'url', 'is_paid']].reset_index(drop=True)

    df["similarity"] = df["text_embedding"].apply(lambda x: cosine_similarity(x, query_emb))

    if 'language' in df.columns:
        filtered_df = df[df["language"].str.startswith(lang)].copy()
        if filtered_df.empty and lang != "en":
            filtered_df = df[df["language"].str.startswith("en")].copy()
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

    required_columns = ['title', 'platform', 'channel_or_instructor', 'description', 'url', 'is_paid']
    for col in required_columns:
        if col not in filtered_df.columns:
            filtered_df[col] = 'N/A'

    return filtered_df.sort_values("similarity", ascending=False).head(top_n)[
        required_columns
    ].reset_index(drop=True)


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
        courses_df = await generate_embedded_dataframe_async(df)
        retrieved_courses = await semantic_search_async(query, courses_df, query_language, paid_status=paid_status)
        return retrieved_courses
    except Exception as e:
        print(f"Retriever error: {e}")
        return pd.DataFrame(columns=['title', 'platform', 'channel_or_instructor', 'description', 'url', 'is_paid'])


def retrieve_courses(keywords: list, learning_session: LearningSession) -> LearningSession:
    try:
        learning_goal = learning_session.learning_goal
        paid_status = learning_session.subscription_preference

        print(f"Generating courses dataframe for keywords: {keywords}")
        courses_dataframe = run_async(generate_courses_dataframe(keywords))

        print(f"Courses dataframe generated. Shape: {courses_dataframe.shape}")
        print(f"Columns: {courses_dataframe.columns.tolist()}")

        results_df = run_async(
            retriever(query=learning_goal, df=courses_dataframe, paid_status=paid_status)
        )

        if not results_df.empty:
            learning_session.recommended_courses = results_df.to_markdown(index=False)
        else:
            learning_session.recommended_courses = "Kayıtlı kurs bulunamadı."

    except Exception as e:
        print(f"retrieve_courses error: {e}")
        learning_session.recommended_courses = f"Kurs getirme sırasında hata oluştu: {str(e)}"

    return learning_session

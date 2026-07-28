"""
Defines the LearningSession data model, which captures a user's learning goal, profile details, chat history,
course search preferences, and recommended courses for personalized learning experiences.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal, List

class LearningSession(BaseModel):
    learning_goal: str = Field( ...,description="The user's main learning objective. For example, 'Become a Data Scientist' or 'Improve Python skills'.",)
    resume: Optional[str] = Field(None, description="The user's resume content, which helps tailor course recommendations. Can be None if not provided.", )
    job_posting: Optional[str] = Field(None,description="The job posting or role description provided by the user. Helps to guide relevant skill/course suggestions. Can be None if not provided.",)
    subscription_preference: Literal['all', 'free', 'paid'] = Field('all',description="User's preference for course types: 'all', 'free', or 'paid'. Default is 'all'.",)
    chat_history:List[dict] = Field(None, description="A record of previous interactions between the user and the system. Each entry is a dictionary containing details such as the message content, sender (user or system), and timestamp. Useful for maintaining context and providing personalized responses.")
    course_titles_to_search:List[str] = Field(None, description="A list of course titles that the user wants to search for or explore. Each item represents the name of a course relevant to the user's learning goals or interests.")
    recommended_courses:str = Field(None, description="A list of courses recommended for the user based on their learning goals, resume, job posting, and preferences.")
    shown_course_titles: List[str] = Field(default_factory=list, description="Bu oturumda kullanıcıya daha önce gösterilmiş/önerilmiş kurs başlıkları. Yeni bir arama yapıldığında bu kurslar aday havuzundan dışlanır ki 'başka kurs öner' dendiğinde gerçekten farklı sonuçlar bulunsun, aynı popüler videolar tekrar tekrar önerilmesin.")
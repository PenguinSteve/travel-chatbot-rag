from pydantic import BaseModel
from typing import Optional

class AskResponse(BaseModel):
    message: str
    answer: str
    
class ChatResponse(BaseModel):
    chat_history: str
    reformulated_question: list[str] = None
    final_answer: str
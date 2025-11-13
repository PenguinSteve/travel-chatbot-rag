from pydantic import BaseModel
from typing import Optional, Any

class AskResponse(BaseModel):
    message: str
    answer: Any
    
class ChatResponse(BaseModel):
    chat_history: str
    reformulated_question: list[str] = None
    final_answer: str
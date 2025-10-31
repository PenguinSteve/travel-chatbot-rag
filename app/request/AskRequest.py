from pydantic import BaseModel
from typing import Optional

class AskRequest(BaseModel):
    query: str
    
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
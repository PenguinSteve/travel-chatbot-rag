from datetime import datetime
from typing import Any, List, Literal
from pydantic import BaseModel, Field

# Một tin nhắn trong cuộc hội thoại
class ChatMessage(BaseModel):
    role: Literal["human", "ai"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    trip_id: Any = None

# Một phiên hội thoại hoàn chỉnh
class ChatSession(BaseModel):
    session_id: str
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

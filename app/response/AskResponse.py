from pydantic import BaseModel
from typing import Optional

class AskResponse(BaseModel):
    query: str
    answer: str
from pydantic import BaseModel
from typing import Optional

class AskRequest(BaseModel):
    query: str
from pydantic import BaseModel
from datetime import datetime

class Document(BaseModel):
    id: int = None
    page_content: str
    size: int = None
    created_at: datetime = None
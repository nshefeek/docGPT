from pydantic import BaseModel

class DocumentSearch(BaseModel):
    query: str
    k: int = 4
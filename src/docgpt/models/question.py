from pydantic import BaseModel

class Question(BaseModel):
    query: str
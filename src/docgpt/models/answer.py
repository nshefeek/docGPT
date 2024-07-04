from typing import Dict, List, Any

from pydantic import BaseModel


class Answer(BaseModel):
    result: str
    sources: List[Dict[str, Any]]

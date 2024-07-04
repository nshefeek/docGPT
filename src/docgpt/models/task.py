from enum import Enum
from pydantic import BaseModel
from datetime import datetime


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
    id: str
    status: TaskStatus
    progress: float = 0.0
    details: str = ""
    created_at: datetime
    updated_at: datetime

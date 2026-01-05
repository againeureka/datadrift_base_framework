from __future__ import annotations
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Any

class DatasetOut(BaseModel):
    id: int
    name: str
    original_filename: str
    created_at: datetime
    download_url: Optional[str] = None

class JobOut(BaseModel):
    task_id: str
    state: str
    result: Optional[Any] = None

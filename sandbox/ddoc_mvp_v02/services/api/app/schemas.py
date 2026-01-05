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


class OperatorOut(BaseModel):
    name: str
    version: str
    input_count: int
    input_types: list[str]
    description: str
    params_schema: dict[str, Any]

class RunIn(BaseModel):
    operator_name: str
    dataset_ids: list[int]
    params: dict[str, Any] = {}

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Literal, Optional

ArtifactType = Literal[
    "file",
    "tabular_df",
]

class Manifest(BaseModel):
    dataset_name: str
    files: list[str]
    sha256: str
    size_bytes: int
    detected: str

class Artifact(BaseModel):
    type: ArtifactType
    uri: Optional[str] = None  # could be s3://... later
    meta: dict[str, Any] = Field(default_factory=dict)

class Report(BaseModel):
    kind: str = "eda"
    summary: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)

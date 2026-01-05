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


from typing import Callable

class OperatorSpec(BaseModel):
    name: str
    version: str = "0.1.0"
    # input artifact types; for MVP we infer from files, but keep contract
    input_types: list[ArtifactType]
    # number of required inputs (1 for EDA, 2 for diff)
    input_count: int = 1
    description: str = ""
    # optional params schema (jsonschema-ish)
    params_schema: dict[str, Any] = Field(default_factory=dict)

class RunRequest(BaseModel):
    operator_name: str
    input_paths: list[str]
    params: dict[str, Any] = Field(default_factory=dict)

class RunResult(BaseModel):
    operator_name: str
    operator_version: str
    report: Report

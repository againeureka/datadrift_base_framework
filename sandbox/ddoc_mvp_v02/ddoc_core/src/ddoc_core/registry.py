from __future__ import annotations

from typing import Callable, Any
from .types import OperatorSpec, Report
from .operators.eda import eda as eda_file_or_csv
from .operators.diff import diff_files, diff_tabular_csv

# Simple in-process registry (v0.2). pluggy can replace this later.
_OPERATOR_HANDLERS: dict[str, Callable[..., Report]] = {}
_OPERATOR_SPECS: dict[str, OperatorSpec] = {}

def register(spec: OperatorSpec, handler: Callable[..., Report]) -> None:
    _OPERATOR_SPECS[spec.name] = spec
    _OPERATOR_HANDLERS[spec.name] = handler

def get_operator_spec(name: str) -> OperatorSpec:
    if name not in _OPERATOR_SPECS:
        raise KeyError(f"Unknown operator: {name}")
    return _OPERATOR_SPECS[name]

def get_operator_handler(name: str) -> Callable[..., Report]:
    if name not in _OPERATOR_HANDLERS:
        raise KeyError(f"Unknown operator: {name}")
    return _OPERATOR_HANDLERS[name]

def list_operators() -> list[OperatorSpec]:
    return list(_OPERATOR_SPECS.values())

def register_builtin_operators() -> None:
    # idempotent
    if _OPERATOR_SPECS:
        return

    register(
        OperatorSpec(
            name="eda",
            version="0.2.0",
            input_types=["file", "tabular_df"],
            input_count=1,
            description="Run simple EDA. CSV -> tabular EDA; otherwise file metadata.",
            params_schema={"type":"object","properties":{"max_top_values":{"type":"integer","default":10}}},
        ),
        lambda path, params=None: eda_file_or_csv(path, max_top_values=int((params or {}).get("max_top_values", 10))),
    )

    register(
        OperatorSpec(
            name="diff.file",
            version="0.2.0",
            input_types=["file"],
            input_count=2,
            description="Compare two files (hash/size/format).",
            params_schema={"type":"object","properties":{}},
        ),
        lambda path_a, path_b, params=None: diff_files(path_a, path_b),
    )

    register(
        OperatorSpec(
            name="diff.tabular",
            version="0.2.0",
            input_types=["tabular_df"],
            input_count=2,
            description="Compare two CSV tables (schema + basic distribution).",
            params_schema={"type":"object","properties":{"max_top_values":{"type":"integer","default":10}}},
        ),
        lambda path_a, path_b, params=None: diff_tabular_csv(
            path_a, path_b, max_top_values=int((params or {}).get("max_top_values", 10))
        ),
    )

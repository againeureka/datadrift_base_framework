from __future__ import annotations
from typing import Any
from .registry import register_builtin_operators, get_operator_handler, get_operator_spec
from .types import RunResult

def run_operator(operator_name: str, input_paths: list[str], params: dict[str, Any] | None = None) -> RunResult:
    register_builtin_operators()
    spec = get_operator_spec(operator_name)
    handler = get_operator_handler(operator_name)

    if len(input_paths) != spec.input_count:
        raise ValueError(f"Operator '{operator_name}' expects {spec.input_count} inputs, got {len(input_paths)}")

    params = params or {}

    if spec.input_count == 1:
        report = handler(input_paths[0], params=params)
    elif spec.input_count == 2:
        report = handler(input_paths[0], input_paths[1], params=params)
    else:
        raise ValueError("MVP supports only 1 or 2 input operators")

    return RunResult(operator_name=spec.name, operator_version=spec.version, report=report)

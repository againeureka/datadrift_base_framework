from __future__ import annotations
from typing import Any
import pandas as pd
from ..types import Report
from ..adapters.detect import detect_format
from ..utils import sha256_file, file_size

def diff_files(path_a: str, path_b: str) -> Report:
    fmt_a = detect_format(path_a)
    fmt_b = detect_format(path_b)
    summary = {
        "kind": "file_diff",
        "a": {"format": fmt_a, "sha256": sha256_file(path_a), "size_bytes": file_size(path_a)},
        "b": {"format": fmt_b, "sha256": sha256_file(path_b), "size_bytes": file_size(path_b)},
        "common": {
            "same_format": fmt_a == fmt_b,
            "same_sha256": sha256_file(path_a) == sha256_file(path_b),
        },
        "differences": {
            "size_bytes_delta": file_size(path_a) - file_size(path_b),
        },
    }
    return Report(kind="diff", summary=summary, artifacts=[])

def _top_values(df: pd.DataFrame, max_top_values: int) -> dict[str, dict[str, int]]:
    top: dict[str, dict[str, int]] = {}
    for col in df.columns[: min(10, df.shape[1])]:
        vc = df[col].value_counts(dropna=False).head(max_top_values)
        top[str(col)] = {str(k): int(v) for k, v in vc.items()}
    return top

def diff_tabular_csv(path_a: str, path_b: str, max_top_values: int = 10) -> Report:
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)

    cols_a = [str(c) for c in a.columns]
    cols_b = [str(c) for c in b.columns]
    set_a, set_b = set(cols_a), set(cols_b)

    common_cols = sorted(list(set_a & set_b))
    only_a = sorted(list(set_a - set_b))
    only_b = sorted(list(set_b - set_a))

    summary: dict[str, Any] = {
        "kind": "tabular_diff",
        "a": {"rows": int(a.shape[0]), "cols": int(a.shape[1])},
        "b": {"rows": int(b.shape[0]), "cols": int(b.shape[1])},
        "schema": {
            "common_columns": common_cols,
            "only_in_a": only_a,
            "only_in_b": only_b,
        },
        "missing": {
            "a_missing_total": int(a.isna().sum().sum()),
            "b_missing_total": int(b.isna().sum().sum()),
        },
        "top_values": {
            "a": _top_values(a, max_top_values),
            "b": _top_values(b, max_top_values),
        },
        "notes": [
            "MVP diff compares schema + simple missingness + top values. Extend with distributions/drift tests later."
        ],
    }
    return Report(kind="diff", summary=summary, artifacts=[])

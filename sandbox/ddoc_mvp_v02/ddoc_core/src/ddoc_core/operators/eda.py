from __future__ import annotations
from typing import Any
import pandas as pd
from ..types import Report, Artifact
from ..adapters.detect import detect_format
from ..utils import sha256_file, file_size

def eda(path: str, max_top_values: int = 10) -> Report:
    fmt = detect_format(path)
    if fmt == "csv":
        df = pd.read_csv(path)
        summary: dict[str, Any] = {
            "format": "csv",
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": list(df.columns.astype(str)),
            "missing_by_column": df.isna().sum().to_dict(),
        }
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] > 0:
            summary["numeric_describe"] = numeric.describe().to_dict()
        # simple top values for first few columns
        top_values: dict[str, Any] = {}
        for col in df.columns[: min(10, df.shape[1])]:
            vc = df[col].value_counts(dropna=False).head(max_top_values)
            top_values[str(col)] = {str(k): int(v) for k, v in vc.items()}
        summary["top_values"] = top_values

        artifacts = [Artifact(type="tabular_df", meta={"columns": summary["columns"]})]
        return Report(kind="eda", summary=summary, artifacts=artifacts)

    # fallback: file-level
    summary = {
        "format": fmt,
        "sha256": sha256_file(path),
        "size_bytes": file_size(path),
        "note": "No structured EDA available for this format in MVP. Add an adapter/operator to extend.",
    }
    return Report(kind="eda", summary=summary, artifacts=[Artifact(type="file", meta={"format": fmt})])

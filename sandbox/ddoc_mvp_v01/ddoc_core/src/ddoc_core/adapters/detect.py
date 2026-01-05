from __future__ import annotations
import os
from typing import Literal

def detect_format(path: str) -> Literal["csv", "zip", "unknown"]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return "csv"
    if ext == ".zip":
        return "zip"
    return "unknown"

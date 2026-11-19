"""Built-in data profile for CSV — what one dataset looks like.

``ddoc analyze eda`` falls back to this when no plugin handles the data,
the way ``ddoc diff`` falls back to its CSV comparator. It describes a
single dataset: size, period, each column's type and distribution, and
quality findings read against short written-up patterns (many missing
values, a constant column, an identifier-like column, far outliers,
duplicated rows, spelling variants, numbers stored as text, too few
rows). Like the drift archetypes, findings are codes and numbers;
:func:`finding_text` renders them through :data:`CATALOG`.

Nothing here compares two datasets; that is ``ddoc diff``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

HIGH_MISSING = 0.20       # ≥ 20 % of a column empty
NEAR_CONSTANT = 0.95      # one value holds ≥ 95 % of the rows
ID_LIKE = 0.95            # ≥ 95 % of the non-empty values are distinct ...
ID_LIKE_MIN_ROWS = 50     # ... in a column with at least this many values
FAR_OUTLIERS = 0.01       # ≥ 1 % of values beyond 3 IQR from the quartiles
DUPLICATE_ROWS = 0.01     # ≥ 1 % of rows repeat an earlier row exactly
NUMERIC_TEXT = 0.9        # ≥ 90 % of a text column parses as numbers
SMALL_SAMPLE = 30
_TOP = 8
_BINS = 20


def _norm(x: Any) -> str:
    return re.sub(r"[\s_\-]+", "", str(x)).casefold()


def _numeric_column(x, bins: int) -> Dict[str, Any]:
    import numpy as np

    q1, med, q3 = (float(v) for v in x.quantile([0.25, 0.5, 0.75]))
    iqr = q3 - q1
    far = ((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).mean() if iqr > 0 else 0.0
    out: Dict[str, Any] = {
        "mean": round(float(x.mean()), 6), "std": round(float(x.std(ddof=0)), 6),
        "min": round(float(x.min()), 6), "q1": round(q1, 6), "median": round(med, 6),
        "q3": round(q3, 6), "max": round(float(x.max()), 6),
        "distinct": int(x.nunique()), "far_outliers": round(float(far), 4),
    }
    lo, hi = float(x.min()), float(x.max())
    whole = bool((x % 1 == 0).all())
    if whole and hi - lo + 1 <= bins:
        counts = x.value_counts()
        out["discrete"] = True
        out["distribution"] = {str(v): int(counts.get(v, 0)) for v in range(int(lo), int(hi) + 1)
                               if counts.get(v, 0)}
    elif hi > lo:
        # Far outliers would stretch the bins until the bulk sits in one
        # of them; bin the bulk (quartiles ± 3 IQR) and count the rest apart.
        if iqr > 0 and far > 0:
            lo, hi = max(lo, q1 - 3 * iqr), min(hi, q3 + 3 * iqr)
            out["outside_bins"] = round(float(((x < lo) | (x > hi)).mean()), 4)
        edges = np.linspace(lo, hi, bins + 1)
        hist, _ = np.histogram(x, bins=edges)
        out["distribution"] = {f"{edges[i]:.4g}–{edges[i + 1]:.4g}": int(hist[i])
                               for i in range(bins) if hist[i]}
    else:
        out["distribution"] = {f"{lo:.4g}": int(len(x))}
    return out


def _categorical_column(x) -> Dict[str, Any]:
    import pandas as pd

    vc = x.astype(str).value_counts()
    top = list(vc.index[:_TOP])
    dist = {k: int(vc[k]) for k in top}
    other = int(vc.sum() - sum(dist.values()))
    if other:
        dist["(other)"] = other
    groups: Dict[str, List[str]] = {}
    for value in vc.index:
        groups.setdefault(_norm(value), []).append(str(value))
    variants = [g for g in groups.values() if len(g) > 1][:5]
    numeric = pd.to_numeric(x.astype(str).str.replace(",", "", regex=False), errors="coerce")
    return {"distinct": int(vc.size), "top_share": round(float(vc.iloc[0] / vc.sum()), 4),
            "top_value": str(vc.index[0]), "distribution": dist, "variants": variants,
            "numeric_share": round(float(numeric.notna().mean()), 4)}


def profile_csv(files: List[str], *, sample_rows: int = 0, bins: int = _BINS) -> Dict[str, Any]:
    """Profile one CSV dataset (one or more files, concatenated)."""
    from .cli.commands.diff import _as_datetime, _load_csv_frame, _profile, _rate

    frame = _load_csv_frame(files)
    time_cols = {c: p for c in frame.columns if (p := _as_datetime(frame[c])) is not None}
    profile = _profile(frame, files, time_cols, sample_rows)
    n = int(len(frame))
    columns: Dict[str, Dict[str, Any]] = {}
    for col, info in zip(frame.columns, profile["columns"]):
        x = frame[col].dropna()
        c: Dict[str, Any] = {"type": info["type"], "missing": _rate(n - len(x), n), "count": int(len(x))}
        if info["type"] == "numeric" and not x.empty:
            c.update(_numeric_column(x, bins))
        elif info["type"] == "categorical" and not x.empty:
            c.update(_categorical_column(x))
        elif info["type"] == "datetime":
            from .cli.commands.diff import _period
            c["period"] = _period(time_cols[col])
        columns[str(col)] = c
    duplicates = _rate(int(frame.duplicated().sum()), n) if n else 0.0
    envelope: Dict[str, Any] = {
        "status": "success", "modality": "tabular", "engine": "builtin", "profiler": "builtin:csv_profile",
        "data_path": files[0] if len(files) == 1 else f"{len(files)} files",
        "profile": profile, "columns": columns, "duplicate_rows": duplicates,
        "missing_cells": round(float(frame.isna().to_numpy().mean()), 4) if n and len(frame.columns) else 0.0,
    }
    envelope["findings"] = findings(envelope)
    return envelope


# ── Findings ──────────────────────────────────────────────────────────


def findings(envelope: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Quality findings: dataset-level first, then per column."""
    out: List[Dict[str, Any]] = []
    rows = (envelope.get("profile") or {}).get("rows") or 0
    if rows < SMALL_SAMPLE:
        out.append({"code": "small_sample", "signal": {"rows": rows}})
    if envelope.get("duplicate_rows", 0) >= DUPLICATE_ROWS:
        out.append({"code": "duplicate_rows", "signal": {"share": envelope["duplicate_rows"]}})
    per: Dict[str, List[Dict[str, Any]]] = {}
    for name, c in (envelope.get("columns") or {}).items():
        def add(code: str, **signal: Any) -> None:
            per.setdefault(code, []).append({"attr": name, **signal})
        if c["type"] == "empty":
            add("empty_column")
            continue
        if c.get("missing", 0) >= HIGH_MISSING:
            add("high_missing", share=c["missing"])
        if c["type"] == "numeric":
            if c.get("distinct") == 1:
                add("constant", value=c.get("min"))
            elif c.get("far_outliers", 0) >= FAR_OUTLIERS:
                add("far_outliers", share=c["far_outliers"], q1=c["q1"], q3=c["q3"])
            if (c.get("count", 0) >= ID_LIKE_MIN_ROWS and not c.get("discrete")
                    and c.get("distinct", 0) / max(c["count"], 1) >= ID_LIKE
                    and c.get("min", 0) % 1 == 0 and c.get("max", 0) % 1 == 0
                    and c.get("max", 0) - c.get("min", 0) + 1 <= c["count"] * 1.2):
                add("id_like", distinct=c["distinct"], count=c["count"])
        elif c["type"] == "categorical":
            if c.get("distinct") == 1:
                add("constant", value=c.get("top_value"))
            elif c.get("top_share", 0) >= NEAR_CONSTANT:
                add("near_constant", value=c["top_value"], share=c["top_share"])
            if c.get("count", 0) >= ID_LIKE_MIN_ROWS and c.get("distinct", 0) / max(c["count"], 1) >= ID_LIKE:
                add("id_like", distinct=c["distinct"], count=c["count"])
            if c.get("variants"):
                add("spelling_variants", groups=c["variants"])
            if c.get("numeric_share", 0) >= NUMERIC_TEXT:
                add("numbers_as_text", share=c["numeric_share"])
    for code in COLUMN_CODES:
        if per.get(code):
            out.append({"code": code, "attrs": [s["attr"] for s in per[code]], "signals": per[code]})
    return out


DATA_CODES = ("small_sample", "duplicate_rows")
COLUMN_CODES = ("empty_column", "constant", "near_constant", "high_missing", "id_like",
                "far_outliers", "spelling_variants", "numbers_as_text")

CATALOG: Dict[str, Dict[str, Dict[str, Any]]] = {
    "small_sample": {
        "en": {"title": "Too few rows", "meaning": "With this few rows, any distribution or comparison is fragile.",
               "actions": ["Collect more data before drawing conclusions"], "signal": "{rows} rows"},
        "ko": {"title": "행 수 부족", "meaning": "행이 적어 분포나 비교 결과가 쉽게 흔들립니다.",
               "actions": ["결론을 내기 전에 데이터를 더 모으기"], "signal": "{rows}행"},
    },
    "duplicate_rows": {
        "en": {"title": "Duplicated rows", "meaning": "Some rows repeat an earlier row exactly. Often a double load or a join that multiplied rows.",
               "actions": ["Check the load and join steps for double counting", "Decide whether duplicates are real events or errors"],
               "signal": "{share} of rows"},
        "ko": {"title": "중복 행", "meaning": "앞의 행과 완전히 같은 행이 있습니다. 중복 적재나 행이 불어난 조인에서 흔합니다.",
               "actions": ["적재·조인 단계에서 중복 집계가 없는지 확인", "중복이 실제 사건인지 오류인지 결정"],
               "signal": "전체 행의 {share}"},
    },
    "empty_column": {
        "en": {"title": "Empty columns", "meaning": "The column has no values at all.",
               "actions": ["Check whether the field is still collected", "Drop it from model inputs if it stays empty"],
               "signal": "{attr}"},
        "ko": {"title": "빈 열", "meaning": "값이 하나도 없는 열입니다.",
               "actions": ["해당 필드가 아직 수집되는지 확인", "계속 비어 있다면 모델 입력에서 제외"],
               "signal": "{attr}"},
    },
    "constant": {
        "en": {"title": "Constant columns", "meaning": "Every value is the same, so the column carries no information.",
               "actions": ["Check whether a default value replaced the real one", "Exclude it from drift checks and model inputs"],
               "signal": "{attr} = {value}"},
        "ko": {"title": "상수 열", "meaning": "모든 값이 같아 정보가 없는 열입니다.",
               "actions": ["기본값이 실제 값을 대신하고 있지 않은지 확인", "드리프트 점검과 모델 입력에서 제외 검토"],
               "signal": "{attr} = {value}"},
    },
    "near_constant": {
        "en": {"title": "Nearly constant columns", "meaning": "One value dominates; the rest is a thin tail.",
               "actions": ["Check whether the rare values matter (they are what a model can learn from)"],
               "signal": "{attr}: {value} holds {share}"},
        "ko": {"title": "거의 상수인 열", "meaning": "한 값이 대부분을 차지하고 나머지는 아주 적습니다.",
               "actions": ["드문 값이 중요한지 확인 (모델이 배울 수 있는 부분은 그쪽)"],
               "signal": "{attr}: {value} 가 {share}"},
    },
    "high_missing": {
        "en": {"title": "Many missing values", "meaning": "A large part of the column is empty.",
               "actions": ["Find whether the gaps cluster in a period or a source", "Decide how gaps are filled before modelling"],
               "signal": "{attr}: {share} empty"},
        "ko": {"title": "결측 많음", "meaning": "열의 상당 부분이 비어 있습니다.",
               "actions": ["결측이 특정 기간·출처에 몰렸는지 확인", "모델링 전에 결측 처리 방식 결정"],
               "signal": "{attr}: {share} 비어 있음"},
    },
    "id_like": {
        "en": {"title": "Identifier-like columns", "meaning": "Nearly every value is unique — an ID, a key or a free-text field. Its distribution says little and it drifts on every new batch.",
               "actions": ["Exclude it from drift comparison and model inputs", "Keep it for joins and tracing"],
               "signal": "{attr}: {distinct} distinct of {count}"},
        "ko": {"title": "식별자형 열", "meaning": "거의 모든 값이 다릅니다. ID·키·자유 입력 필드로 보이며, 분포가 의미가 적고 새 데이터마다 드리프트로 잡힙니다.",
               "actions": ["드리프트 비교와 모델 입력에서 제외", "조인·추적용으로만 유지"],
               "signal": "{attr}: {count}개 중 {distinct}개가 서로 다름"},
    },
    "far_outliers": {
        "en": {"title": "Far outliers", "meaning": "Some values lie far beyond the bulk of the data (more than 3 interquartile ranges out).",
               "actions": ["Pull those rows and check the original records", "Decide on a valid range or a capping rule"],
               "signal": "{attr}: {share} beyond 3 IQR (middle half {q1} – {q3})"},
        "ko": {"title": "먼 극단값", "meaning": "일부 값이 데이터 몸통에서 멀리(사분위 범위의 3배 밖) 떨어져 있습니다.",
               "actions": ["해당 행을 뽑아 원본 기록 확인", "유효 범위나 상한 규칙 결정"],
               "signal": "{attr}: {share} 가 3 IQR 밖 (중간 50% {q1} ~ {q3})"},
    },
    "spelling_variants": {
        "en": {"title": "Spelling variants", "meaning": "Values that differ only in case, spaces or separators are counted as different categories.",
               "actions": ["Normalise the spelling at the source or in preprocessing"],
               "signal": "{attr}: {groups}"},
        "ko": {"title": "표기 변형", "meaning": "대소문자·공백·구분자만 다른 값이 서로 다른 범주로 세어집니다.",
               "actions": ["원천이나 전처리에서 표기를 통일"],
               "signal": "{attr}: {groups}"},
    },
    "numbers_as_text": {
        "en": {"title": "Numbers stored as text", "meaning": "Most values read as numbers but the column is text (a stray symbol, unit or blank marker).",
               "actions": ["Find the values that are not numbers", "Convert the column so it is profiled and compared as numbers"],
               "signal": "{attr}: {share} parse as numbers"},
        "ko": {"title": "텍스트로 저장된 숫자", "meaning": "대부분 숫자로 읽히는데 열이 텍스트입니다 (기호·단위·빈칸 표시가 섞임).",
               "actions": ["숫자가 아닌 값 찾기", "숫자로 변환해 수치형으로 분석·비교되게 하기"],
               "signal": "{attr}: {share} 가 숫자로 읽힘"},
    },
}


def _fields(s: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(s)
    for key in ("share",):
        if key in s:
            out[key] = f"{s[key] * 100:.1f}%"
    for key in ("value", "q1", "q3"):
        if isinstance(s.get(key), float):
            out[key] = f"{s[key]:.4g}"
    if "groups" in s:
        out["groups"] = "; ".join(" ≈ ".join(g) for g in s["groups"])
    if "rows" in s:
        out["rows"] = f"{s['rows']:,}"
    return out


def finding_text(finding: Dict[str, Any], lang: str) -> Dict[str, Any]:
    entry = CATALOG[finding["code"]].get(lang) or CATALOG[finding["code"]]["en"]
    signals = finding.get("signals") or [finding.get("signal") or {}]
    return {"title": entry["title"], "meaning": entry["meaning"], "actions": entry["actions"],
            "signals": [entry["signal"].format(**_fields(s)) for s in signals]}


def is_profile(envelope: Optional[Dict[str, Any]]) -> bool:
    return isinstance(envelope, dict) and envelope.get("profiler") == "builtin:csv_profile"

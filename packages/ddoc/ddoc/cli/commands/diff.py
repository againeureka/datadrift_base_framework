"""``ddoc diff`` — one-line drift verdict between two datasets.

The front door of ddoc: point it at two paths (directories or files)
and get a human verdict, not just a score.

    ddoc diff baseline_data/ todays_data/

Design goals (the "ffmpeg test"):

* **Verb** — ``diff`` is the whole mental model: what changed between
  REF and CUR?
* **Tolerance** — sniffs the input modality itself. Prepared ddoc
  layouts go through the plugin ``drift_detect`` hook; bare CSV
  files/folders fall back to a built-in comparator (pandas +
  Jensen-Shannon on per-column histograms) so the common case works
  with zero setup and zero extra plugins.
* **Verdict** — every run ends in one of ``OK`` / ``DRIFT`` /
  ``BLIND``. BLIND is a first-class answer ("cannot determine"), not
  a stack trace.
* **Pipes** — exit code carries the verdict: 0 = OK, 1 = DRIFT,
  2 = BLIND/error. ``--json`` emits exactly one JSON object on stdout.

``ddoc analyze drift`` remains the low-level instrument (detectors,
fusion, snapshots); ``diff`` is the opinionated facade over it.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich import print as rprint

from .utils import call_hook_isolated, get_pmgr
from ... import archetypes
from ... import diagnosis as dx
from ...i18n import resolve_lang, severity_label, t

EXIT_OK = 0
EXIT_DRIFT = 1
EXIT_BLIND = 2

# File-extension → modality sniffing table.
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
_TEXT_EXTS = {".txt", ".md"}


# ── Sniffing ──────────────────────────────────────────────────────────


def sniff_modality(path: Path) -> Tuple[str, Dict[str, Any]]:
    """Best-effort guess of what kind of dataset ``path`` is.

    Returns ``(kind, detail)`` where kind is one of
    ``categorical | vision | audio | text | csv | unknown``.
    Layout markers win over extension counting.
    """
    if path.is_file():
        ext = path.suffix.lower()
        if ext == ".csv":
            return "csv", {"files": [str(path)]}
        if ext in _IMAGE_EXTS:
            return "vision", {"files": [str(path)]}
        if ext in _AUDIO_EXTS:
            return "audio", {"files": [str(path)]}
        if ext in _TEXT_EXTS:
            return "text", {"files": [str(path)]}
        return "unknown", {"files": [str(path)]}

    # Directory: explicit ddoc layout markers first.
    for marker in ("distributions.json", "distributions_series.json"):
        if (path / marker).is_file():
            return "categorical", {"marker": marker}

    counts: Dict[str, int] = {}
    csvs: List[str] = []
    for p in sorted(path.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        ext = p.suffix.lower()
        if ext == ".csv":
            counts["csv"] = counts.get("csv", 0) + 1
            csvs.append(str(p))
        elif ext in _IMAGE_EXTS:
            counts["vision"] = counts.get("vision", 0) + 1
        elif ext in _AUDIO_EXTS:
            counts["audio"] = counts.get("audio", 0) + 1
        elif ext in _TEXT_EXTS:
            counts["text"] = counts.get("text", 0) + 1
    if not counts:
        return "unknown", {"counts": {}}
    kind = max(counts, key=lambda k: counts[k])
    detail: Dict[str, Any] = {"counts": counts}
    if kind == "csv":
        detail["files"] = csvs
    return kind, detail


# ── Built-in CSV comparator (zero-plugin fallback) ────────────────────


def _js_divergence(p: Dict[Any, float], q: Dict[Any, float]) -> float:
    """Base-2 Jensen-Shannon divergence over aligned count dicts, ∈[0,1]."""
    import math

    keys = set(p) | set(q)
    if not keys:
        return 0.0

    def norm(d: Dict[Any, float]) -> Dict[Any, float]:
        total = sum(d.get(k, 0.0) for k in keys) or 1.0
        return {k: d.get(k, 0.0) / total for k in keys}

    pn, qn = norm(p), norm(q)
    m = {k: (pn[k] + qn[k]) / 2.0 for k in keys}

    def kl(a: Dict[Any, float], b: Dict[Any, float]) -> float:
        s = 0.0
        for k in keys:
            if a[k] > 0 and b[k] > 0:
                s += a[k] * math.log2(a[k] / b[k])
        return s

    return max(0.0, min(1.0, 0.5 * kl(pn, m) + 0.5 * kl(qn, m)))


def _rate(part: float, whole: float) -> float:
    return round(float(part) / whole, 4) if whole else 0.0


def _numeric_summary(r, c) -> Dict[str, Any]:
    """Location, spread and range of two numeric series, plus the share
    of current values that fall outside the baseline's [min, max]."""
    def stats(x) -> Dict[str, Optional[float]]:
        if x.empty:
            return {"mean": None, "std": None, "median": None, "q1": None, "q3": None,
                    "min": None, "max": None}
        return {k: round(float(v), 6) for k, v in (
            ("mean", x.mean()), ("std", x.std(ddof=0)), ("median", x.median()),
            ("q1", x.quantile(0.25)), ("q3", x.quantile(0.75)),
            ("min", x.min()), ("max", x.max()))}
    rs, cs = stats(r), stats(c)
    out: Dict[str, Any] = {k: {"ref": rs[k], "cur": cs[k]} for k in rs}
    if not r.empty and not c.empty:
        outside = ((c < rs["min"]) | (c > rs["max"])).sum()
        out["outside_ref_range"] = _rate(outside, len(c))
    return out


def _category_changes(ref_counts: Dict[Any, float], cur_counts: Dict[Any, float],
                      min_share: float = 0.01, limit: int = 5) -> Dict[str, Any]:
    """Values that appear only on one side with at least ``min_share``."""
    rt, ct = sum(ref_counts.values()) or 1.0, sum(cur_counts.values()) or 1.0
    new = sorted(((str(k), v / ct) for k, v in cur_counts.items()
                  if ref_counts.get(k, 0) == 0 and v / ct >= min_share),
                 key=lambda kv: kv[1], reverse=True)
    gone = sorted(((str(k), v / rt) for k, v in ref_counts.items()
                   if cur_counts.get(k, 0) == 0 and v / rt >= min_share),
                  key=lambda kv: kv[1], reverse=True)
    labels = list(dict.fromkeys([*ref_counts, *cur_counts]))
    movers = sorted(((str(k), ref_counts.get(k, 0) / rt, cur_counts.get(k, 0) / ct) for k in labels),
                    key=lambda m: abs(m[2] - m[1]), reverse=True)
    # A "new" value that matches a baseline value once case, spaces and
    # separators are ignored is more likely a spelling variant than news.
    def norm(x: Any) -> str:
        return re.sub(r"[\s_\-]+", "", str(x)).casefold()
    known = {norm(k): str(k) for k in ref_counts}
    variants = [[k, known[norm(k)]] for k, _ in new if norm(k) in known][:limit]
    return {"new_values": [[k, round(v, 4)] for k, v in new[:limit]],
            "variants": variants,
            "vanished_values": [[k, round(v, 4)] for k, v in gone[:limit]],
            "top_mover": [movers[0][0], round(movers[0][1], 4), round(movers[0][2], 4)]
                         if movers else None}


def _load_csv_frame(files: List[str]):
    import pandas as pd

    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


_TIME_NAME = re.compile(r"(date|time|timestamp|datetime|^ts$|_at$|_dt$|일자|날짜|일시|시각)", re.I)
SAMPLE_ROWS_MAX = 20
_SAMPLE_COLS_MAX = 12
_SAMPLE_CELL_MAX = 40


def _as_datetime(series):
    """The column parsed as date/time when it clearly is one, else None.
    Only text columns with a time-like name are tried (numbers such as an
    integer ``timestamp`` index stay numeric)."""
    import warnings

    import pandas as pd

    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series) or not _TIME_NAME.search(str(series.name)):
        return None
    values = series.dropna()
    if values.empty:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(values, errors="coerce")
    return parsed if parsed.notna().mean() >= 0.9 else None


def _period(parsed) -> Optional[Dict[str, str]]:
    parsed = parsed.dropna()
    if parsed.empty:
        return None
    lo, hi = parsed.min(), parsed.max()
    dated = all(x.hour == 0 and x.minute == 0 and x.second == 0 for x in (lo, hi))
    fmt = "%Y-%m-%d" if dated else "%Y-%m-%d %H:%M"
    return {"start": lo.strftime(fmt), "end": hi.strftime(fmt)}


def _column_type(series) -> str:
    import pandas as pd

    if series.dropna().empty:
        return "empty"
    return "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"


def _sample(frame, n: int) -> Dict[str, Any]:
    """First ``n`` rows as short strings (opt-in: raw data leaves the file)."""
    cols = list(frame.columns)
    shown = cols[:_SAMPLE_COLS_MAX]

    def cell(v: Any) -> str:
        if v is None or (isinstance(v, float) and v != v):
            return ""
        text = str(v)
        return text if len(text) <= _SAMPLE_CELL_MAX else text[:_SAMPLE_CELL_MAX - 1] + "…"

    return {"columns": [str(c) for c in shown], "hidden_columns": len(cols) - len(shown),
            "rows": [[cell(v) for v in row] for row in frame[shown].head(n).itertuples(index=False)]}


def _profile(frame, files: List[str], time_cols: Dict[str, Any], sample_rows: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "files": len(files),
        "rows": int(len(frame)),
        "columns": [{"name": str(c), "type": "datetime" if c in time_cols else _column_type(frame[c])}
                    for c in frame.columns],
        "period": None,
    }
    for col, parsed in time_cols.items():
        period = _period(parsed)
        if period:
            out["period"] = {"column": str(col), **period}
            break
    if sample_rows > 0:
        out["sample"] = _sample(frame, min(sample_rows, SAMPLE_ROWS_MAX))
    return out


def csv_diff(ref_files: List[str], cur_files: List[str], bins: int = 20,
             sample_rows: int = 0) -> Dict[str, Any]:
    """Column-wise drift between two CSV datasets.

    Numeric columns → shared-range histogram JS divergence;
    non-numeric columns → value-count JS divergence.
    ``overall_score`` = mean over shared columns. The per-column
    ref/cur distributions ride along under ``distributions`` so the
    consultation report can show *evidence*, not just scores, and
    ``summaries`` carries per-column statistics (sample size, missing
    rate, location / spread / range for numeric columns, new and vanished
    values for categorical ones) for the diagnosis.

    ``profile`` describes what was compared: rows, columns and their
    types, the period covered (from a date/time column), which columns
    exist on one side only, and — when ``sample_rows`` > 0 — the first
    rows of each side. Date/time columns give the period and are not
    compared: two periods always differ in their timestamps.
    """
    import numpy as np
    import pandas as pd

    ref, cur = _load_csv_frame(ref_files), _load_csv_frame(cur_files)
    ref_time = {c: p for c in ref.columns if (p := _as_datetime(ref[c])) is not None}
    cur_time = {c: p for c in cur.columns if (p := _as_datetime(cur[c])) is not None}
    time_cols = [c for c in ref.columns if c in ref_time and c in cur_time]
    shared = [c for c in ref.columns if c in cur.columns and c not in time_cols]
    if not shared:
        raise ValueError(
            f"no shared columns between ref {list(ref.columns)} and cur {list(cur.columns)}"
        )
    profile = {
        "ref": _profile(ref, ref_files, ref_time, sample_rows),
        "cur": _profile(cur, cur_files, cur_time, sample_rows),
        "columns": {
            "compared": [str(c) for c in shared],
            "time": [str(c) for c in time_cols],
            "only_ref": [str(c) for c in ref.columns if c not in cur.columns],
            "only_cur": [str(c) for c in cur.columns if c not in ref.columns],
        },
    }

    def _fmt_edge(x: float) -> str:
        return f"{x:.4g}"

    attribute_drifts: Dict[str, float] = {}
    distributions: Dict[str, Dict[str, Dict[str, float]]] = {}
    summaries: Dict[str, Dict[str, Any]] = {}
    for col in shared:
        r, c = ref[col].dropna(), cur[col].dropna()
        base = {
            "n": {"ref": int(len(ref)), "cur": int(len(cur))},
            "missing": {"ref": _rate(len(ref) - len(r), len(ref)),
                        "cur": _rate(len(cur) - len(c), len(cur))},
        }
        if r.empty and c.empty:
            continue
        if pd.api.types.is_numeric_dtype(r) and pd.api.types.is_numeric_dtype(c):
            lo = float(min(r.min(), c.min()))
            hi = float(max(r.max(), c.max()))
            summaries[col] = {"kind": "numeric", **base, **_numeric_summary(r, c)}
            if hi == lo:
                attribute_drifts[col] = 0.0
                continue
            whole = bool((r % 1 == 0).all() and (c % 1 == 0).all())
            if whole and hi - lo + 1 <= bins:
                # Few distinct integers (counts, ratings, levels): one bar per
                # value. Each integer would already sit in its own bin, so the
                # score is the same as the histogram's; only the labels differ.
                values = range(int(lo), int(hi) + 1)
                rvc, cvc = r.value_counts(), c.value_counts()
                rh = [int(rvc.get(v, 0)) for v in values]
                ch = [int(cvc.get(v, 0)) for v in values]
                attribute_drifts[col] = _js_divergence(dict(enumerate(rh)), dict(enumerate(ch)))
                summaries[col]["discrete"] = True
                distributions[col] = {
                    "ref": {str(v): n for v, n, m in zip(values, rh, ch) if n or m},
                    "cur": {str(v): m for v, n, m in zip(values, rh, ch) if n or m},
                }
                continue
            edges = np.linspace(lo, hi, bins + 1)
            rh, _ = np.histogram(r, bins=edges)
            ch, _ = np.histogram(c, bins=edges)
            attribute_drifts[col] = _js_divergence(
                dict(enumerate(rh.tolist())), dict(enumerate(ch.tolist()))
            )
            labels = [f"{_fmt_edge(edges[i])}–{_fmt_edge(edges[i+1])}"
                      for i in range(len(edges) - 1)]
            distributions[col] = {
                "ref": {labels[i]: int(rh[i]) for i in range(len(labels)) if rh[i] or ch[i]},
                "cur": {labels[i]: int(ch[i]) for i in range(len(labels)) if rh[i] or ch[i]},
            }
        else:
            rvc = r.astype(str).value_counts()
            cvc = c.astype(str).value_counts()
            attribute_drifts[col] = _js_divergence(rvc.to_dict(), cvc.to_dict())
            summaries[col] = {"kind": "categorical", **base,
                              **_category_changes(rvc.to_dict(), cvc.to_dict())}
            # Evidence: top categories by combined mass + an "(other)" bucket.
            top = list((rvc.add(cvc, fill_value=0)).sort_values(ascending=False).index[:8])
            def _bucket(vc) -> Dict[str, float]:
                d = {k: float(vc.get(k, 0)) for k in top}
                other = float(vc.sum() - sum(d.values()))
                if other > 0:
                    d["(other)"] = other
                return d
            distributions[col] = {"ref": _bucket(rvc), "cur": _bucket(cvc)}

    overall = sum(attribute_drifts.values()) / len(attribute_drifts) if attribute_drifts else 0.0
    return {
        "status": "success",
        "modality": "tabular",
        "detector": "builtin:csv_js",
        "overall_score": round(overall, 4),
        "attribute_drifts": {k: round(v, 4) for k, v in attribute_drifts.items()},
        "distributions": distributions,
        "summaries": summaries,
        "rows": {"ref": int(len(ref)), "cur": int(len(cur))},
        "columns_compared": shared,
        "profile": profile,
    }


# ── Verdict ───────────────────────────────────────────────────────────


def compute_verdict(
    result: Dict[str, Any], *, threshold: float, attr_threshold: float
) -> Tuple[str, str, List[Tuple[str, float]]]:
    """Map a drift envelope to (verdict, severity, top_attributes).

    DRIFT when overall ≥ threshold OR any single attribute ≥
    attr_threshold (a strong local shift shouldn't be averaged away).
    """
    overall = float(result.get("overall_score") or 0.0)
    attrs = result.get("attribute_drifts") or {}
    top = sorted(
        ((k, float(v)) for k, v in attrs.items() if isinstance(v, (int, float))),
        key=lambda kv: kv[1], reverse=True,
    )[:5]
    max_attr = top[0][1] if top else 0.0

    if overall >= attr_threshold or max_attr >= attr_threshold:
        return "DRIFT", "high", top
    if overall >= threshold:
        return "DRIFT", "medium", top
    if overall >= threshold / 3:
        return "OK", "low", top
    return "OK", "none", top


# ── Output ────────────────────────────────────────────────────────────


def _emit_json(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _print_human(payload: Dict[str, Any], lang: str) -> None:
    verdict = payload["verdict"]
    icon = {"OK": "[green]✅ OK[/green]",
            "DRIFT": "[red]⚠ DRIFT[/red]",
            "BLIND": "[yellow]◌ BLIND[/yellow]"}[verdict]
    head = (f"{icon} — {t('label.severity', lang)}: "
            f"{severity_label(payload['severity'], lang)}")
    if payload.get("overall_score") is not None:
        head += " · " + t("console.score", lang, score=payload["overall_score"],
                          threshold=payload["threshold"])
    rprint(head)
    if payload.get("top_attributes"):
        tops = " · ".join(f"{k} {v:.3f}" for k, v in payload["top_attributes"][:3])
        rprint("   " + t("console.top_attrs", lang, tops=tops))
    rprint("   " + t("console.meta", lang, modality=payload.get("modality") or "?",
                      engine=payload.get("engine"), ref=payload["ref"], cur=payload["cur"]))
    if verdict == "BLIND":
        rprint(f"   [yellow]{t('console.reason', lang)}:[/yellow] {payload.get('reason')}")
    for hint in payload.get("hints", []):
        rprint(f"   ⤷ {hint}")


def _write_report(payload: Dict[str, Any], report: Optional[Path],
                  distributions: Optional[Dict[str, Any]] = None, *,
                  lang: str, issuer: Optional[str] = None) -> None:
    """Render the diagnostic report when ``--report`` was given.
    Best-effort: a report failure must not change the verdict/exit."""
    if report is None:
        return
    try:
        from .diff_report import build_consultation_html
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            build_consultation_html(payload, distributions, lang=lang, issuer=issuer),
            encoding="utf-8",
        )
        payload["report_path"] = str(report)
        payload.setdefault("hints", []).insert(0, t("hint.report_written", lang, path=report))
    except Exception as e:  # noqa: BLE001
        payload["report_error"] = str(e)


def _finish(payload: Dict[str, Any], *, json_out: bool, lang: str,
            report: Optional[Path] = None,
            distributions: Optional[Dict[str, Any]] = None,
            issuer: Optional[str] = None) -> None:
    _write_report(payload, report, distributions, lang=lang, issuer=issuer)
    if json_out:
        _emit_json(payload)
    else:
        _print_human(payload, lang)
    code = {"OK": EXIT_OK, "DRIFT": EXIT_DRIFT, "BLIND": EXIT_BLIND}[payload["verdict"]]
    raise typer.Exit(code=code)


def _blind(reason: str, *, ref: str, cur: str, json_out: bool, lang: str,
           modality: Optional[str] = None, hints: Optional[List[str]] = None,
           report: Optional[Path] = None, issuer: Optional[str] = None) -> None:
    _finish({
        "verdict": "BLIND", "severity": "unknown", "overall_score": None,
        "modality": modality, "engine": None, "detector": None,
        "ref": ref, "cur": cur, "reason": reason, "hints": hints or [],
        "top_attributes": [], "threshold": None,
        "diagnosis": dx.build(verdict="BLIND", overall=None, top=[], threshold=0.0,
                              attr_threshold=0.0, reason=reason),
    }, json_out=json_out, lang=lang, report=report, issuer=issuer)


def _collect_evidence(kind: str, ref: Path, cur: Path,
                      result: Dict[str, Any]) -> Dict[str, Any]:
    """Per-attribute ref/cur distributions for the evidence section.

    * builtin CSV results carry ``distributions`` already;
    * categorical marker layouts: read both ``distributions.json``;
    * other engines: empty (score bars only in the report).
    """
    if isinstance(result.get("distributions"), dict):
        return result["distributions"]
    if kind == "categorical":
        try:
            rd = json.loads((ref / "distributions.json").read_text(encoding="utf-8"))
            cd = json.loads((cur / "distributions.json").read_text(encoding="utf-8"))
            out: Dict[str, Any] = {}
            for attr, rv in rd.items():
                cv = cd.get(attr)
                if (isinstance(rv, dict) and isinstance(cv, dict)
                        and all(isinstance(x, (int, float)) for x in rv.values())
                        and all(isinstance(x, (int, float)) for x in cv.values())):
                    out[attr] = {"ref": rv, "cur": cv}
            return out
        except Exception:  # noqa: BLE001 — evidence is optional
            return {}
    return {}


# ── Command ───────────────────────────────────────────────────────────


def diff_command(
    ref: Path = typer.Argument(..., help="Baseline dataset (directory or file)."),
    cur: Path = typer.Argument(..., help="Current dataset (directory or file)."),
    threshold: float = typer.Option(
        0.15, "--threshold",
        help="Overall-score threshold for the DRIFT verdict.",
    ),
    attr_threshold: float = typer.Option(
        0.25, "--attr-threshold",
        help="Single-attribute threshold — one strongly drifted attribute "
             "triggers DRIFT even when the average looks calm.",
    ),
    detector: str = typer.Option(
        "default", "--detector",
        help="Detector passed through to plugins (see `ddoc plugin detectors`).",
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit exactly one JSON verdict object on stdout.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Show plugin diagnostics instead of silencing them.",
    ),
    report: Optional[Path] = typer.Option(
        None, "--report",
        help="Write a self-contained HTML diagnostic report (verdict, why, "
             "what changed, evidence, what to check next) to this path.",
    ),
    lang: Optional[str] = typer.Option(
        None, "--lang",
        help="Language for the report, hints and console text: en | ko "
             "(default: $DDOC_LANG, else en). JSON keys and verdicts stay English.",
    ),
    issuer: Optional[str] = typer.Option(
        None, "--issuer",
        help="Organization printed as the report's issuer (default: $DDOC_ISSUER, else 'ddoc').",
    ),
    with_embeddings: bool = typer.Option(
        False, "--with-embeddings",
        help="Ask image / text plugins to compare embeddings too (loads the model, slower); "
             "the report then includes an embedding map.",
    ),
    sample_rows: int = typer.Option(
        0, "--sample-rows", min=0, max=20,
        help="Include the first N rows of each CSV side in the report and JSON "
             "(off by default: the rows are raw data).",
    ),
):
    """One-line drift verdict between REF and CUR.

    Exit code carries the verdict: 0 = OK, 1 = DRIFT, 2 = BLIND/error.

    Examples:
        ddoc diff baseline_data/ todays_data/
        ddoc diff jan.csv aug.csv --json
        ddoc diff camA_ref/ camA_now/ --attr-threshold 0.3
    """
    lang = resolve_lang(lang)
    issuer = issuer or os.environ.get("DDOC_ISSUER") or None
    ref_s, cur_s = str(ref), str(cur)
    for label, p in (("ref", ref), ("cur", cur)):
        if not p.exists():
            _blind(t("reason.path_missing", lang, label=label, path=p),
                   ref=ref_s, cur=cur_s, json_out=json_out, lang=lang, report=report,
                   issuer=issuer)

    ref_kind, ref_detail = sniff_modality(ref)
    cur_kind, cur_detail = sniff_modality(cur)
    kind = ref_kind if ref_kind == cur_kind else "mixed"

    # 1) Plugin hook first — prepared layouts get the full detector stack.
    from .analyze.drift import _SilencePluginIO, _merge_plugin_results

    result: Optional[Dict[str, Any]] = None
    engine: Optional[str] = None
    try:
        with _SilencePluginIO(json_out=True, quiet=not verbose):
            hook_results, hook_errors = call_hook_isolated(
                get_pmgr(), "drift_detect",
                snapshot_id_ref="__path__", snapshot_id_cur="__path__",
                data_path_ref=ref_s, data_path_cur=cur_s,
                data_hash_ref="", data_hash_cur="",
                detector=detector,
                cfg={"baseline_cache": None, "current_cache": None,
                     "baseline_metadata": None, "current_metadata": None,
                     "with_embeddings": with_embeddings},
                output_path="analysis/diff",
            )
        if verbose:
            for plugin_name, err in hook_errors:
                rprint(f"[yellow]plugin {plugin_name} failed: {err}[/yellow]")
        valid = [r for r in (hook_results or []) if r is not None
                 and r.get("status") not in ("error", "unsupported")]
        if valid:
            result = _merge_plugin_results(valid, hook_name="drift_detect")
            engine = "plugin"
    except Exception as e:  # noqa: BLE001 — fall through to builtin/BLIND
        if verbose:
            rprint(f"[yellow]plugin hook failed: {e}[/yellow]")

    # 2) Built-in CSV fallback — the bare-folder common case.
    if result is None and kind == "csv":
        try:
            result = csv_diff(ref_detail.get("files") or [ref_s],
                              cur_detail.get("files") or [cur_s], sample_rows=sample_rows)
            engine = "builtin"
        except Exception as e:  # noqa: BLE001
            _blind(t("reason.csv_failed", lang, error=e), ref=ref_s, cur=cur_s,
                   json_out=json_out, lang=lang, modality="tabular", report=report,
                   issuer=issuer)

    # 3) Still nothing → honest BLIND with what to do about it.
    if result is None:
        hints = []
        if kind == "vision":
            hints.append(t("hint.install_vision", lang))
        elif kind in ("audio", "text"):
            hints.append(t("hint.install_plugin", lang, kind=kind))
        elif kind == "unknown":
            hints.append(t("hint.supported_inputs", lang))
        _blind(
            t("reason.no_engine", lang, kind=kind),
            ref=ref_s, cur=cur_s, json_out=json_out, lang=lang, modality=kind,
            hints=hints, report=report, issuer=issuer,
        )

    verdict, severity, top = compute_verdict(
        result, threshold=threshold, attr_threshold=attr_threshold,
    )
    evidence = _collect_evidence(kind, ref, cur, result)
    diagnosis = dx.build(
        verdict=verdict, overall=float(result.get("overall_score") or 0.0), top=top,
        threshold=threshold, attr_threshold=attr_threshold,
        result=result, evidence=evidence,
    )
    hints: List[str] = [dx.warning_text(w, lang) for w in diagnosis["warnings"]
                        if w["code"] == "small_sample"]
    try:  # the reading line is a convenience; it must never cost the verdict
        titles = [archetypes.card_text(c, lang)["title"] for c in diagnosis["archetypes"]
                  if c["match"] == "match" and c["code"] != "stable"]
    except Exception:  # noqa: BLE001
        titles = []
    if titles:
        hints.append(t("reading.headline", lang) + " " + " · ".join(titles))
    if verdict == "DRIFT":
        hints += dx.check_texts(diagnosis, lang)[:3]
        if report is None:
            hints.append(t("hint.report_suggest", lang))

    _finish({
        "verdict": verdict,
        "severity": severity,
        "overall_score": float(result.get("overall_score") or 0.0),
        "threshold": threshold,
        "attr_threshold": attr_threshold,
        "top_attributes": top,
        "modality": result.get("modality") or kind,
        "detector": result.get("detector") or detector,
        "engine": engine,
        "ref": ref_s, "cur": cur_s,
        "hints": hints,
        "diagnosis": diagnosis,
        "raw": result,
    }, json_out=json_out, lang=lang, report=report, distributions=evidence, issuer=issuer)

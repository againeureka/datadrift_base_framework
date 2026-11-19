"""Turn a drift result into a diagnosis a person can act on.

`ddoc diff` produces scores. This module explains them: why the verdict
came out the way it did, what changed in each attribute, which known
patterns the change matches (new or vanished values, a shifted mix, values
outside the baseline range, a moved mean, a changed spread or missing
rate), how much data the comparison rests on, and what to check next.

The output is language-neutral — codes and numbers only — so it can ride
in the JSON payload; :func:`render` turns it into sentences through the
message catalog. Every rule here is a plain threshold on the summaries;
none of it claims to know how the change affects a model.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .i18n import t

# Pattern thresholds.
SHARE_SHIFT = 0.10        # a category's share moved by ≥ 10 percentage points
NEW_VALUE_SHARE = 0.01    # a value counts as new / vanished at ≥ 1 % share
OUT_OF_RANGE = 0.05       # ≥ 5 % of current values outside the baseline [min, max]
MEAN_SHIFT_STD = 0.5      # mean moved by ≥ 0.5 baseline standard deviations
SPREAD_RATIO = 1.5        # std grew or shrank by ≥ 1.5×
MISSING_CHANGE = 0.05     # missing rate moved by ≥ 5 percentage points
SMALL_SAMPLE = 30         # fewer rows than this on either side
MAX_ATTRIBUTES = 5        # attributes described in the diagnosis

_STAT_KEYS = {"mean", "std", "min", "max", "median", "p50", "var", "variance"}


def is_stat_dict(values: Dict[str, Any]) -> bool:
    """True for summary statistics (mean / std / ...) rather than counts."""
    return bool(values) and set(values) <= _STAT_KEYS


def _shares(counts: Dict[str, float]) -> Dict[str, float]:
    total = sum(counts.values())
    return {k: (v / total if total else 0.0) for k, v in counts.items()}


def _looks_like_counts(counts: Dict[str, float]) -> bool:
    """Proportions (summing to ~1) carry no sample size."""
    total = sum(counts.values())
    return total > 1.5 or any(v > 1 for v in counts.values())


def _summary_from_counts(ref: Dict[str, float], cur: Dict[str, float]) -> Dict[str, Any]:
    rs, cs = _shares(ref), _shares(cur)
    labels = list(dict.fromkeys([*ref, *cur]))
    movers = sorted(((k, rs.get(k, 0.0), cs.get(k, 0.0)) for k in labels),
                    key=lambda m: abs(m[2] - m[1]), reverse=True)
    summary: Dict[str, Any] = {
        "kind": "categorical",
        "top_mover": [movers[0][0], round(movers[0][1], 4), round(movers[0][2], 4)] if movers else None,
        "new_values": [[k, round(cs[k], 4)] for k in labels
                       if rs.get(k, 0.0) == 0 and cs.get(k, 0.0) >= NEW_VALUE_SHARE][:5],
        "vanished_values": [[k, round(rs[k], 4)] for k in labels
                            if cs.get(k, 0.0) == 0 and rs.get(k, 0.0) >= NEW_VALUE_SHARE][:5],
    }
    if _looks_like_counts(ref) and _looks_like_counts(cur):
        summary["n"] = {"ref": int(sum(ref.values())), "cur": int(sum(cur.values()))}
    return summary


def _patterns(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    kind = summary.get("kind")
    found: List[Dict[str, Any]] = []
    if kind == "categorical":
        if summary.get("new_values"):
            found.append({"code": "new_values", "values": summary["new_values"]})
        if summary.get("vanished_values"):
            found.append({"code": "vanished_values", "values": summary["vanished_values"]})
        mover = summary.get("top_mover")
        # When values appeared or vanished, the mix change is usually that
        # same event seen from the other side (the remaining values grow or
        # shrink to make room); only report it when it is larger than that.
        told = (summary.get("new_values") or []) + (summary.get("vanished_values") or [])
        told_mass = sum(v[1] for v in told)
        if (mover and abs(mover[2] - mover[1]) >= SHARE_SHIFT
                and mover[0] not in {v[0] for v in told}
                and abs(mover[2] - mover[1]) > told_mass + 1e-9):
            found.append({"code": "share_shift", "label": mover[0],
                          "ref": mover[1], "cur": mover[2]})
    elif kind == "numeric":
        outside = summary.get("outside_ref_range")
        if outside is not None and outside >= OUT_OF_RANGE:
            found.append({"code": "out_of_range", "share": outside})
        mean, std = summary.get("mean") or {}, summary.get("std") or {}
        if mean.get("ref") is not None and mean.get("cur") is not None:
            ref_std = std.get("ref") or 0.0
            moved = abs(mean["cur"] - mean["ref"])
            if (ref_std > 0 and moved >= MEAN_SHIFT_STD * ref_std) or (ref_std == 0 and moved > 0):
                found.append({"code": "mean_shift", "ref": mean["ref"], "cur": mean["cur"]})
        if std.get("ref") and std.get("cur") is not None:
            ratio = std["cur"] / std["ref"]
            if ratio >= SPREAD_RATIO or ratio <= 1 / SPREAD_RATIO:
                found.append({"code": "spread_change", "ref": std["ref"], "cur": std["cur"]})
    missing = summary.get("missing") or {}
    if missing and abs((missing.get("cur") or 0) - (missing.get("ref") or 0)) >= MISSING_CHANGE:
        found.append({"code": "missing_change", "ref": missing.get("ref"), "cur": missing.get("cur")})
    return found


def patterns(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Change patterns for one attribute summary."""
    return _patterns(summary)


def _rationale(verdict: str, overall: Optional[float], top: List[Tuple[str, float]],
               threshold: float, attr_threshold: float, reason: Optional[str]) -> List[Dict[str, Any]]:
    if verdict == "BLIND":
        return [{"code": "blind", "reason": reason or ""}]
    overall = overall or 0.0
    max_attr = top[0] if top else ("", 0.0)
    if verdict == "OK":
        return [{"code": "ok", "score": overall, "threshold": threshold,
                 "attr_threshold": attr_threshold, "max_attr": max_attr[1]}]
    out: List[Dict[str, Any]] = []
    for name, score in top:
        if score >= attr_threshold:
            out.append({"code": "attr", "attr": name, "score": score, "threshold": attr_threshold})
    if overall >= threshold:
        out.append({"code": "overall", "score": overall, "threshold": threshold})
    return out


def build(
    *,
    verdict: str,
    overall: Optional[float],
    top: List[Tuple[str, float]],
    threshold: float,
    attr_threshold: float,
    result: Optional[Dict[str, Any]] = None,
    evidence: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the language-neutral diagnosis for one comparison."""
    result = result or {}
    evidence = evidence or {}
    summaries: Dict[str, Any] = dict(result.get("summaries") or {})
    for name, dists in evidence.items():
        if name in summaries:
            continue
        ref, cur = dists.get("ref") or {}, dists.get("cur") or {}
        if is_stat_dict(ref) or is_stat_dict(cur):
            summaries[name] = {"kind": "stats",
                               "values": {k: {"ref": ref.get(k), "cur": cur.get(k)}
                                          for k in dict.fromkeys([*ref, *cur])}}
        else:
            summaries[name] = _summary_from_counts(ref, cur)

    scores = dict(result.get("attribute_drifts") or {}) or dict(top)
    ordered = sorted(((k, float(v)) for k, v in scores.items() if isinstance(v, (int, float))),
                     key=lambda kv: kv[1], reverse=True)

    attributes, checks = [], []
    for name, score in ordered[:MAX_ATTRIBUTES]:
        summary = summaries.get(name)
        patterns = _patterns(summary) if summary else []
        attributes.append({"name": name, "score": round(score, 4),
                           "summary": summary, "patterns": patterns})
        # Checks only for attributes that drove the verdict; patterns in
        # calmer attributes are still listed under "what changed".
        if verdict == "DRIFT" and score >= threshold:
            checks += [{"code": p["code"], "attr": name, **{k: v for k, v in p.items() if k != "code"}}
                       for p in patterns]

    # One sample size for the whole comparison when it is well defined
    # (CSV rows, or count-based attributes that all sum to the same N);
    # the small-sample check uses the smallest size seen anywhere.
    sized = [s["n"] for s in summaries.values() if isinstance(s, dict) and s.get("n")]
    rows = result.get("rows")
    if isinstance(rows, dict) and rows.get("ref") is not None:
        samples = {"ref": rows.get("ref"), "cur": rows.get("cur")}
    elif sized and all(n == sized[0] for n in sized):
        samples = dict(sized[0])
    else:
        samples = None
    smallest = min(sized, key=lambda n: min(n["ref"], n["cur"])) if sized else samples

    warnings: List[Dict[str, Any]] = []
    if smallest and min(smallest["ref"], smallest["cur"]) < SMALL_SAMPLE:
        warnings.append({"code": "small_sample", "ref": smallest["ref"], "cur": smallest["cur"]})
    if verdict != "BLIND" and ordered and not summaries:
        warnings.append({"code": "no_breakdown"})

    if verdict == "DRIFT" and not checks and ordered:
        checks.append({"code": "inspect", "attr": ordered[0][0]})

    # A column the current data no longer has is worth checking whatever
    # the verdict: the comparison could not see it at all.
    profile = result.get("profile") if isinstance(result.get("profile"), dict) else None
    dropped = ((profile or {}).get("columns") or {}).get("only_ref") or []
    if verdict != "BLIND" and dropped:
        checks.insert(0, {"code": "column_dropped", "columns": list(dropped)})

    inspection = [] if verdict == "BLIND" else _inspection(
        summaries={k: v for k, v in summaries.items() if isinstance(v, dict)},
        ordered=ordered, overall=overall, threshold=threshold, attr_threshold=attr_threshold,
        smallest=smallest, uniform=samples is not None, profile=profile)

    return {
        "rationale": _rationale(verdict, overall, ordered[:1] if verdict == "OK" else ordered,
                                threshold, attr_threshold, reason),
        "samples": samples,
        "attributes": attributes,
        "warnings": warnings,
        "checks": checks,
        "inspection": inspection,
    }


# ── Inspection table ──────────────────────────────────────────────────
#
# Like a vehicle inspection sheet: items grouped by system, each with a
# measured value, the criterion and a result. The verdict rules are not
# changed by this table; it lays out what was measured against what.
# "fail" marks what produced a DRIFT verdict (the scores); "caution"
# marks a pattern over its threshold; "pass" rows summarise the rest.

SYSTEMS = ("schema", "sample", "distribution", "completeness", "numeric", "category")
_ROWS_PER_ITEM = 8


def _measures(summary: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """The number each inspection item looks at, for one attribute."""
    out: Dict[str, Optional[float]] = {}
    missing = summary.get("missing") or {}
    if missing:
        out["missing_change"] = round((missing.get("cur") or 0.0) - (missing.get("ref") or 0.0), 4)
    if summary.get("kind") == "numeric":
        out["out_of_range"] = summary.get("outside_ref_range")
        mean, std = summary.get("mean") or {}, summary.get("std") or {}
        if mean.get("ref") is not None and mean.get("cur") is not None:
            moved, ref_std = abs(mean["cur"] - mean["ref"]), std.get("ref") or 0.0
            out["mean_shift"] = round(moved / ref_std, 3) if ref_std else (0.0 if not moved else None)
        if std.get("ref") and std.get("cur") is not None:
            out["spread_change"] = round(std["cur"] / std["ref"], 3)
    elif summary.get("kind") == "categorical":
        out["new_values"] = round(sum(v[1] for v in summary.get("new_values") or []), 4)
        out["vanished_values"] = round(sum(v[1] for v in summary.get("vanished_values") or []), 4)
        mover = summary.get("top_mover")
        if mover:
            out["share_shift"] = round(abs(mover[2] - mover[1]), 4)
    return out


def _spread_distance(ratio: Optional[float]) -> float:
    """How far a std ratio is from 1, either way (×2 and ×0.5 are equal)."""
    if not ratio:
        return 0.0
    return ratio if ratio >= 1 else 1 / ratio


def _pattern_rows(system: str, item: str, summaries: Dict[str, Any],
                  flagged: Dict[str, Dict[str, Any]], skip: set) -> List[Dict[str, Any]]:
    """Caution rows for attributes whose pattern fired, then one pass row
    with the largest value among the others."""
    measured = {name: m[item] for name, sm in summaries.items() if item in (m := _measures(sm))}
    if not measured:
        return []
    rows: List[Dict[str, Any]] = []
    hits = [name for name in measured if item in flagged.get(name, {})]
    skip = skip - set(hits)
    for name in hits[:_ROWS_PER_ITEM]:
        rows.append({"system": system, "item": item, "attr": name, "status": "caution",
                     "value": measured[name], "detail": flagged[name][item]})
    if len(hits) > _ROWS_PER_ITEM:
        rows.append({"system": system, "item": item, "more": len(hits) - _ROWS_PER_ITEM,
                     "status": "caution"})
    rest = {k: v for k, v in measured.items() if k not in hits and k not in skip and v is not None}
    if rest:
        key = (lambda kv: _spread_distance(kv[1])) if item == "spread_change" else (lambda kv: abs(kv[1]))
        name, value = max(rest.items(), key=key)
        rows.append({"system": system, "item": item, "count": len(rest), "attr": name,
                     "value": value, "status": "pass", "all": not hits and not skip & set(measured)})
    return rows


def _inspection(*, summaries: Dict[str, Any], ordered: List[Tuple[str, float]],
                overall: Optional[float], threshold: float, attr_threshold: float,
                smallest: Optional[Dict[str, int]], uniform: bool,
                profile: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    cols = (profile or {}).get("columns")
    if cols:
        one_sided = len(cols.get("only_ref") or []) + len(cols.get("only_cur") or [])
        rows.append({"system": "schema", "item": "columns",
                     "status": "caution" if one_sided else "pass",
                     "value": {"compared": len(cols.get("compared") or []),
                               "only_ref": cols.get("only_ref") or [],
                               "only_cur": cols.get("only_cur") or []}})

    if smallest:
        rows.append({"system": "sample", "item": "sample_size", "smallest": not uniform,
                     "status": "pass" if min(smallest["ref"], smallest["cur"]) >= SMALL_SAMPLE else "caution",
                     "value": {"ref": smallest["ref"], "cur": smallest["cur"]}})

    if overall is not None and ordered:
        rows.append({"system": "distribution", "item": "overall", "value": round(overall, 4),
                     "threshold": threshold,
                     "status": "fail" if overall >= threshold else "pass"})
        over = [(k, v) for k, v in ordered if v >= threshold]
        for name, score in over[:_ROWS_PER_ITEM]:
            rows.append({"system": "distribution", "item": "attr_score", "attr": name,
                         "value": round(score, 4), "threshold": threshold,
                         "attr_threshold": attr_threshold,
                         "status": "fail" if score >= attr_threshold else "caution"})
        if len(over) > _ROWS_PER_ITEM:
            rows.append({"system": "distribution", "item": "attr_score",
                         "more": len(over) - _ROWS_PER_ITEM, "status": "caution"})
        rest = [(k, v) for k, v in ordered if v < threshold]
        if rest:
            rows.append({"system": "distribution", "item": "attr_score", "count": len(rest),
                         "attr": rest[0][0], "value": round(rest[0][1], 4), "status": "pass",
                         "threshold": threshold, "attr_threshold": attr_threshold, "all": not over})

    flagged = {name: {p["code"]: p for p in _patterns(sm)} for name, sm in summaries.items()}
    rows += _pattern_rows("completeness", "missing_change", summaries, flagged, set())
    numeric = {k: v for k, v in summaries.items() if v.get("kind") == "numeric"}
    for item in ("out_of_range", "mean_shift", "spread_change"):
        rows += _pattern_rows("numeric", item, numeric, flagged, set())
    category = {k: v for k, v in summaries.items() if v.get("kind") == "categorical"}
    told = {k for k in category if {"new_values", "vanished_values"} & set(flagged.get(k, {}))}
    for item in ("new_values", "vanished_values", "share_shift"):
        # a mix change explained by values appearing / vanishing is not
        # also reported as a clean pass on the mix
        rows += _pattern_rows("category", item, category, flagged, told if item == "share_shift" else set())
    return rows


# ── Rendering ─────────────────────────────────────────────────────────


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _num(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.4g}"


def _values(pairs: List[List[Any]]) -> str:
    return ", ".join(f"{k} ({_pct(v)})" for k, v in pairs)


def rationale_text(item: Dict[str, Any], lang: str) -> str:
    return t(f"rationale.{item['code']}", lang, **{k: v for k, v in item.items() if k != "code"})


def summary_text(summary: Optional[Dict[str, Any]], lang: str) -> Optional[str]:
    """One line describing what changed in an attribute."""
    if not summary:
        return None
    kind = summary.get("kind")
    if kind == "numeric":
        mean, median = summary.get("mean") or {}, summary.get("median") or {}
        if mean.get("ref") is None or mean.get("cur") is None:
            return None
        change = ""
        if mean["ref"]:
            change = f" ({(mean['cur'] - mean['ref']) / abs(mean['ref']) * 100:+.1f}%)"
        return t("summary.numeric", lang, mean_ref=_num(mean["ref"]), mean_cur=_num(mean["cur"]),
                 change=change, median_ref=_num(median.get("ref")), median_cur=_num(median.get("cur")))
    if kind == "categorical":
        mover = summary.get("top_mover")
        if not mover:
            return None
        return t("summary.categorical", lang, label=mover[0], ref=_pct(mover[1]), cur=_pct(mover[2]),
                 delta=t("fmt.pp", lang, value=(mover[2] - mover[1]) * 100))
    if kind == "stats":
        parts = [f"{k} {_num(v.get('ref'))} → {_num(v.get('cur'))}"
                 for k, v in (summary.get("values") or {}).items()]
        return " · ".join(parts) or None
    return None


def pattern_text(pattern: Dict[str, Any], lang: str) -> str:
    code = pattern["code"]
    if code in ("new_values", "vanished_values"):
        return t(f"pattern.{code}", lang, values=_values(pattern["values"]))
    if code == "share_shift":
        return t("pattern.share_shift", lang, label=pattern["label"],
                 ref=_pct(pattern["ref"]), cur=_pct(pattern["cur"]))
    if code == "out_of_range":
        return t("pattern.out_of_range", lang, share=_pct(pattern["share"]))
    if code == "mean_shift":
        return t("pattern.mean_shift", lang, ref=_num(pattern["ref"]), cur=_num(pattern["cur"]))
    if code == "spread_change":
        return t("pattern.spread_change", lang, ref=_num(pattern["ref"]), cur=_num(pattern["cur"]))
    if code == "missing_change":
        return t("pattern.missing_change", lang, ref=_pct(pattern["ref"]), cur=_pct(pattern["cur"]))
    return code


def check_text(check: Dict[str, Any], lang: str) -> str:
    code, attr = check["code"], check.get("attr", "?")
    if code in ("new_values", "vanished_values"):
        return t(f"check.{code}", lang, attr=attr, values=_values(check["values"]))
    if code == "share_shift":
        return t("check.share_shift", lang, attr=attr, label=check["label"],
                 ref=_pct(check["ref"]), cur=_pct(check["cur"]))
    if code in ("out_of_range", "mean_shift"):
        return t("check.value_shift", lang, attr=attr)
    if code == "spread_change":
        return t("check.spread_change", lang, attr=attr)
    if code == "column_dropped":
        return t("check.column_dropped", lang, columns=", ".join(check.get("columns") or []))
    if code == "missing_change":
        return t("check.missing_change", lang, attr=attr,
                 ref=_pct(check["ref"]), cur=_pct(check["cur"]))
    return t("check.inspect", lang, attr=attr)


def check_texts(diagnosis: Dict[str, Any], lang: str) -> List[str]:
    """Check sentences, de-duplicated (one value-shift check per attribute)."""
    seen, out = set(), []
    for check in diagnosis.get("checks") or []:
        text = check_text(check, lang)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def warning_text(warning: Dict[str, Any], lang: str) -> str:
    return t(f"warning.{warning['code']}", lang, **{k: v for k, v in warning.items() if k != "code"})


_CRITERIA = {  # item → (message key, format values)
    "sample_size": ("insp.c.at_least", {"value": SMALL_SAMPLE}),
    "missing_change": ("insp.c.below", {"value": f"{MISSING_CHANGE * 100:.0f}%p"}),
    "out_of_range": ("insp.c.below", {"value": f"{OUT_OF_RANGE * 100:.0f}%"}),
    "mean_shift": ("insp.c.below", {"value": f"{MEAN_SHIFT_STD:g}σ"}),
    "spread_change": ("insp.c.spread", {"lo": f"{1 / SPREAD_RATIO:.2f}", "hi": f"{SPREAD_RATIO:g}"}),
    "new_values": ("insp.c.no_value", {"value": f"{NEW_VALUE_SHARE * 100:.0f}%"}),
    "vanished_values": ("insp.c.no_value", {"value": f"{NEW_VALUE_SHARE * 100:.0f}%"}),
    "share_shift": ("insp.c.below", {"value": f"{SHARE_SHIFT * 100:.0f}%p"}),
}


def _measured(row: Dict[str, Any], lang: str) -> str:
    item, v, detail = row["item"], row.get("value"), row.get("detail") or {}
    if row.get("more"):
        return "—"
    if item == "columns":
        parts = [t("insp.m.columns", lang, count=v["compared"])]
        if v["only_cur"]:
            parts.append(t("insp.m.only_cur", lang, columns=", ".join(v["only_cur"])))
        if v["only_ref"]:
            parts.append(t("insp.m.only_ref", lang, columns=", ".join(v["only_ref"])))
        return " · ".join(parts)
    if item == "sample_size":
        return f"{v['ref']:,} / {v['cur']:,}"
    if item in ("overall", "attr_score"):
        text = f"{v:.3f}"
    elif item == "missing_change":
        text = t("fmt.pp", lang, value=v * 100)
    elif item == "out_of_range":
        text = _pct(v)
    elif item == "mean_shift":
        text = "—" if v is None else f"{v:.2f}σ"
        if detail:
            text += f" ({_num(detail.get('ref'))} → {_num(detail.get('cur'))})"
    elif item == "spread_change":
        text = f"×{v:.2f}"
    elif item in ("new_values", "vanished_values"):
        return _values(detail["values"]) if detail else t("insp.m.none", lang)
    elif item == "share_shift":
        if detail:
            return (f"{detail['label']} {_pct(detail['ref'])} → {_pct(detail['cur'])} "
                    f"({t('fmt.pp', lang, value=(detail['cur'] - detail['ref']) * 100)})")
        text = t("fmt.pp", lang, value=v * 100).lstrip("+")
    else:
        text = str(v)
    if row["status"] == "pass" and row.get("count", 0) > 1 and v:
        return t("insp.m.max", lang, value=text, attr=row["attr"])
    return text


def _criterion(row: Dict[str, Any], lang: str) -> str:
    item = row["item"]
    if item == "columns":
        return t("insp.c.columns", lang)
    if item == "overall":
        return t("insp.c.below", lang, value=f"{row['threshold']:g}")
    if item == "attr_score":
        return t("insp.c.score", lang, value=f"{row['threshold']:g}", fail=f"{row['attr_threshold']:g}")
    key, values = _CRITERIA[item]
    return t(key, lang, **values)


def inspection_text(row: Dict[str, Any], lang: str) -> Dict[str, str]:
    """One inspection row as display strings."""
    if row.get("more"):
        target = t("insp.more", lang, count=row["more"])
    elif row.get("count") is not None:
        target = t("insp.all" if row.get("all") else "insp.rest", lang, count=row["count"])
    elif row.get("smallest"):
        target = t("insp.smallest", lang)
    else:
        target = row.get("attr") or "—"
    return {
        "system": t(f"insp.sys.{row['system']}", lang),
        "item": t(f"insp.item.{row['item']}", lang),
        "target": target,
        "measured": _measured(row, lang),
        "criterion": _criterion(row, lang),
        "result": t(f"insp.status.{row['status']}", lang),
    }

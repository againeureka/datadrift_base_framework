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

    return {
        "rationale": _rationale(verdict, overall, ordered[:1] if verdict == "OK" else ordered,
                                threshold, attr_threshold, reason),
        "samples": samples,
        "attributes": attributes,
        "warnings": warnings,
        "checks": checks,
    }


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

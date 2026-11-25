"""The diagnosis as display data for the GUI's result view.

``POST /diff`` returns the CLI's payload; this turns it into what a screen
shows — the verdict line, the pattern cards with their sentences, one row
per attribute with a baseline-vs-current thumbnail — using the same text
and drawing functions as the diagnostic report, so the two never disagree.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .. import archetypes as ar
from .. import diagnosis as dx
from ..cli.commands import diff_report as rep
from ..i18n import severity_label, t

MAX_CARDS = 6
MAX_ATTRIBUTES = 12


def _side(profile: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(profile, dict):
        return None
    period = profile.get("period") if isinstance(profile.get("period"), dict) else None
    return {"rows": profile.get("rows"), "total_rows": profile.get("total_rows"),
            "columns": len(profile.get("columns") or []), "files": profile.get("files") or 1,
            "period": {"start": period.get("start"), "end": period.get("end")} if period else None}


def diff_view(payload: Dict[str, Any], lang: str) -> Dict[str, Any]:
    verdict = payload.get("verdict", "BLIND")
    severity = payload.get("severity", "unknown")
    threshold = payload.get("threshold") or 0.15
    attr_threshold = payload.get("attr_threshold") or 0.25
    raw = payload.get("raw") if isinstance(payload.get("raw"), dict) else {}
    diagnosis = payload.get("diagnosis") if isinstance(payload.get("diagnosis"), dict) else {}
    profile = raw.get("profile") if isinstance(raw.get("profile"), dict) else None
    dists = raw.get("distributions") if isinstance(raw.get("distributions"), dict) else {}
    token, icon = rep._status(verdict, severity)

    cards: List[Dict[str, Any]] = []
    archetypes = [c for c in diagnosis.get("archetypes") or [] if c.get("code") in ar.CATALOG]
    for card in sorted(archetypes, key=lambda c: c.get("match") != "match")[:MAX_CARDS]:
        text = ar.card_text(card, lang, profile=profile)
        cards.append({"code": card["code"], "match": card["match"],
                      "level": "attribute" if card["code"] in ar.ATTR_CODES else "data",
                      "title": text["title"], "meaning": text["meaning"], "signals": text["signals"],
                      "actions": text["actions"][:2], "caveat": text["caveat"],
                      "attrs": card.get("attrs") or (card.get("signal") or {}).get("attrs") or []})

    scores = sorted(((k, float(v)) for k, v in (raw.get("attribute_drifts")
                                                 or dict(payload.get("top_attributes") or [])).items()
                     if isinstance(v, (int, float))), key=lambda kv: kv[1], reverse=True)
    status, _ = rep._attr_status(diagnosis, dict(scores), threshold, attr_threshold)
    described = {a.get("name"): a for a in diagnosis.get("attributes") or [] if isinstance(a, dict)}
    summaries = raw.get("summaries") if isinstance(raw.get("summaries"), dict) else {}
    attributes = []
    for name, score in scores[:MAX_ATTRIBUTES]:
        summary = (described.get(name) or {}).get("summary") or summaries.get(name)
        attributes.append({
            "name": name, "score": round(score, 4), "status": status.get(name, "pass"),
            "kind": (summary or {}).get("kind"),
            "spark": rep._spark(dists.get(name), summary),
            "summary": dx.summary_text(summary, lang) if summary else None,
            "patterns": [dx.pattern_text(p, lang) for p in (described.get(name) or {}).get("patterns") or []],
        })

    columns = (profile or {}).get("columns") or {}
    return {
        "verdict": verdict, "severity": severity, "severity_label": severity_label(severity, lang),
        "token": token, "icon": icon,
        "score": payload.get("overall_score"), "threshold": threshold, "attr_threshold": attr_threshold,
        "reason": payload.get("reason"),
        "headline": " · ".join(c["title"] for c in cards if c["match"] == "match"),
        "rationale": [dx.rationale_text(item, lang) for item in diagnosis.get("rationale") or []],
        "cards": cards,
        "attributes": attributes, "attributes_total": len(scores),
        "sides": {"ref": _side((profile or {}).get("ref")), "cur": _side((profile or {}).get("cur"))},
        "columns": {k: columns.get(k) or [] for k in ("compared", "time", "running", "only_ref", "only_cur")},
        "checks": dx.check_texts(diagnosis, lang)[:5],
        "warnings": [dx.warning_text(w, lang) for w in diagnosis.get("warnings") or []],
        "hints": [h for h in payload.get("hints") or [] if isinstance(h, str)],
        "labels": {"ref": t("chart.ref", lang), "cur": t("chart.cur", lang)},
        "modality": payload.get("modality"), "engine": payload.get("engine"),
    }

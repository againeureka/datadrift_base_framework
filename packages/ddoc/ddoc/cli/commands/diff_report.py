"""Consultation note renderer for ``ddoc diff --report``.

A drift claim is only useful if the reader can *believe* it. This
module turns a diff verdict payload into a self-contained one-page
HTML "consultation note" — verdict, per-attribute drift bars, ref/cur
distribution evidence, and a prescription — in the spirit of a
doctor's note: diagnosis, findings, treatment plan.

Deliberately dependency-free (stdlib only, inline CSS, no JS) so the
file opens anywhere and ships over email/air-gap unchanged.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ... import diagnosis as dx
from ...i18n import DEFAULT_LANG, severity_label, t

_VERDICT_STYLE = {
    "OK":    ("#0a7d33", "#e7f6ec", "✅"),
    "DRIFT": ("#b3261e", "#fdeceb", "⚠"),
    "BLIND": ("#8a6d00", "#fdf6dd", "◌"),
}

_CSS = """
body { font-family: -apple-system, 'Segoe UI', 'Apple SD Gothic Neo', 'Malgun Gothic',
       'Noto Sans KR', sans-serif;
       margin: 0; background: #f5f6f8; color: #1c1e21; }
.page { max-width: 860px; margin: 24px auto; background: #fff;
        border: 1px solid #e3e5e8; border-radius: 10px; padding: 28px 32px; }
h1 { font-size: 19px; margin: 0 0 2px; }
.sub { color: #667; font-size: 12px; margin-bottom: 18px; }
.badge { display: inline-block; font-size: 22px; font-weight: 700;
         padding: 8px 18px; border-radius: 8px; margin: 6px 0 14px; }
table.meta { border-collapse: collapse; font-size: 13px; margin-bottom: 20px; }
table.meta td { padding: 3px 14px 3px 0; color: #444; }
table.meta td.k { color: #888; white-space: nowrap; }
h2 { font-size: 14px; border-bottom: 1px solid #eceef1; padding-bottom: 5px;
     margin: 22px 0 10px; }
.attr { margin: 7px 0; font-size: 13px; }
.attr .name { display: inline-block; width: 220px; overflow: hidden;
              text-overflow: ellipsis; white-space: nowrap; vertical-align: middle; }
.bar-rail { display: inline-block; width: 420px; height: 13px; background: #eef0f3;
            border-radius: 4px; vertical-align: middle; position: relative; }
.bar { display: block; height: 100%; border-radius: 4px; background: #7a93b8; }
.bar.warn { background: #d9a13b; } .bar.crit { background: #c4443c; }
.tick { position: absolute; top: -2px; height: 17px; width: 1px; background: #999; }
.score { margin-left: 8px; color: #555; font-variant-numeric: tabular-nums; }
.evi { margin: 4px 0 14px; padding: 10px 12px; background: #fafbfc;
       border: 1px solid #eef0f3; border-radius: 6px; }
.evi h3 { font-size: 13px; margin: 0 0 8px; }
.row { display: flex; align-items: center; font-size: 12px; margin: 2px 0; }
.row .lbl { width: 170px; color: #555; overflow: hidden; text-overflow: ellipsis;
            white-space: nowrap; }
.row .cell { flex: 1; display: flex; align-items: center; gap: 6px; }
.mini { height: 10px; border-radius: 3px; }
.mini.ref { background: #9fb6d4; } .mini.cur { background: #e0876f; }
.pct { color: #777; font-size: 11px; min-width: 76px; font-variant-numeric: tabular-nums; }
.legend { font-size: 11px; color: #666; margin-bottom: 6px; }
.legend .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
              vertical-align: middle; margin: 0 3px 0 10px; }
.rx li { font-size: 13px; margin: 4px 0; }
.reason { background: #fdf6dd; border: 1px solid #eadfa8; border-radius: 6px;
          padding: 10px 12px; font-size: 13px; }
.foot { color: #99a; font-size: 11px; margin-top: 24px; border-top: 1px solid #eceef1;
        padding-top: 8px; }
.why li, .checks li { font-size: 13px; margin: 5px 0; line-height: 1.5; }
.chg { font-size: 13px; margin: 8px 0; }
.chg .nm { font-weight: 600; }
.chg .sum { color: #444; margin-left: 6px; }
.pat { display: inline-block; font-size: 11px; background: #fff4e5; color: #8a5300;
       border: 1px solid #f1d9b5; border-radius: 10px; padding: 1px 8px; margin: 3px 4px 0 0; }
.caution { background: #fdf6dd; border: 1px solid #eadfa8; border-radius: 6px;
           padding: 8px 12px; font-size: 13px; margin-bottom: 14px; }
.limits { font-size: 12px; color: #555; background: #f7f8fa; border-radius: 6px;
          padding: 8px 12px; }
.more { font-size: 11px; color: #888; margin-top: 4px; }
.delta { color: #999; font-size: 11px; min-width: 64px; font-variant-numeric: tabular-nums; }
"""


def _e(v: Any) -> str:
    return html.escape(str(v))


def _norm(dist: Dict[str, float]) -> Dict[str, float]:
    total = sum(dist.values()) or 1.0
    return {k: v / total for k, v in dist.items()}


def _attr_bar(name: str, score: float, threshold: float, attr_threshold: float) -> str:
    cls = "crit" if score >= attr_threshold else ("warn" if score >= threshold else "")
    width = max(0.0, min(1.0, score)) * 100
    t1 = max(0.0, min(1.0, threshold)) * 100
    t2 = max(0.0, min(1.0, attr_threshold)) * 100
    return (
        f'<div class="attr"><span class="name" title="{_e(name)}">{_e(name)}</span>'
        f'<span class="bar-rail"><span class="bar {cls}" style="width:{width:.1f}%"></span>'
        f'<span class="tick" style="left:{t1:.1f}%"></span>'
        f'<span class="tick" style="left:{t2:.1f}%"></span></span>'
        f'<span class="score">{score:.3f}</span></div>'
    )


_EVIDENCE_ROWS = 10


def _evidence_block(attr: str, dists: Dict[str, Dict[str, float]], lang: str,
                    keep_order: bool = False) -> str:
    """ref → cur rows for one attribute: rows with no mass on either side
    are dropped; categorical rows are ordered by size of change and capped.
    Summary statistics (mean / std / ...) are shown as values, not shares."""
    ref_raw, cur_raw = dists.get("ref", {}), dists.get("cur", {})
    if dx.is_stat_dict(ref_raw) or dx.is_stat_dict(cur_raw):
        rows = [
            f'<div class="row"><span class="lbl">{_e(k)}</span><span class="cell">'
            f'<span class="pct">{dx._num(ref_raw.get(k))} → {dx._num(cur_raw.get(k))}</span>'
            f'</span></div>'
            for k in dict.fromkeys([*ref_raw, *cur_raw])
        ]
        return f'<div class="evi"><h3>{_e(attr)}</h3>' + "".join(rows) + "</div>"

    ref_n, cur_n = _norm(ref_raw), _norm(cur_raw)
    labels = [k for k in dict.fromkeys([*ref_raw, *cur_raw])
              if max(ref_n.get(k, 0.0), cur_n.get(k, 0.0)) >= 0.0005]
    if not labels:
        return ""
    hidden = 0
    if not keep_order:
        labels.sort(key=lambda k: abs(cur_n.get(k, 0.0) - ref_n.get(k, 0.0)), reverse=True)
        if len(labels) > _EVIDENCE_ROWS:
            hidden = len(labels) - _EVIDENCE_ROWS
            labels = labels[:_EVIDENCE_ROWS]
    peak = max([*ref_n.values(), *cur_n.values(), 1e-9])
    legend = (
        f'<div class="legend">{_e(t("note.legend_ref", lang))}'
        f'<span class="sw" style="background:#9fb6d4"></span>ref &nbsp;·&nbsp; '
        f'{_e(t("note.legend_cur", lang))}'
        f'<span class="sw" style="background:#e0876f"></span>cur</div>'
    )
    rows = []
    for lbl in labels:
        r, c = ref_n.get(lbl, 0.0), cur_n.get(lbl, 0.0)
        rows.append(
            f'<div class="row"><span class="lbl" title="{_e(lbl)}">{_e(lbl)}</span>'
            f'<span class="cell"><span class="mini ref" style="width:{r / peak * 100:.1f}%"></span>'
            f'<span class="mini cur" style="width:{c / peak * 100:.1f}%"></span>'
            f'<span class="pct">{r*100:.1f}% → {c*100:.1f}%</span>'
            f'<span class="delta">{_e(t("fmt.pp", lang, value=(c - r) * 100))}</span></span></div>'
        )
    more = (f'<div class="more">{_e(t("note.more_rows", lang, count=hidden))}</div>'
            if hidden else "")
    return f'<div class="evi"><h3>{_e(attr)}</h3>' + legend + "".join(rows) + more + "</div>"


def build_consultation_html(
    payload: Dict[str, Any],
    distributions: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    *,
    max_evidence: int = 5,
    lang: str = DEFAULT_LANG,
) -> str:
    """Render the verdict payload (+ optional per-attribute ref/cur
    distributions) as a standalone consultation-note HTML page.

    Sections: verdict, cautions, why this verdict, attribute scores, what
    changed, evidence, what to check next, and what the note cannot tell.
    """
    verdict = payload.get("verdict", "BLIND")
    color, bg, icon = _VERDICT_STYLE.get(verdict, _VERDICT_STYLE["BLIND"])
    severity = payload.get("severity", "unknown")
    score = payload.get("overall_score")
    threshold = payload.get("threshold") or 0.15
    attr_threshold = payload.get("attr_threshold") or 0.25
    top: List[Tuple[str, float]] = [tuple(t) for t in payload.get("top_attributes") or []]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    raw = payload.get("raw") or {}
    diagnosis = payload.get("diagnosis") or dx.build(
        verdict=verdict, overall=score, top=top, threshold=threshold,
        attr_threshold=attr_threshold, result=raw, evidence=distributions,
        reason=payload.get("reason"),
    )

    meta_rows = [
        (t("note.ref", lang), payload.get("ref")),
        (t("note.cur", lang), payload.get("cur")),
        (t("note.modality", lang), payload.get("modality")),
        (t("note.engine", lang), f"{payload.get('engine')} / {payload.get('detector')}"),
        (t("note.overall", lang), "—" if score is None else f"{score:.4f}"),
        (t("note.thresholds", lang),
         t("note.thresholds_value", lang, threshold=threshold, attr_threshold=attr_threshold)),
        (t("label.severity", lang), severity_label(severity, lang)),
    ]
    samples = diagnosis.get("samples")
    if samples:
        meta_rows.append((t("note.samples", lang), t("note.samples_value", lang, **samples)))
    meta_html = "".join(
        f'<tr><td class="k">{_e(k)}</td><td>{_e(v)}</td></tr>' for k, v in meta_rows
    )

    parts = [
        f"<!doctype html><html lang='{_e(lang)}'><head><meta charset='utf-8'>",
        f"<title>ddoc diff — {_e(verdict)}</title><style>{_CSS}</style></head><body>",
        "<div class='page'>",
        f"<h1>{_e(t('note.title', lang))}</h1>",
        f"<div class='sub'>{t('note.generated', lang, time=_e(generated))}</div>",
        f"<div class='badge' style='color:{color};background:{bg}'>{icon} {_e(verdict)}"
        f"<span style='font-size:13px;font-weight:400'> &nbsp;{_e(t('label.severity', lang))}: "
        f"{_e(severity_label(severity, lang))}</span></div>",
        f"<table class='meta'>{meta_html}</table>",
    ]

    warnings = diagnosis.get("warnings") or []
    if warnings:
        parts.append(
            f"<div class='caution'><b>{_e(t('note.caution', lang))}</b> — "
            + " ".join(_e(dx.warning_text(w, lang)) for w in warnings) + "</div>"
        )

    if verdict == "BLIND":
        parts.append(f"<h2>{_e(t('note.reason', lang))}</h2>")
        parts.append(f"<div class='reason'>{_e(payload.get('reason') or '')}</div>")
        hints = payload.get("hints") or []
        if hints:
            parts.append(f"<h2>{_e(t('note.checks', lang))}</h2><ul class='checks'>")
            parts += [f"<li>{_e(h)}</li>" for h in hints]
            parts.append("</ul>")
    else:
        parts.append(f"<h2>{_e(t('note.why', lang))}</h2><ul class='why'>")
        parts += [f"<li>{_e(dx.rationale_text(r, lang))}</li>"
                  for r in diagnosis.get("rationale") or []]
        parts.append("</ul>")

        scores = raw.get("attribute_drifts") or dict(top)
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if ordered:
            parts.append(f"<h2>{_e(t('note.findings', lang))}</h2>")
            parts += [_attr_bar(k, float(v), threshold, attr_threshold) for k, v in ordered]

        described = [a for a in diagnosis.get("attributes") or []
                     if dx.summary_text(a.get("summary"), lang) or a.get("patterns")]
        if described:
            parts.append(f"<h2>{_e(t('note.changes', lang))}</h2>")
            for a in described:
                line = dx.summary_text(a.get("summary"), lang)
                # The summary line already states the largest share change
                # and the mean, so those patterns are not repeated as badges.
                badges = "".join(f"<span class='pat'>{_e(dx.pattern_text(p, lang))}</span>"
                                 for p in a.get("patterns") or []
                                 if p["code"] not in ("share_shift", "mean_shift"))
                parts.append(
                    f"<div class='chg'><span class='nm'>{_e(a['name'])}</span>"
                    + (f"<span class='sum'>{_e(line)}</span>" if line else "")
                    + (f"<div>{badges}</div>" if badges else "") + "</div>"
                )

        dists = distributions or {}
        kinds = {a["name"]: (a.get("summary") or {}).get("kind")
                 for a in diagnosis.get("attributes") or []}
        blocks = [_evidence_block(k, dists[k], lang, keep_order=kinds.get(k) == "numeric")
                  for k, _ in ordered if k in dists][:max_evidence]
        blocks = [b for b in blocks if b]
        if blocks:
            parts.append(f"<h2>{_e(t('note.evidence', lang))}</h2>")
            parts += blocks

        checks = dx.check_texts(diagnosis, lang)
        if checks:
            parts.append(f"<h2>{_e(t('note.checks', lang))}</h2><ul class='checks'>")
            parts += [f"<li>{_e(c)}</li>" for c in checks]
            parts.append("</ul>")

        parts.append(f"<h2>{_e(t('note.limits', lang))}</h2>"
                     f"<div class='limits'>{_e(t('note.limits_text', lang))}</div>")

    parts.append(f"<div class='foot'>{_e(t('note.footer', lang))}</div>")
    parts.append("</div></body></html>")
    return "".join(parts)

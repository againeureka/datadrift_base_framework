"""Data profile report — one dataset, in the diagnostic report's format.

Renders an EDA envelope as a document with the same masthead, toolbar,
diagram and cards as ``ddoc diff --report``. The built-in CSV profile
(:mod:`ddoc.profile`) gets the full layout: subject, summary tiles, a
diagram of the columns, quality findings, a column table and the
distribution of each flagged column. Envelopes from EDA plugins carry
only a summary; they get the same shell around it.

Single-dataset charts use the accent colour only; there is no baseline.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ... import profile as pf
from ...i18n import DEFAULT_LANG, t
from .diff_report import (_CSS, _INSP_STATUS, _SCRIPT, _THEME_BOOT, _doc_number, _ddoc_version, _e,
                          _histogram, _ico, _int, _norm, _pct, _spark, _table_view, _toolbar,
                          _value_columns, anchor_id)

_MAX_PODS = 6


def _shell_head(title: str, lang: str, issuer: Optional[str], doc_no: str, issued: datetime) -> str:
    return (f"<!doctype html><html lang='{_e(lang)}'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{_e(title)}</title><style>{_CSS}</style><script>{_THEME_BOOT}</script></head>"
            f"<body><div class='doc'>{_toolbar(lang)}<div class='sheet'>"
            f"<header class='mast'><div><div class='issuer'>{_e(issuer or t('doc.issuer_default', lang))}</div>"
            f"<h1>{_e(title)}</h1></div>"
            f"<div class='docmeta'>{_e(t('doc.number', lang))} <b>{_e(doc_no)}</b><br>"
            f"{_e(t('doc.issued', lang))} <b>{issued:%Y-%m-%d %H:%M} UTC</b><br>"
            f"{_e(t('doc.tool', lang))} <b>ddoc {_e(_ddoc_version())}</b></div></header>")


def _shell_foot(lang: str, doc_no: str) -> str:
    return (f"<footer class='foot'><span>{_e(t('doc.footer', lang, version=_ddoc_version()))}</span>"
            f"<span>{_e(doc_no)}</span></footer>"
            f"</div></div><div id='tip' role='tooltip'></div><script>{_SCRIPT}</script></body></html>")


def _info(rows: List[tuple]) -> str:
    return "<table class='info'>" + "".join(
        f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in rows) + "</table>"


def _flagged(findings: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Column → the findings that name it."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for f in findings:
        for attr in f.get("attrs") or []:
            out.setdefault(attr, []).append(f)
    return out


def _single(dist: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {"ref": {}, "cur": dist or {}}


def _bars(dist: Dict[str, float], lang: str, max_rows: int = 10) -> str:
    """Horizontal share bars for one categorical column."""
    shares = _norm(dist)
    labels = sorted(shares, key=lambda k: (k == "(other)", -shares[k]))[:max_rows]
    if not labels:
        return ""
    W, label_w, right_w, row, top = 720, 160, 70, 26, 6
    H = top + row * len(labels)
    peak = max(shares[k] for k in labels) or 1.0
    out = [f"<svg viewBox='0 0 {W} {H}' role='img'>"]
    for i, k in enumerate(labels):
        y = top + i * row
        w = (W - label_w - right_w) * shares[k] / peak
        label = k if len(k) <= 20 else k[:19] + "…"
        out.append(f"<text x='{label_w - 12}' y='{y + row / 2 + 4:.1f}' text-anchor='end'>{_e(label)}</text>")
        if w > 0:
            r = min(4.0, w, 7.0)
            x0, x1, y0, h = label_w, label_w + w, y + 5, row - 10
            out.append(f"<path d='M{x0:.1f},{y0:.1f}H{x1 - r:.1f}Q{x1:.1f},{y0:.1f} {x1:.1f},{y0 + r:.1f}"
                       f"V{y0 + h - r:.1f}Q{x1:.1f},{y0 + h:.1f} {x1 - r:.1f},{y0 + h:.1f}H{x0:.1f}Z' fill='var(--cur)'/>")
        out.append(f"<text x='{label_w + w + 8:.1f}' y='{y + row / 2 + 4:.1f}' class='val-lbl'>{_pct(shares[k])}</text>")
        out.append(f"<rect class='hit' x='0' y='{y}' width='{W}' height='{row}' tabindex='0' "
                   f"data-tip='{_e(k)}&#10;{_pct(shares[k])}'><title>{_e(k)}: {_pct(shares[k])}</title></rect>")
    out.append("</svg>")
    return "".join(out)


def _pod(name: str, c: Dict[str, Any], found: List[Dict[str, Any]], lang: str) -> str:
    kind = c.get("type")
    head = (f"<div class='hd'><h3>{_e(name)}</h3><span class='kind'>{_e(t('kind.' + kind, lang))}"
            f" · {_e(t('col.missing', lang))} {_pct(c.get('missing'))}</span></div>")
    body = []
    chips = [line for f in found for line in pf.finding_text(
        {**f, "signals": [s for s in f.get("signals") or [] if s.get("attr") == name]}, lang)["signals"]]
    if chips:
        body.append("<div class='chips'>" + "".join(f"<span class='chip'>{_e(x)}</span>" for x in chips) + "</div>")
    dist = c.get("distribution") or {}
    if kind == "numeric" and c.get("discrete"):
        svg, table = _value_columns(_single(dist), lang)
        body.append(svg + _table_view(lang, [t("col.value", lang), t("chart.share", lang)],
                                      [[r[0], r[2]] for r in table], (1,)))
    elif kind == "numeric":
        svg, table = _histogram(_single(dist), {"median": {"cur": c.get("median")}}, lang)
        if c.get("outside_bins"):
            svg += f"<div class='legend'>{_e(t('profile.outside_bins', lang, share=_pct(c['outside_bins'])))}</div>"
        body.append(svg + _table_view(lang, [t("col.range", lang), t("chart.share", lang)],
                                      [[r[0], r[2]] for r in table], (1,)))
    elif kind == "categorical":
        shares = _norm(dist)
        body.append(_bars(dist, lang) + _table_view(
            lang, [t("col.value", lang), t("chart.share", lang)],
            [[k, _pct(v)] for k, v in sorted(shares.items(), key=lambda kv: -kv[1])], (1,)))
    if not body:
        return ""
    return f"<section class='pod' id='{anchor_id(name)}'>{head}<div class='bd'>{''.join(body)}</div></section>"


def _diagram(envelope: Dict[str, Any], flagged: Dict[str, List[Dict[str, Any]]],
             linked: set, lang: str) -> str:
    cells, numbered = [], []
    for name, c in (envelope.get("columns") or {}).items():
        kind = c.get("type")
        off = kind in ("datetime", "empty")
        st = "caution" if name in flagged else "pass"
        if not off and st == "caution":
            numbered.append(name)
        no = f"<span class='no'>{len(numbered)}</span>" if (not off and st == "caution") else ""
        if off:
            state = f"<span class='st'>{_e(t('diagram.time', lang) if kind == 'datetime' else t('kind.empty', lang))}</span>"
            bar = "<div class='bar-s'></div>"
        else:
            token, icon = _INSP_STATUS[st]
            state = f"<span class='st status'>{_ico(token, icon, small=True)}{_e(t('insp.status.' + st, lang))}</span>"
            bar = f"<div class='bar-s' style='background:var(--{token})'></div>" if st != "pass" else "<div class='bar-s'></div>"
        summary = {"kind": kind, "discrete": c.get("discrete")}
        spark = "" if off else _spark(_single(c.get("distribution")), summary)
        tip = f"{name}\n{t('kind.' + kind, lang)} · {t('col.missing', lang)} {_pct(c.get('missing'))}"
        inner = (f"{bar}<div class='nm'>{_e(name)}</div><div class='ty'>{_e(t('kind.' + kind, lang))}&nbsp;</div>"
                 f"{spark}{state}{no}")
        cls = f"part {'off' if off else 's-' + st}{' numbered' if no else ''}"
        if name in linked:
            cells.append(f"<a class='{cls}' href='#{anchor_id(name)}' data-col='{_e(name)}' "
                         f"data-tip='{_e(tip)}' aria-label='{_e(tip)}'>{inner}</a>")
        else:
            cells.append(f"<div class='{cls}' data-col='{_e(name)}' data-tip='{_e(tip)}' tabindex='0' "
                         f"aria-label='{_e(tip)}'>{inner}</div>")
    out = [f"<p class='dg-lead'>{_e(t('profile.diagram_lead', lang))}</p><div class='dg'>{''.join(cells)}</div>"]
    if numbered:
        items = []
        for i, name in enumerate(numbered, 1):
            lines = [line for f in flagged[name] for line in pf.finding_text(
                {**f, "signals": [s for s in f.get("signals") or [] if s.get("attr") == name]}, lang)["signals"]]
            titles = " · ".join(pf.finding_text(f, lang)["title"] for f in flagged[name])
            title = f"<a href='#{anchor_id(name)}'>{_e(name)}</a>" if name in linked else f"<b>{_e(name)}</b>"
            items.append(f"<li data-col='{_e(name)}'><span class='no'>{i}</span><div>{title} "
                         f"<span class='tg'>{_e(titles)}</span><div class='fd'>{_e(' · '.join(lines))}</div></div></li>")
        out.append(f"<ol class='callouts'>{''.join(items)}</ol>")
    return "".join(out)


def _cards(findings: List[Dict[str, Any]], linked: set, lang: str) -> str:
    if not findings:
        return f"<p>{_e(t('profile.none', lang))}</p>"
    out = [f"<p class='dg-lead'>{_e(t('profile.findings_lead', lang))}</p><div class='cards'>"]
    for f in findings:
        x = pf.finding_text(f, lang)
        names = f.get("attrs") or []
        tg = ", ".join(f"<a href='#{anchor_id(n)}'>{_e(n)}</a>" if n in linked else _e(n) for n in names)
        out.append(
            f"<article class='card'><header><h3>{_e(x['title'])}</h3>"
            + (f"<span class='tg'>{_e(t('reading.targets', lang))}: {tg}</span>" if tg else "") + "</header>"
            "<ul class='sig'>" + "".join(f"<li>{_e(s)}</li>" for s in x["signals"]) + "</ul>"
            f"<p class='mean'>{_e(x['meaning'])}</p>"
            f"<div class='two one'><div><h4>{_e(t('reading.actions', lang))}</h4><ul class='todo'>"
            + "".join(f"<li>{_e(a)}</li>" for a in x["actions"]) + "</ul></div></div>"
            f"<p class='crit-s'>{_e(t('reading.criterion', lang))}: {_e(x['criterion'])}</p></article>")
    out.append("</div>")
    return "".join(out)


def _columns_table(envelope: Dict[str, Any], lang: str) -> str:
    head = ["col.attribute", "col.type", "col.missing", "col.distinct", "col.mean", "col.std",
            "col.min", "col.median", "col.max", "col.top"]
    rows = []
    for name, c in (envelope.get("columns") or {}).items():
        num = lambda k: "—" if c.get(k) is None else f"{c[k]:.4g}"  # noqa: E731
        top = (f"{c['top_value']} ({_pct(c['top_share'])})" if c.get("top_value") is not None else "—")
        if c.get("type") == "datetime" and c.get("period"):
            top = f"{c['period']['start']} ~ {c['period']['end']}"
        rows.append([name, t("kind." + c.get("type", "empty"), lang), _pct(c.get("missing")),
                     _int(c.get("distinct")) if c.get("distinct") is not None else "—",
                     num("mean"), num("std"), num("min"), num("median"), num("max"), top])
    th = "".join(f"<th class='{'num' if 1 < i < 9 else ''}'>{_e(t(h, lang))}</th>" for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f"<td class='{'nw' if 1 < i < 9 else ''}'>{_e(v)}</td>"
                                    for i, v in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def _tile(label: str, value: str, sub: str = "") -> str:
    return (f"<div class='tile'><div class='lbl'>{_e(label)}</div><div class='val'>{_e(value)}</div>"
            + (f"<div class='sub'>{_e(sub)}</div>" if sub else "") + "</div>")


def build_profile_html(envelope: Dict[str, Any], *, lang: str = DEFAULT_LANG,
                       issuer: Optional[str] = None, title: Optional[str] = None) -> str:
    issued = datetime.now(timezone.utc)
    doc_no = _doc_number(envelope, issued)
    doc_title = title or t("profile.title", lang)
    if not pf.is_profile(envelope):
        return _plugin_html(envelope, lang=lang, issuer=issuer, title=doc_title, doc_no=doc_no, issued=issued)

    prof = envelope.get("profile") or {}
    columns = envelope.get("columns") or {}
    findings = envelope.get("findings") or []
    flagged = _flagged(findings)
    pod_names = list(dict.fromkeys([n for n in columns if n in flagged] +
                                   [n for n, c in columns.items() if c.get("type") in ("numeric", "categorical")]))
    pods = [(n, _pod(n, columns[n], flagged.get(n, []), lang)) for n in pod_names[:_MAX_PODS]]
    pods = [(n, h) for n, h in pods if h]
    linked = {n for n, _ in pods}

    parts = [_shell_head(doc_title, lang, issuer, doc_no, issued),
             _info([(t("info.modality", lang), envelope.get("modality") or "—"),
                    (t("profile.method_label", lang), t("profile.method", lang))])]
    n = 0

    def section(key: str) -> str:
        nonlocal n
        n += 1
        return f"<h2>{n}. {_e(t(key, lang))}</h2>"

    # 1. Subject.
    parts.append(section("profile.sec_subject"))
    parts.append(f"<p class='lead'>{_e(t('profile.lead', lang))}</p>")
    rows = [(t("subject.path", lang), envelope.get("data_path") or "—"),
            (t("subject.size", lang), t("subject.size_value", lang, rows=prof.get("rows", 0),
                                         cols=len(prof.get("columns") or []))
             + (t("subject.files", lang, files=prof["files"]) if (prof.get("files") or 1) > 1 else ""))]
    if prof.get("period"):
        rows.append((t("subject.period", lang), t("subject.period_value", lang, **prof["period"])))
    rows.append((t("profile.duplicates", lang), _pct(envelope.get("duplicate_rows"))))
    parts.append(_info(rows))
    chips = "".join(f"<span class='chip'>{_e(c['name'])}<small>{_e(t('kind.' + c['type'], lang))}</small></span>"
                    for c in prof.get("columns") or [])
    parts.append(f"<div class='subj chips'>{chips}</div>")
    sample = prof.get("sample")
    if sample:
        cols_h = "".join(f"<th>{_e(c)}</th>" for c in sample["columns"])
        body = "".join("<tr>" + "".join(f"<td>{_e(v)}</td>" for v in r) + "</tr>" for r in sample["rows"])
        parts.append(f"<div class='sample'><div class='caution'>{_e(t('subject.sample_note', lang))}</div>"
                     f"<div class='wrap'><table><thead><tr>{cols_h}</tr></thead><tbody>{body}</tbody></table></div></div>")

    # 2. Summary.
    parts.append(section("sec.summary"))
    titles = [pf.finding_text(f, lang)["title"] for f in findings]
    if titles:
        parts.append(f"<p class='headline'><span class='k'>{_e(t('profile.headline', lang))}</span>"
                     f"{_e(' · '.join(titles))}</p>")
    parts.append("<div class='tiles'>"
                 + _tile(t("tile.rows", lang), _int(prof.get("rows")))
                 + _tile(t("tile.cols", lang), _int(len(columns)))
                 + _tile(t("tile.missing_cells", lang), _pct(envelope.get("missing_cells")))
                 + _tile(t("tile.findings", lang), _int(len(findings)),
                         ", ".join(titles[:3]) if titles else t("profile.none", lang))
                 + "</div>")

    # 3. Diagram, 4. findings, 5. columns, 6. distributions, 7. limits.
    parts.append(section("sec.diagram"))
    parts.append(_diagram(envelope, flagged, linked, lang))
    parts.append(section("sec.findings"))
    parts.append(_cards(findings, linked, lang))
    parts.append(section("sec.columns"))
    parts.append(_columns_table(envelope, lang))
    if pods:
        parts.append(section("sec.distributions"))
        parts += [h for _, h in pods]
    parts.append(section("sec.limits"))
    parts.append(f"<p class='limits'>{_e(t('profile.limits', lang))}</p>")
    parts.append(_shell_foot(lang, doc_no))
    return "".join(parts)


def _plugin_html(envelope: Dict[str, Any], *, lang: str, issuer: Optional[str], title: str,
                 doc_no: str, issued: datetime) -> str:
    """An EDA plugin's envelope: its summary inside the same document."""
    plugin = envelope.get("modality") or "?"
    rows = [(t("info.modality", lang), plugin)]
    for key in ("files_analyzed", "series_analyzed", "texts_analyzed"):
        if envelope.get(key) is not None:
            rows.append((key, _int(envelope[key])))
    if envelope.get("metrics_file"):
        rows.append((t("profile.metrics_file", lang), envelope["metrics_file"]))
    summary = envelope.get("summary") if isinstance(envelope.get("summary"), dict) else {}
    body = "".join(
        f"<tr><td><code>{_e(k)}</code></td><td>{_e(json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)}</td></tr>"
        for k, v in summary.items())
    raw = json.dumps(envelope, indent=2, ensure_ascii=False, default=str)
    return "".join([
        _shell_head(title, lang, issuer, doc_no, issued), _info(rows),
        f"<h2>1. {_e(t('sec.plugin_summary', lang))}</h2>",
        f"<p class='lead'>{_e(t('profile.plugin_lead', lang, plugin=plugin))}</p>",
        f"<table><thead><tr><th>{_e(t('col.key', lang))}</th><th>{_e(t('col.value', lang))}</th></tr></thead>"
        f"<tbody>{body}</tbody></table>" if body else "",
        f"<details class='tv'><summary>{_e(t('report.raw_envelope', lang))}</summary><pre>{_e(raw)}</pre></details>",
        _shell_foot(lang, doc_no)])


def build_profile_markdown(envelope: Dict[str, Any], *, lang: str = DEFAULT_LANG,
                           title: Optional[str] = None) -> str:
    lines = [f"# {title or t('profile.title', lang)}", ""]
    if not pf.is_profile(envelope):
        lines += [f"- {t('info.modality', lang)}: `{envelope.get('modality', '?')}`", ""]
        for k, v in (envelope.get("summary") or {}).items():
            lines.append(f"- `{k}`: {v}")
        return "\n".join(lines) + "\n"
    prof = envelope.get("profile") or {}
    lines += [f"- {t('subject.path', lang)}: `{envelope.get('data_path')}`",
              f"- {t('subject.size', lang)}: " + t("subject.size_value", lang, rows=prof.get("rows", 0),
                                                   cols=len(prof.get("columns") or []))]
    if prof.get("period"):
        lines.append(f"- {t('subject.period', lang)}: " + t("subject.period_value", lang, **prof["period"]))
    lines.append(f"- {t('profile.duplicates', lang)}: {_pct(envelope.get('duplicate_rows'))}")
    lines += ["", f"## {t('sec.findings', lang)}", ""]
    findings = envelope.get("findings") or []
    if not findings:
        lines.append(t("profile.none", lang))
    for f in findings:
        x = pf.finding_text(f, lang)
        lines += [f"### {x['title']}"] + [f"- {s}" for s in x["signals"]]
        lines += ["", x["meaning"], ""] + [f"- [ ] {a}" for a in x["actions"]] + [""]
    head = [t(k, lang) for k in ("col.attribute", "col.type", "col.missing", "col.distinct",
                                 "col.mean", "col.median", "col.top")]
    lines += [f"## {t('sec.columns', lang)}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, c in (envelope.get("columns") or {}).items():
        num = lambda k: "—" if c.get(k) is None else f"{c[k]:.4g}"  # noqa: E731
        top = f"{c['top_value']} ({_pct(c['top_share'])})" if c.get("top_value") is not None else "—"
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in (
            name, t("kind." + c.get("type", "empty"), lang), _pct(c.get("missing")),
            c.get("distinct", "—"), num("mean"), num("median"), top)) + " |")
    return "\n".join(lines) + "\n"

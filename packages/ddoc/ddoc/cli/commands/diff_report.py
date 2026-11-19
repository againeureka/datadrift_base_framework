"""Diagnostic report renderer for ``ddoc diff --report``.

Turns a diff verdict payload into a self-contained HTML document laid out
like an inspection report: a masthead with issuer, document number and
issue time; a subject / method / result table; then numbered sections —
summary tiles, the basis for the verdict, results by attribute, findings
by attribute (one panel per attribute with its chart), characteristics of
the data, items to check, and limitations.

Charts are inline SVG built here (no plotting library). A short inline
script adds hover / focus tooltips; every value a tooltip shows is also in
the chart labels or the "show as table" view, so the document reads fully
without it, offline, and in print.

Colour roles: the current data is the accent (blue); the baseline is
context (gray). Status colours appear only on the verdict, always with an
icon and a label.
"""
from __future__ import annotations

import hashlib
import html
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ... import diagnosis as dx
from ...i18n import DEFAULT_LANG, severity_label, t

# ── Style ─────────────────────────────────────────────────────────────

_CSS = """
.doc {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --cur: #2a78d6; --ref: #898781; --track: #cde2fb; --link: #c3c2b7;
  --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
  --chip: #f0efec; --tip-bg: #0b0b0b; --tip-ink: #f0efec;
}
/* Night theme: follows the OS unless the reader picks one; screen only,
   so a printed or saved PDF always comes out on white paper. */
@media screen and (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .doc {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --cur: #3987e5; --ref: #75746e; --track: #184f95; --link: #383835;
    --chip: #2c2c2a; --tip-bg: #f0efec; --tip-ink: #0b0b0b;
  }
}
@media screen {
  :root[data-theme="dark"] .doc {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --cur: #3987e5; --ref: #75746e; --track: #184f95; --link: #383835;
    --chip: #2c2c2a; --tip-bg: #f0efec; --tip-ink: #0b0b0b;
  }
}
html, body { margin: 0; }
body { background: var(--page); }
.doc { font-family: system-ui, -apple-system, "Segoe UI", "Apple SD Gothic Neo",
       "Malgun Gothic", "Noto Sans KR", sans-serif;
       font-size: 15px; line-height: 1.6; color: var(--ink); background: var(--page);
       padding: 28px 16px; }
.sheet { max-width: 920px; margin: 0 auto; background: var(--surface);
         border: 1px solid var(--border); border-radius: 6px; padding: 40px 48px 28px; }
.mast { display: flex; justify-content: space-between; align-items: flex-end; gap: 24px;
        border-bottom: 2px solid var(--ink); padding-bottom: 14px; }
.issuer { font-size: 13px; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-2);
          font-weight: 600; }
h1 { font-size: 28px; line-height: 1.25; margin: 6px 0 0; font-weight: 700; }
.docmeta { font-size: 13px; color: var(--ink-2); text-align: right; white-space: nowrap; }
.docmeta b { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; vertical-align: top; padding: 8px 10px; border-bottom: 1px solid var(--grid); }
th { color: var(--ink-2); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.info { margin: 18px 0 8px; }
.info th { width: 150px; background: var(--chip); border-bottom: 1px solid var(--surface); }
.info td { word-break: break-all; }
h2 { font-size: 19px; margin: 34px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--grid); }
h3 { font-size: 16px; margin: 0; }
.tiles { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.tile { border: 1px solid var(--border); border-radius: 6px; padding: 14px 16px; }
.tile .lbl { font-size: 13px; color: var(--ink-2); }
.tile .val { font-size: 28px; font-weight: 600; line-height: 1.2; margin-top: 4px; }
.tile .sub { font-size: 13px; color: var(--ink-2); margin-top: 6px; }
.status { display: inline-flex; align-items: center; gap: 8px; }
.status .ico { width: 22px; height: 22px; border-radius: 50%; display: inline-flex;
               align-items: center; justify-content: center; font-size: 13px; font-weight: 700;
               color: #ffffff; }
.status .ico.warning { color: #0b0b0b; }
.meter { position: relative; height: 8px; border-radius: 4px; background: var(--track); margin-top: 10px; }
.meter .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px; }
.meter .tick { position: absolute; top: -4px; width: 2px; height: 16px; background: var(--ink-2); }
.caution { border: 1px solid var(--warning); border-left-width: 4px; border-radius: 6px;
           padding: 10px 14px; margin-top: 14px; font-size: 14px; }
.why li, .plain li { margin: 6px 0; }
.pod { border: 1px solid var(--border); border-radius: 6px; margin: 14px 0; break-inside: avoid; }
.pod > .hd { display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
             padding: 10px 16px; border-bottom: 1px solid var(--grid); background: var(--chip);
             border-radius: 6px 6px 0 0; }
.pod > .hd .kind { font-size: 13px; color: var(--ink-2); }
.pod > .bd { padding: 12px 16px 14px; }
.summary { margin: 0 0 6px; }
.chips { margin: 4px 0 8px; }
.chip { display: inline-block; font-size: 13px; background: var(--chip); color: var(--ink);
        border-radius: 12px; padding: 2px 10px; margin: 2px 6px 2px 0; }
.legend { font-size: 13px; color: var(--ink-2); margin: 6px 0 2px; }
.legend .key { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
               vertical-align: middle; margin: 0 6px 0 14px; }
.legend .key:first-child { margin-left: 0; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { font-family: inherit; fill: var(--ink-2); font-size: 13px; }
svg .tick-lbl { fill: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
svg .val-lbl { fill: var(--ink); font-size: 13px; font-variant-numeric: tabular-nums; }
svg .halo { paint-order: stroke; stroke: var(--surface); stroke-width: 4px; stroke-linejoin: round; }
td.nw { white-space: nowrap; font-variant-numeric: tabular-nums; }
svg .hit { fill: transparent; cursor: default; outline: none; }
svg .hit:hover, svg .hit:focus { fill: var(--ink); fill-opacity: .04; }
details.tv { margin-top: 8px; font-size: 14px; }
details.tv summary { cursor: pointer; color: var(--ink-2); font-size: 13px; }
details.tv table { margin-top: 6px; }
.checks td.done { width: 56px; text-align: center; color: var(--muted); }
.limits { color: var(--ink-2); }
.foot { margin-top: 32px; padding-top: 10px; border-top: 1px solid var(--grid);
        font-size: 12px; color: var(--muted); display: flex; justify-content: space-between; gap: 16px; }
.bar { max-width: 920px; margin: 0 auto 10px; display: flex; justify-content: flex-end;
       align-items: center; gap: 10px; font-size: 13px; color: var(--ink-2); }
.bar[hidden] { display: none; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.bar button { font: inherit; color: var(--ink); background: var(--surface); border: 0;
              padding: 5px 12px; cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.bar button[aria-pressed="true"] { background: var(--ink); color: var(--surface); }
.bar .print { border: 1px solid var(--border); border-radius: 6px; }
.bar button:hover { background: var(--chip); }
.bar button[aria-pressed="true"]:hover { background: var(--ink); }
.bar button:focus-visible { outline: 2px solid var(--cur); outline-offset: -2px; }
#tip { position: fixed; pointer-events: none; z-index: 10; display: none; max-width: 320px;
       background: var(--tip-bg); color: var(--tip-ink); font-size: 13px; line-height: 1.45;
       padding: 6px 10px; border-radius: 6px; white-space: pre-line; }
@media (max-width: 720px) {
  .sheet { padding: 24px 18px; }
  .tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .mast { flex-direction: column; align-items: flex-start; }
  .docmeta { text-align: left; }
}
@page { size: A4; margin: 14mm; }
@media print {
  .doc { background: #ffffff; padding: 0; font-size: 13px;
         -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .bar { display: none !important; }
  .sheet { border: none; padding: 0; max-width: none; background: #ffffff; }
  .tile { padding: 10px 12px; }
  .tile .val { font-size: 21px; white-space: nowrap; }
  .tile .sub { font-size: 11px; }
  details.tv, #tip { display: none; }
  h2 { break-after: avoid; }
}
"""

# Runs in <head>: applies a saved theme before first paint.
_THEME_BOOT = """
try { var t = localStorage.getItem('ddoc-theme');
      if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t);
} catch (e) {}
"""

_SCRIPT = """
(function () {
  var root = document.documentElement, bar = document.querySelector('.bar');
  if (bar) {
    bar.hidden = false;
    var btns = bar.querySelectorAll('[data-set-theme]');
    function apply(mode) {
      if (mode === 'light' || mode === 'dark') root.setAttribute('data-theme', mode);
      else root.removeAttribute('data-theme');
      btns.forEach(function (b) {
        b.setAttribute('aria-pressed', String(b.getAttribute('data-set-theme') === mode));
      });
      try { if (mode === 'auto') localStorage.removeItem('ddoc-theme');
            else localStorage.setItem('ddoc-theme', mode); } catch (e) {}
    }
    apply(root.getAttribute('data-theme') || 'auto');
    btns.forEach(function (b) {
      b.addEventListener('click', function () { apply(b.getAttribute('data-set-theme')); });
    });
    var p = bar.querySelector('.print');
    if (p) p.addEventListener('click', function () { window.print(); });
  }
})();
(function () {
  var tip = document.getElementById('tip');
  if (!tip) return;
  function show(el, x, y) {
    tip.textContent = el.getAttribute('data-tip');
    tip.style.display = 'block';
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var left = Math.min(x + 14, window.innerWidth - w - 8);
    var top = y - h - 12 < 0 ? y + 16 : y - h - 12;
    tip.style.left = left + 'px'; tip.style.top = top + 'px';
  }
  function hide() { tip.style.display = 'none'; }
  document.querySelectorAll('[data-tip]').forEach(function (el) {
    el.addEventListener('pointermove', function (e) { show(el, e.clientX, e.clientY); });
    el.addEventListener('pointerleave', hide);
    el.addEventListener('focus', function () {
      var r = el.getBoundingClientRect(); show(el, r.left + r.width / 2, r.top);
    });
    el.addEventListener('blur', hide);
  });
})();
"""

_STATUS = {  # verdict/severity → (status token, icon)
    ("OK", None): ("good", "✓"),
    ("DRIFT", "medium"): ("serious", "!"),
    ("DRIFT", "high"): ("critical", "!"),
    ("BLIND", None): ("warning", "?"),
}

# ── Small helpers ─────────────────────────────────────────────────────


def _e(v: Any) -> str:
    return html.escape(str(v), quote=True)


def _norm(dist: Dict[str, float]) -> Dict[str, float]:
    total = sum(dist.values()) or 1.0
    return {k: v / total for k, v in dist.items()}


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _int(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "—"


def _status(verdict: str, severity: str) -> Tuple[str, str]:
    return _STATUS.get((verdict, severity if verdict == "DRIFT" else None),
                       _STATUS[("DRIFT", "high")] if verdict == "DRIFT" else _STATUS[("BLIND", None)])


def _doc_number(payload: Dict[str, Any], issued: datetime) -> str:
    seed = f"{payload.get('ref')}|{payload.get('cur')}|{payload.get('overall_score')}|{issued.isoformat()}"
    return f"DD-{issued:%Y%m%d}-{hashlib.sha1(seed.encode()).hexdigest()[:6].upper()}"


def _ddoc_version() -> str:
    try:
        from importlib.metadata import version
        return version("ddoc")
    except Exception:  # noqa: BLE001
        return "?"


def _rounded_bar(x0: float, y: float, x1: float, h: float, r: float = 4.0) -> str:
    """Horizontal bar, square at the baseline (x0), rounded at the data end."""
    w = x1 - x0
    if w <= 0:
        return ""
    r = min(r, w, h / 2)
    return (f"M{x0:.1f},{y:.1f}H{x1 - r:.1f}Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f}"
            f"V{y + h - r:.1f}Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f}H{x0:.1f}Z")


def _table_view(lang: str, head: List[str], rows: List[List[str]], num_cols: Tuple[int, ...] = ()) -> str:
    th = "".join(f"<th class='{'num' if i in num_cols else ''}'>{_e(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td class='{'num' if i in num_cols else ''}'>{_e(c)}</td>"
                         for i, c in enumerate(r)) + "</tr>"
        for r in rows
    )
    return (f"<details class='tv'><summary>{_e(t('chart.table_view', lang))}</summary>"
            f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></details>")


def _legend(lang: str) -> str:
    return (f"<div class='legend'><span class='key' style='background:var(--ref)'></span>"
            f"{_e(t('chart.ref', lang))}<span class='key' style='background:var(--cur)'></span>"
            f"{_e(t('chart.cur', lang))}</div>")


# ── Charts ────────────────────────────────────────────────────────────


def _score_chart(scores: List[Tuple[str, float]], threshold: float, attr_threshold: float,
                 lang: str) -> str:
    """Ranked horizontal bars with the two thresholds as reference lines.
    Attributes at or over the threshold carry the accent; the rest are gray."""
    W, label_w, right_pad, row, bar = 720, 200, 60, 32, 14
    top = 40
    H = top + row * len(scores) + 8
    hi = min(1.0, max(0.5, attr_threshold * 1.4, *(s for _, s in scores)) * 1.0)
    x0, x1 = label_w, W - right_pad
    sx = lambda v: x0 + (x1 - x0) * max(0.0, min(v, hi)) / hi  # noqa: E731
    out = [f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='{_e(t('sec.scores', lang))}'>"]
    out.append(f"<line x1='{x0}' y1='{top - 4}' x2='{x0}' y2='{H - 6}' stroke='var(--axis)' stroke-width='1'/>")
    # Two reference lines, labels on separate rows so they never collide.
    for (value, key), label_y in zip(((threshold, "chart.threshold"),
                                      (attr_threshold, "chart.attr_threshold")),
                                     (top - 24, top - 8)):
        x = sx(value)
        out.append(f"<line x1='{x:.1f}' y1='{label_y + 4}' x2='{x:.1f}' y2='{H - 6}' "
                   f"stroke='var(--ink-2)' stroke-width='1'/>")
        out.append(f"<text x='{x + 4:.1f}' y='{label_y}' class='tick-lbl'>"
                   f"{_e(t(key, lang))} {value:g}</text>")
    for i, (name, score) in enumerate(scores):
        y = top + i * row
        cy = y + (row - bar) / 2
        color = "var(--cur)" if score >= threshold else "var(--ref)"
        state = ("status.over_attr" if score >= attr_threshold
                 else "status.over" if score >= threshold else "status.under")
        label = name if len(name) <= 26 else name[:25] + "…"
        out.append(f"<text x='{label_w - 12}' y='{y + row / 2 + 4:.1f}' text-anchor='end'>{_e(label)}</text>")
        path = _rounded_bar(x0, cy, sx(score), bar)
        if path:
            out.append(f"<path d='{path}' fill='{color}'/>")
        out.append(f"<text x='{sx(score) + 8:.1f}' y='{y + row / 2 + 4:.1f}' class='val-lbl'>{score:.3f}</text>")
        tip = f"{score:.3f}\n{name} · {t(state, lang)}"
        out.append(f"<rect class='hit' x='0' y='{y}' width='{W}' height='{row}' tabindex='0' "
                   f"data-tip='{_e(tip)}'><title>{_e(name)}: {score:.3f}</title></rect>")
    out.append("</svg>")
    return "".join(out)


def _dumbbell(dists: Dict[str, Dict[str, float]], lang: str, max_rows: int = 10) -> Tuple[str, List[List[str]], int]:
    """Before → after per value: a gray dot for ref, an accent dot for cur."""
    ref_n, cur_n = _norm(dists.get("ref", {})), _norm(dists.get("cur", {}))
    labels = [k for k in dict.fromkeys([*dists.get("ref", {}), *dists.get("cur", {})])
              if max(ref_n.get(k, 0.0), cur_n.get(k, 0.0)) >= 0.0005]
    labels.sort(key=lambda k: abs(cur_n.get(k, 0.0) - ref_n.get(k, 0.0)), reverse=True)
    hidden = max(0, len(labels) - max_rows)
    shown = labels[:max_rows]
    table = [[k, _pct(ref_n.get(k, 0.0)), _pct(cur_n.get(k, 0.0)),
              t("fmt.pp", lang, value=(cur_n.get(k, 0.0) - ref_n.get(k, 0.0)) * 100)] for k in labels]
    if not shown:
        return "", table, 0
    W, label_w, right_w, row, top = 720, 160, 90, 30, 22
    H = top + row * len(shown) + 6
    peak = max([*ref_n.values(), *cur_n.values(), 0.01])
    step = 0.1 if peak <= 0.5 else 0.25
    hi = min(1.0, step * (int(peak / step) + 1))
    x0, x1 = label_w, W - right_w
    sx = lambda v: x0 + (x1 - x0) * v / hi  # noqa: E731
    out = [f"<svg viewBox='0 0 {W} {H}' role='img'>"]
    tick = 0.0
    while tick <= hi + 1e-9:
        x = sx(tick)
        out.append(f"<line x1='{x:.1f}' y1='{top - 6}' x2='{x:.1f}' y2='{H - 4}' stroke='var(--grid)' stroke-width='1'/>")
        out.append(f"<text x='{x:.1f}' y='{top - 10}' text-anchor='middle' class='tick-lbl'>{tick * 100:.0f}%</text>")
        tick += step
    for i, k in enumerate(shown):
        y = top + i * row + row / 2
        r, c = ref_n.get(k, 0.0), cur_n.get(k, 0.0)
        label = k if len(k) <= 20 else k[:19] + "…"
        out.append(f"<text x='{label_w - 12}' y='{y + 4:.1f}' text-anchor='end'>{_e(label)}</text>")
        out.append(f"<line x1='{sx(r):.1f}' y1='{y:.1f}' x2='{sx(c):.1f}' y2='{y:.1f}' "
                   f"stroke='var(--link)' stroke-width='2' stroke-linecap='round'/>")
        for v, color in ((r, "var(--ref)"), (c, "var(--cur)")):
            out.append(f"<circle cx='{sx(v):.1f}' cy='{y:.1f}' r='5.5' fill='{color}' "
                       f"stroke='var(--surface)' stroke-width='2'/>")
        delta = t("fmt.pp", lang, value=(c - r) * 100)
        out.append(f"<text x='{W - 8}' y='{y + 4:.1f}' text-anchor='end' class='val-lbl'>{_e(delta)}</text>")
        tip = f"{_pct(r)} → {_pct(c)} ({delta})\n{k}"
        out.append(f"<rect class='hit' x='0' y='{y - row / 2:.1f}' width='{W}' height='{row}' tabindex='0' "
                   f"data-tip='{_e(tip)}'><title>{_e(k)}: {_pct(r)} → {_pct(c)}</title></rect>")
    out.append("</svg>")
    return "".join(out), table, hidden


_RANGE = re.compile(r"^\s*(-?[\d.eE+]+)\s*[–-]\s*(-?[\d.eE+]+)\s*$")


def _histogram(dists: Dict[str, Dict[str, float]], summary: Optional[Dict[str, Any]],
               lang: str) -> Tuple[str, List[List[str]]]:
    """Overlaid ref / cur distributions over value bins, with medians."""
    ref_n, cur_n = _norm(dists.get("ref", {})), _norm(dists.get("cur", {}))
    bins = []
    for label in dict.fromkeys([*dists.get("ref", {}), *dists.get("cur", {})]):
        m = _RANGE.match(str(label))
        if not m:
            return "", []
        bins.append((float(m.group(1)), float(m.group(2)), label))
    if not bins:
        return "", []
    bins.sort()
    table = [[label, _pct(ref_n.get(label, 0.0)), _pct(cur_n.get(label, 0.0))] for _, _, label in bins]
    W, left, right, top, plot_h, axis_h = 720, 48, 16, 18, 180, 26
    H = top + plot_h + axis_h
    lo, hi = bins[0][0], bins[-1][1]
    span = (hi - lo) or 1.0
    peak = max([*ref_n.values(), *cur_n.values(), 1e-9])
    sx = lambda v: left + (W - left - right) * (v - lo) / span  # noqa: E731
    sy = lambda p: top + plot_h * (1 - p / peak)  # noqa: E731
    base = top + plot_h
    out = [f"<svg viewBox='0 0 {W} {H}' role='img'>"]
    for frac in (0.5, 1.0):
        y = sy(peak * frac)
        out.append(f"<line x1='{left}' y1='{y:.1f}' x2='{W - right}' y2='{y:.1f}' stroke='var(--grid)' stroke-width='1'/>")
        out.append(f"<text x='{left - 8}' y='{y + 4:.1f}' text-anchor='end' class='tick-lbl'>{peak * frac * 100:.0f}%</text>")
    out.append(f"<line x1='{left}' y1='{base}' x2='{W - right}' y2='{base}' stroke='var(--axis)' stroke-width='1'/>")
    for share, color in ((ref_n, "var(--ref)"), (cur_n, "var(--cur)")):
        pts = [f"M{sx(lo):.1f},{base}"]
        for a, b, label in bins:
            y = sy(share.get(label, 0.0))
            pts.append(f"L{sx(a):.1f},{base}L{sx(a):.1f},{y:.1f}L{sx(b):.1f},{y:.1f}")
            pts.append(f"L{sx(b):.1f},{base}")
        d = "".join(pts)
        out.append(f"<path d='{d}Z' fill='{color}' fill-opacity='0.10' stroke='none'/>")
        out.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2' stroke-linejoin='round'/>")
    for value in (lo, (lo + hi) / 2, hi):
        out.append(f"<text x='{sx(value):.1f}' y='{base + 18}' text-anchor='middle' class='tick-lbl'>{value:.4g}</text>")
    median = (summary or {}).get("median") or {}
    marks = [(side, median.get(side), color) for side, color in
             (("ref", "var(--ref)"), ("cur", "var(--cur)"))
             if median.get(side) is not None and lo <= median.get(side) <= hi]
    for _, v, color in marks:
        x = sx(v)
        out.append(f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{base}' stroke='{color}' stroke-width='1.5'/>")

    def _median_label(x: float, y: float, text: str) -> str:
        anchor = "start" if x < W * 0.75 else "end"
        nx = x + 6 if anchor == "start" else x - 6
        return (f"<text x='{nx:.1f}' y='{y}' text-anchor='{anchor}' class='val-lbl halo'>"
                f"{_e(text)}</text>")

    name = t("chart.median", lang)
    if len(marks) == 2 and abs(sx(marks[0][1]) - sx(marks[1][1])) < 90:
        x = max(sx(marks[0][1]), sx(marks[1][1]))
        out.append(_median_label(x, top + 14, f"{name} {marks[0][1]:.4g} → {marks[1][1]:.4g}"))
    else:
        for side, v, _ in marks:
            out.append(_median_label(sx(v), top + 14, f"{name} {v:.4g}"))
    for a, b, label in bins:
        tip = f"{_pct(ref_n.get(label, 0.0))} → {_pct(cur_n.get(label, 0.0))}\n{label}"
        out.append(f"<rect class='hit' x='{sx(a):.1f}' y='{top}' width='{max(sx(b) - sx(a), 2):.1f}' "
                   f"height='{plot_h}' tabindex='0' data-tip='{_e(tip)}'><title>{_e(label)}</title></rect>")
    out.append("</svg>")
    return "".join(out), table


# ── Sections ──────────────────────────────────────────────────────────


def _tiles(verdict: str, severity: str, score: Optional[float], threshold: float,
           attr_threshold: float, scores: List[Tuple[str, float]],
           samples: Optional[Dict[str, Any]], lang: str) -> str:
    token, icon = _status(verdict, severity)
    verdict_tile = (
        f"<div class='tile'><div class='lbl'>{_e(t('tile.verdict', lang))}</div>"
        f"<div class='val status'><span class='ico {token}' style='background:var(--{token})'>{icon}</span>"
        f"{_e(verdict)}</div><div class='sub'>{_e(t('label.severity', lang))}: "
        f"{_e(severity_label(severity, lang))}</div></div>"
    )
    if score is None:
        score_tile = (f"<div class='tile'><div class='lbl'>{_e(t('tile.overall', lang))}</div>"
                      f"<div class='val'>—</div></div>")
    else:
        hi = max(1e-9, min(1.0, max(score, attr_threshold) * 1.25))
        fill = ("var(--critical)" if score >= attr_threshold
                else "var(--serious)" if score >= threshold else "var(--cur)")
        ticks = "".join(f"<span class='tick' style='left:{min(v / hi, 1) * 100:.1f}%'></span>"
                        for v in (threshold, attr_threshold))
        score_tile = (
            f"<div class='tile'><div class='lbl'>{_e(t('tile.overall', lang))}</div>"
            f"<div class='val'>{score:.3f}</div>"
            f"<div class='meter' role='img' aria-label='{score:.3f}'>"
            f"<span class='fill' style='width:{min(score / hi, 1) * 100:.1f}%;background:{fill}'></span>{ticks}</div>"
            f"<div class='sub'>{_e(t('tile.criteria_sub', lang, threshold=threshold, attr_threshold=attr_threshold))}</div></div>"
        )
    over = [n for n, s in scores if s >= threshold]
    over_tile = (
        f"<div class='tile'><div class='lbl'>{_e(t('tile.over', lang))}</div>"
        f"<div class='val'>{len(over)} / {len(scores)}</div>"
        f"<div class='sub'>{_e(', '.join(over[:3]) + (' …' if len(over) > 3 else '') or '—')}</div></div>"
    )
    if samples:
        samples_val = f"{_int(samples.get('ref'))} / {_int(samples.get('cur'))}"
        samples_sub = ""
    else:
        samples_val, samples_sub = "—", t("tile.samples_varies", lang) if scores else ""
    samples_tile = (f"<div class='tile'><div class='lbl'>{_e(t('tile.samples', lang))}</div>"
                    f"<div class='val'>{_e(samples_val)}</div><div class='sub'>{_e(samples_sub)}</div></div>")
    return f"<div class='tiles'>{verdict_tile}{score_tile}{over_tile}{samples_tile}</div>"


def _pod(name: str, score: float, attr: Dict[str, Any], dists: Optional[Dict[str, Any]],
         threshold: float, attr_threshold: float, lang: str) -> str:
    summary = attr.get("summary") or {}
    kind = summary.get("kind")
    state = ("status.over_attr" if score >= attr_threshold
             else "status.over" if score >= threshold else "status.under")
    head = (f"<div class='hd'><h3>{_e(name)}</h3><span class='kind'>"
            f"{_e(t('kind.' + kind, lang) if kind else '')}{' · ' if kind else ''}"
            f"{_e(t('chart.score', lang))} {score:.3f} · {_e(t(state, lang))}</span></div>")
    body = []
    line = dx.summary_text(summary, lang)
    if line:
        body.append(f"<p class='summary'>{_e(line)}</p>")
    chips = [dx.pattern_text(p, lang) for p in attr.get("patterns") or []
             if p["code"] not in ("share_shift", "mean_shift")]
    if chips:
        body.append("<div class='chips'>" + "".join(f"<span class='chip'>{_e(c)}</span>" for c in chips) + "</div>")
    if dists:
        ref_raw, cur_raw = dists.get("ref", {}), dists.get("cur", {})
        if dx.is_stat_dict(ref_raw) or dx.is_stat_dict(cur_raw):
            rows = [[k, dx._num(ref_raw.get(k)), dx._num(cur_raw.get(k))]
                    for k in dict.fromkeys([*ref_raw, *cur_raw])]
            body.append("<table><thead><tr>" + "".join(
                f"<th class='{c}'>{_e(h)}</th>" for c, h in
                (("", t("col.value", lang)), ("num", t("col.ref", lang)), ("num", t("col.cur", lang))))
                + "</tr></thead><tbody>" + "".join(
                f"<tr><td>{_e(a)}</td><td class='num'>{_e(b)}</td><td class='num'>{_e(c)}</td></tr>"
                for a, b, c in rows) + "</tbody></table>")
        else:
            svg, table = (_histogram(dists, summary, lang) if kind == "numeric" else ("", []))
            if svg:
                body.append(_legend(lang) + svg)
                body.append(_table_view(lang, [t("col.range", lang), t("col.ref", lang), t("col.cur", lang)],
                                        table, (1, 2)))
            else:
                svg, table, hidden = _dumbbell(dists, lang)
                if svg:
                    body.append(_legend(lang) + svg)
                    if hidden:
                        body.append(f"<div class='legend'>{_e(t('note.more_rows', lang, count=hidden))}</div>")
                    body.append(_table_view(lang, [t("col.value", lang), t("col.ref", lang),
                                                   t("col.cur", lang), t("col.change", lang)], table, (1, 2, 3)))
    if not body:
        return ""
    return f"<section class='pod'>{head}<div class='bd'>{''.join(body)}</div></section>"


def _features_table(attributes: List[Dict[str, Any]], lang: str) -> str:
    def arrow(pair: Dict[str, Any], fmt) -> str:
        if not pair or pair.get("ref") is None:
            return "—"
        return f"{fmt(pair.get('ref'))} → {fmt(pair.get('cur'))}"

    rows = []
    for a in attributes:
        s = a.get("summary") or {}
        kind = s.get("kind")
        if kind not in ("numeric", "categorical"):
            continue
        notes = "; ".join(dx.pattern_text(p, lang) for p in a.get("patterns") or [])
        if kind == "numeric":
            rng = "—"
            mn, mx = s.get("min") or {}, s.get("max") or {}
            if mn.get("ref") is not None:
                rng = (f"{dx._num(mn.get('ref'))} – {dx._num(mx.get('ref'))} → "
                       f"{dx._num(mn.get('cur'))} – {dx._num(mx.get('cur'))}")
            rows.append([a["name"], t("kind.numeric", lang), arrow(s.get("missing"), _pct),
                         arrow(s.get("mean"), dx._num), arrow(s.get("std"), dx._num), rng, notes])
        else:
            rows.append([a["name"], t("kind.categorical", lang), arrow(s.get("missing"), _pct),
                         "—", "—", "—", notes])
    if not rows:
        return ""
    head = [t(k, lang) for k in ("col.attribute", "col.type", "col.missing", "col.mean",
                                 "col.std", "col.minmax", "col.notes")]
    # Drop value columns that are empty in every row (e.g. categorical-only data).
    keep = [i for i in range(len(head)) if i in (0, 1, 6) or any(r[i] != "—" for r in rows)]
    th = "".join(f"<th>{_e(head[i])}</th>" for i in keep)
    # Short value columns stay on one line; the findings column wraps.
    body = "".join("<tr>" + "".join(
        f"<td class='{'nw' if 0 < i < 5 else ''}'>{_e(r[i])}</td>" for i in keep) + "</tr>"
        for r in rows)
    return f"<table class='features'><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def _checks_table(items: List[Tuple[str, str]], lang: str) -> str:
    head = (f"<tr><th class='num'>{_e(t('col.no', lang))}</th><th>{_e(t('col.attribute', lang))}</th>"
            f"<th>{_e(t('col.item', lang))}</th><th>{_e(t('col.done', lang))}</th></tr>")
    body = "".join(
        f"<tr><td class='num'>{i}</td><td>{_e(attr)}</td><td>{_e(text)}</td><td class='done'>☐</td></tr>"
        for i, (attr, text) in enumerate(items, 1)
    )
    return f"<table class='checks'><thead>{head}</thead><tbody>{body}</tbody></table>"


# ── Document ──────────────────────────────────────────────────────────


def _toolbar(lang: str) -> str:
    """Theme switch (auto / day / night) and a print button. Hidden until
    the script runs, so a reader without JavaScript sees no dead controls;
    never printed."""
    modes = "".join(
        f"<button type='button' data-set-theme='{m}' aria-pressed='false'>{_e(t(key, lang))}</button>"
        for m, key in (("auto", "ui.theme_auto"), ("light", "ui.theme_light"), ("dark", "ui.theme_dark"))
    )
    return (f"<nav class='bar' hidden aria-label='{_e(t('ui.theme', lang))}'>"
            f"<span>{_e(t('ui.theme', lang))}</span><span class='seg' role='group'>{modes}</span>"
            f"<button type='button' class='print'>{_e(t('ui.print', lang))}</button></nav>")


def build_consultation_html(
    payload: Dict[str, Any],
    distributions: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    *,
    max_evidence: int = 5,
    lang: str = DEFAULT_LANG,
    issuer: Optional[str] = None,
) -> str:
    """Render the verdict payload (+ optional per-attribute ref/cur
    distributions) as a standalone diagnostic report page."""
    verdict = payload.get("verdict", "BLIND")
    severity = payload.get("severity", "unknown")
    score = payload.get("overall_score")
    threshold = payload.get("threshold") or 0.15
    attr_threshold = payload.get("attr_threshold") or 0.25
    top: List[Tuple[str, float]] = [tuple(x) for x in payload.get("top_attributes") or []]
    raw = payload.get("raw") or {}
    dists = distributions or {}
    issued = datetime.now(timezone.utc)
    diagnosis = payload.get("diagnosis") or dx.build(
        verdict=verdict, overall=score, top=top, threshold=threshold,
        attr_threshold=attr_threshold, result=raw, evidence=dists,
        reason=payload.get("reason"),
    )
    scores = sorted(((k, float(v)) for k, v in (raw.get("attribute_drifts") or dict(top)).items()
                     if isinstance(v, (int, float))), key=lambda kv: kv[1], reverse=True)
    version = _ddoc_version()
    token, icon = _status(verdict, severity)

    # Masthead + subject / method / result table.
    parts = [
        f"<!doctype html><html lang='{_e(lang)}'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_e(t('doc.title', lang))} — {_e(verdict)}</title><style>{_CSS}</style>"
        f"<script>{_THEME_BOOT}</script></head>"
        f"<body><div class='doc'>{_toolbar(lang)}<div class='sheet'>",
        f"<header class='mast'><div><div class='issuer'>{_e(issuer or t('doc.issuer_default', lang))}</div>"
        f"<h1>{_e(t('doc.title', lang))}</h1></div>"
        f"<div class='docmeta'>{_e(t('doc.number', lang))} <b>{_e(_doc_number(payload, issued))}</b><br>"
        f"{_e(t('doc.issued', lang))} <b>{issued:%Y-%m-%d %H:%M} UTC</b><br>"
        f"{_e(t('doc.tool', lang))} <b>ddoc {_e(version)}</b></div></header>",
    ]
    info = [
        (t("info.subject", lang), f"ref: {payload.get('ref')}\ncur: {payload.get('cur')}"),
        (t("info.modality", lang), payload.get("modality") or "—"),
        (t("info.method", lang), f"{payload.get('engine')} / {payload.get('detector')}"),
        (t("info.criteria", lang), t("tile.criteria_sub", lang, threshold=threshold,
                                     attr_threshold=attr_threshold)),
    ]
    rows = "".join(f"<tr><th>{_e(k)}</th><td style='white-space:pre-line'>{_e(v)}</td></tr>" for k, v in info)
    rows += (f"<tr><th>{_e(t('info.result', lang))}</th><td><span class='status'>"
             f"<span class='ico {token}' style='background:var(--{token})'>{icon}</span>"
             f"<b>{_e(verdict)}</b> · {_e(t('label.severity', lang))} "
             f"{_e(severity_label(severity, lang))}</span></td></tr>")
    parts.append(f"<table class='info'>{rows}</table>")

    n = 0

    def section(key: str) -> str:
        nonlocal n
        n += 1
        return f"<h2>{n}. {_e(t(key, lang))}</h2>"

    # 1. Summary.
    parts.append(section("sec.summary"))
    parts.append(_tiles(verdict, severity, score, threshold, attr_threshold, scores,
                        diagnosis.get("samples"), lang))
    for w in diagnosis.get("warnings") or []:
        parts.append(f"<div class='caution'><b>{_e(t('note.caution', lang))}</b> — "
                     f"{_e(dx.warning_text(w, lang))}</div>")

    if verdict == "BLIND":
        parts.append(section("sec.reason"))
        parts.append(f"<p>{_e(payload.get('reason') or '')}</p>")
        hints = payload.get("hints") or []
        if hints:
            parts.append(section("sec.checks"))
            parts.append(_checks_table([("—", h) for h in hints], lang))
    else:
        # 2. Basis for the verdict.
        parts.append(section("sec.why"))
        parts.append("<ul class='why'>" + "".join(
            f"<li>{_e(dx.rationale_text(r, lang))}</li>" for r in diagnosis.get("rationale") or []) + "</ul>")

        # 3. Results by attribute.
        if scores:
            parts.append(section("sec.scores"))
            parts.append(_score_chart(scores, threshold, attr_threshold, lang))
            parts.append(_table_view(
                lang, [t("col.attribute", lang), t("col.score", lang), t("col.status", lang)],
                [[k, f"{s:.4f}", t("status.over_attr" if s >= attr_threshold else
                                   "status.over" if s >= threshold else "status.under", lang)]
                 for k, s in scores], (1,)))

        # 4. Findings by attribute.
        by_name = {a["name"]: a for a in diagnosis.get("attributes") or []}
        pods = [_pod(k, s, by_name.get(k, {}), dists.get(k), threshold, attr_threshold, lang)
                for k, s in scores[:max_evidence]]
        pods = [p for p in pods if p]
        if pods:
            parts.append(section("sec.detail"))
            parts += pods

        # 5. Characteristics of the data.
        features = _features_table(diagnosis.get("attributes") or [], lang)
        if features:
            parts.append(section("sec.features"))
            parts.append(features)

        # 6. Items to check.
        parts.append(section("sec.checks"))
        seen, items = set(), []
        for c in diagnosis.get("checks") or []:
            text = dx.check_text(c, lang)
            if text not in seen:
                seen.add(text)
                items.append((c.get("attr", "—"), text))
        parts.append(_checks_table(items, lang) if items else f"<p>{_e(t('checks.none', lang))}</p>")

    # Limitations and remarks.
    parts.append(section("sec.limits"))
    parts.append(f"<p class='limits'>{_e(t('note.limits_text', lang))}</p>")

    parts.append(f"<footer class='foot'><span>{_e(t('doc.footer', lang, version=version))}</span>"
                 f"<span>{_e(_doc_number(payload, issued))}</span></footer>")
    parts.append(f"</div></div><div id='tip' role='tooltip'></div><script>{_SCRIPT}</script></body></html>")
    return "".join(parts)

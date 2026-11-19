"""``ddoc report render`` — render a drift / EDA result to a report file.

The CLI:

1. Reads the ``--input`` JSON: a ``ddoc diff --json`` payload, a drift
   envelope (``ddoc analyze drift --json``) or an EDA envelope.
2. Tries the ``report_render`` hookspec — if any plugin returns a
   non-None result, use that.
3. Otherwise renders built-in. Drift input becomes the same diagnostic
   report as ``ddoc diff --report`` (a drift envelope is judged with
   the same thresholds first); Markdown carries the same diagnosis for
   pull requests and tickets; PDF prints the HTML. EDA input becomes a
   data profile report in the same format (the built-in CSV profile in
   full, an EDA plugin's summary inside the same document). Multi-modal
   drift envelopes use the Jinja template in ``ddoc/templates/``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich import print as rprint

from .utils import get_pmgr
from ... import diagnosis as dx
from ...i18n import resolve_lang, t


def _load_input(input_path: Path) -> Dict[str, Any]:
    if not input_path.exists():
        raise typer.BadParameter(f"input file does not exist: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))


def _normalize_attribute_drifts(value: Any) -> list[tuple[str, Optional[float], Dict[str, Any]]]:
    """Normalize the many possible ``attribute_drifts`` shapes into a
    uniform list of ``(name, score, extras)`` tuples.

    Closes the markdown renderer's ``{v:.4f}``
    crash on non-flat shapes. Accepted inputs:

    * Flat dict: ``{"blur": 0.18, "exposure": 0.07}`` — original shape.
    * Nested dict: ``{"blur": {"score": 0.18, "status": "warning"}, ...}``.
    * List of dicts: ``[{"attribute": "blur", "score": 0.18,
      "status": "warning"}, ...]``.
    * Empty / None / unrecognized → ``[]`` (renderers skip the section).

    Score may be ``None`` if the source shape doesn't supply a numeric
    value; renderers should display "—" or skip in that case.
    """
    if not value:
        return []

    out: list[tuple[str, Optional[float], Dict[str, Any]]] = []

    def _coerce_score(v: Any) -> Optional[float]:
        if isinstance(v, (int, float)):
            return float(v)
        return None

    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, dict):
                score = _coerce_score(v.get("score"))
                extras = {kk: vv for kk, vv in v.items() if kk != "score"}
                out.append((str(k), score, extras))
            else:
                out.append((str(k), _coerce_score(v), {}))
        return out

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                name = str(
                    item.get("attribute")
                    or item.get("name")
                    or item.get("key")
                    or "?"
                )
                score = _coerce_score(item.get("score"))
                extras = {
                    k: v for k, v in item.items()
                    if k not in ("attribute", "name", "key", "score")
                }
                out.append((name, score, extras))
        return out

    return []  # unrecognized — skip the section gracefully


def _classify_envelope(payload: Dict[str, Any]) -> tuple[Optional[Dict], Optional[Dict]]:
    """Return ``(drift_result, eda_result)`` after a heuristic.

    drift envelopes carry ``modality`` + (``overall_score`` |
    ``attribute_drifts``). EDA envelopes have ``modality`` + ``summary``
    /``files_analyzed``/``series_analyzed``. When ambiguous, treat as
    drift (the more common case).
    """
    if "modalities" in payload:
        # multi-modal stacked result; pass as drift since each modality
        # is a drift result. Templates will iterate.
        return payload, None
    is_eda_shaped = (
        payload.get("profiler") == "builtin:csv_profile"
        or "files_analyzed" in payload
        or "series_analyzed" in payload
        or "texts_analyzed" in payload
        or ("summary" in payload and "overall_score" not in payload)
    )
    if is_eda_shaped:
        return None, payload
    return payload, None


def _diagnostic_payload(payload: Dict[str, Any], *, threshold: float,
                        attr_threshold: float) -> Optional[Dict[str, Any]]:
    """A ``ddoc diff``-shaped payload for the diagnostic report, or None
    when the input is not a single drift result.

    A ``ddoc diff --json`` payload is used as is. A drift envelope gets a
    verdict and diagnosis computed exactly as ``ddoc diff`` would."""
    from .diff import compute_verdict

    if "verdict" in payload and "diagnosis" in payload:
        return payload
    if "modalities" in payload:
        return None
    if "overall_score" not in payload and not payload.get("attribute_drifts"):
        return None
    result = dict(payload)
    result["attribute_drifts"] = {name: score for name, score, _ in
                                  _normalize_attribute_drifts(payload.get("attribute_drifts"))
                                  if score is not None}
    verdict, severity, top = compute_verdict(result, threshold=threshold, attr_threshold=attr_threshold)
    evidence = result["distributions"] if isinstance(result.get("distributions"), dict) else {}
    overall = float(result.get("overall_score") or 0.0)
    return {
        "verdict": verdict, "severity": severity, "overall_score": overall,
        "threshold": threshold, "attr_threshold": attr_threshold, "top_attributes": top,
        "modality": result.get("modality"), "detector": result.get("detector"),
        "engine": result.get("engine") or "plugin",
        "ref": payload.get("ref") or payload.get("data_path_ref") or payload.get("snapshot_ref") or "—",
        "cur": payload.get("cur") or payload.get("data_path_cur") or payload.get("snapshot_cur") or "—",
        "hints": [],
        "diagnosis": dx.build(verdict=verdict, overall=overall, top=top, threshold=threshold,
                              attr_threshold=attr_threshold, result=result, evidence=evidence),
        "raw": result,
    }


def _written(out_path: Path, fmt: str) -> Dict[str, Any]:
    return {"status": "success", "format": fmt, "output_path": str(out_path),
            "size_bytes": out_path.stat().st_size, "renderer": "builtin"}


def _builtin_render(
    drift_result: Optional[Dict[str, Any]],
    eda_result: Optional[Dict[str, Any]],
    fmt: str,
    output_path: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Built-in fallback renderer used when no plugin handles the
    request. HTML via Jinja (templates ship with the wheel), PDF via
    weasyprint, Markdown via inline string formatting."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lang = cfg.get("lang") or resolve_lang()

    diag = None
    if drift_result is not None:
        diag = _diagnostic_payload(drift_result, threshold=cfg.get("threshold") or 0.15,
                                   attr_threshold=cfg.get("attr_threshold") or 0.25)
    if diag is not None:
        from .diff_report import build_consultation_html, build_consultation_markdown, for_paged_media

        if fmt == "md":
            out_path.write_text(build_consultation_markdown(diag, lang=lang, title=cfg.get("title")),
                                encoding="utf-8")
            return _written(out_path, "md")
        raw = diag.get("raw") or {}
        html = build_consultation_html(
            diag, raw.get("distributions") if isinstance(raw.get("distributions"), dict) else None,
            lang=lang, issuer=cfg.get("issuer"), title=cfg.get("title"))
        if fmt == "html":
            out_path.write_text(html, encoding="utf-8")
            return _written(out_path, "html")
        if fmt == "pdf":
            from weasyprint import HTML
            HTML(string=for_paged_media(html)).write_pdf(str(out_path))
            return _written(out_path, "pdf")
        raise typer.BadParameter(f"built-in fallback supports html|pdf|md; got {fmt!r}")

    if drift_result is None and eda_result is not None:
        from .diff_report import for_paged_media
        from .profile_report import build_profile_html, build_profile_markdown

        if fmt == "md":
            out_path.write_text(build_profile_markdown(eda_result, lang=lang, title=cfg.get("title")),
                                encoding="utf-8")
            return _written(out_path, "md")
        html = build_profile_html(eda_result, lang=lang, issuer=cfg.get("issuer"), title=cfg.get("title"))
        if fmt == "html":
            out_path.write_text(html, encoding="utf-8")
            return _written(out_path, "html")
        if fmt == "pdf":
            from weasyprint import HTML
            HTML(string=for_paged_media(html)).write_pdf(str(out_path))
            return _written(out_path, "pdf")
        raise typer.BadParameter(f"built-in fallback supports html|pdf|md; got {fmt!r}")

    if fmt == "md":
        body = _render_markdown(drift_result, eda_result, cfg)
        out_path.write_text(body, encoding="utf-8")
        return {
            "status": "success",
            "format": "md",
            "output_path": str(out_path),
            "size_bytes": out_path.stat().st_size,
            "renderer": "builtin",
        }

    # HTML / PDF both go through Jinja first.
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    templates_dir = Path(__file__).resolve().parent.parent.parent / "templates"
    if not templates_dir.exists():
        raise typer.BadParameter(
            f"built-in templates dir missing: {templates_dir} "
            "(reinstall ddoc; templates are package-data)"
        )

    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.globals["lang"] = lang
    env.globals["_"] = lambda key, **kw: t(key, lang, **kw)

    def _status_class(score: Optional[float]) -> str:
        if score is None:
            return "normal"
        if score < 0.15:
            return "normal"
        if score < 0.25:
            return "warning"
        return "critical"

    # Only multi-modal drift envelopes reach the template now.
    drift_result = drift_result or {}
    tpl = env.get_template("drift_report.html")
    # Pre-normalize attribute_drifts so the template iterates a uniform
    # shape regardless of how the producing detector serialized it.
    attr_rows = _normalize_attribute_drifts(drift_result.get("attribute_drifts"))
    html = tpl.render(
        drift=drift_result,
        attribute_drifts_rows=attr_rows,
        title=cfg.get("title"),
        status_class=_status_class(drift_result.get("overall_score")),
        raw_json=json.dumps(drift_result, indent=2, ensure_ascii=False, default=str),
    )

    if fmt == "html":
        out_path.write_text(html, encoding="utf-8")
        return {
            "status": "success",
            "format": "html",
            "output_path": str(out_path),
            "size_bytes": out_path.stat().st_size,
            "renderer": "builtin",
        }

    if fmt == "pdf":
        from weasyprint import HTML
        HTML(string=html).write_pdf(str(out_path))
        return {
            "status": "success",
            "format": "pdf",
            "output_path": str(out_path),
            "size_bytes": out_path.stat().st_size,
            "renderer": "builtin",
        }

    raise typer.BadParameter(f"built-in fallback supports html|pdf|md; got {fmt!r}")


def _render_markdown(drift, eda, cfg) -> str:
    lang = cfg.get("lang") or resolve_lang()
    title = cfg.get("title") or t("report.drift_title" if drift else "report.eda_title", lang)
    lines = [f"# {title}", ""]
    if drift is not None:
        lines.append(f"- {t('report.modality', lang)}: `{drift.get('modality', 'unknown')}`")
        lines.append(f"- {t('report.timestamp', lang)}: {drift.get('timestamp', 'n/a')}")
        lines.append(f"- {t('report.overall_score', lang)}: **{drift.get('overall_score', 0):.4f}**")
        if drift.get("status"):
            lines.append(f"- {t('report.status', lang)}: **{drift['status']}**")
        if drift.get("embedding_drift_detector"):
            lines.append(f"- {t('report.embedding_detector', lang)}: `{drift['embedding_drift_detector']}`")
        normalized = _normalize_attribute_drifts(drift.get("attribute_drifts"))
        if normalized:
            lines += ["", f"## {t('report.attribute_drifts', lang)}", ""]
            for name, score, extras in normalized:
                score_str = "—" if score is None else f"{score:.4f}"
                extras_str = (
                    " (" + ", ".join(f"{k}={v}" for k, v in extras.items()) + ")"
                    if extras else ""
                )
                lines.append(f"- `{name}`: {score_str}{extras_str}")
    lines += ["", f"## {t('report.raw_envelope', lang)}", "", "```json"]
    lines.append(json.dumps(drift if drift is not None else eda, indent=2, ensure_ascii=False, default=str))
    lines += ["```", ""]
    return "\n".join(lines)


report_app = typer.Typer(help="Render drift / EDA results to HTML / PDF / Markdown reports.")


@report_app.command("render")
def render_report(
    input: Path = typer.Option(
        ..., "--input", "-i",
        help="Path to a drift or EDA envelope JSON (e.g. the stdout of `ddoc analyze drift --json`).",
    ),
    output: Path = typer.Option(
        ..., "--out", "-o",
        help="Output report path. Format inferred from extension if --format omitted.",
    ),
    format: Optional[str] = typer.Option(
        None, "--format", "-f",
        help="Report format: html | pdf | md. Defaults to suffix of --out.",
    ),
    title: Optional[str] = typer.Option(
        None, "--title",
        help="Report title (defaults to 'Drift Report' / 'EDA Report').",
    ),
    json_out: bool = typer.Option(
        False, "--json",
        help="Emit machine-readable JSON envelope to stdout instead of pretty progress.",
    ),
    lang: Optional[str] = typer.Option(
        None, "--lang",
        help="Report language: en | ko (default: $DDOC_LANG, else en).",
    ),
    issuer: Optional[str] = typer.Option(
        None, "--issuer",
        help="Organization printed as the report's issuer (default: $DDOC_ISSUER, else 'ddoc').",
    ),
    threshold: float = typer.Option(
        0.15, "--threshold",
        help="Overall-score threshold used to judge a drift envelope (as in `ddoc diff`).",
    ),
    attr_threshold: float = typer.Option(
        0.25, "--attr-threshold",
        help="Single-attribute threshold used to judge a drift envelope (as in `ddoc diff`).",
    ),
):
    """Render a drift / EDA envelope to an HTML, PDF, or Markdown report.

    Examples:
        ddoc report render --input drift.json --out report.html
        ddoc report render -i drift.json -o report.pdf --title "Q4 release drift"
        ddoc report render -i eda.json -o eda.md
        ddoc diff ref.csv cur.csv --json > d.json; ddoc report render -i d.json -o d.md
    """
    fmt = (format or output.suffix.lstrip(".")).lower()
    if fmt not in ("html", "pdf", "md"):
        rprint(f"[red]❌ Unsupported format: {fmt!r}. Pick one of: html, pdf, md.[/red]")
        raise typer.Exit(code=2)

    try:
        payload = _load_input(input)
    except typer.BadParameter as e:
        rprint(f"[red]❌ {e}[/red]")
        raise typer.Exit(code=2)
    except json.JSONDecodeError as e:
        rprint(f"[red]❌ Failed to parse {input}: {e}[/red]")
        raise typer.Exit(code=2)

    drift_result, eda_result = _classify_envelope(payload)
    cfg = {"title": title, "lang": resolve_lang(lang),
           "issuer": issuer or os.environ.get("DDOC_ISSUER") or None,
           "threshold": threshold, "attr_threshold": attr_threshold}

    # Try plugin hookimpls first; fall back to built-in if all return None.
    pm = get_pmgr().pm
    try:
        plugin_result = pm.hook.report_render(
            drift_result=drift_result,
            eda_result=eda_result,
            format=fmt,
            output_path=str(output),
            cfg=cfg,
        )
    except Exception as e:
        # Plugin failure shouldn't kill the CLI — fall through to fallback
        plugin_result = None
        if not json_out:
            rprint(f"[yellow]⚠️  plugin report_render raised: {e} — using built-in fallback[/yellow]")

    if plugin_result is not None:
        result = plugin_result
    else:
        try:
            result = _builtin_render(drift_result, eda_result, fmt, str(output), cfg)
        except Exception as e:
            msg = f"render failed: {e}"
            if json_out:
                sys.stdout.write(json.dumps({"status": "error", "error_code": "render_failed", "message": msg}) + "\n")
            else:
                rprint(f"[red]❌ {msg}[/red]")
            raise typer.Exit(code=1)

    if json_out:
        sys.stdout.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
    else:
        rprint(f"[green]✅ {result.get('format')} report → {result.get('output_path')} "
               f"({result.get('size_bytes', 0)} bytes, renderer={result.get('renderer', 'plugin')})[/green]")

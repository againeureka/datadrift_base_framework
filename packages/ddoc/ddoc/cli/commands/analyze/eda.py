"""EDA analysis command.

``ddoc analyze eda --data-path <p>`` runs the installed plugins'
``eda_run`` hooks directly on a path. Orchestrating backends can
subprocess-invoke this instead of importing ddoc as a library.

``--json`` emits the merged plugin result to stdout as a single JSON
object (no rich formatting).

When no plugin handles the path and it is CSV, the built-in profiler
(:mod:`ddoc.profile`) describes it instead — the same fallback ``ddoc
diff`` has for comparing CSVs.
"""
import sys

import typer
from rich import print as rprint
from pathlib import Path
from typing import Optional

from ..utils import call_hook_isolated, get_pmgr
from ....plugin_install import plugin_command
from .drift import _emit, _emit_error, _merge_plugin_results, emit_progress, _SilencePluginIO


def analyze_eda_command(
    data_path: Optional[str] = typer.Option(
        None, "--data-path",
        help="Data path to analyze.",
    ),
    invalidate_cache: bool = typer.Option(False, "--invalidate-cache", help="Invalidate existing cache before analysis"),
    json_out: bool = typer.Option(
        False, "--json",
        help="Emit machine-readable JSON envelope to stdout (no rich formatting).",
    ),
    ndjson_progress: bool = typer.Option(
        False, "--ndjson-progress",
        help="Emit NDJSON progress lines on stderr (orchestrator streaming).",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Silence all plugin stdout/stderr during hook invocation. CLI's own NDJSON progress (if --ndjson-progress) and final --json envelope still emit.",
    ),
    sample_rows: int = typer.Option(
        0, "--sample-rows", min=0, max=20,
        help="Built-in CSV profile: include the first N rows (off by default: they are raw data).",
    ),
    max_rows: int = typer.Option(
        200_000, "--max-rows", min=0,
        help="Built-in CSV profile: read a uniform random sample of this many rows when larger (0 = all).",
    ),
    report: Optional[Path] = typer.Option(
        None, "--report",
        help="Also write the data profile report: .html, .md or .pdf.",
    ),
    lang: Optional[str] = typer.Option(
        None, "--lang", help="Report language: en | ko (default: $DDOC_LANG, else en).",
    ),
):
    """
    Run EDA analysis on a data path.

    Examples:
        ddoc analyze eda --data-path /data/x
        ddoc analyze eda --data-path /data/x --json
    """
    if not data_path:
        _emit_error(
            "--data-path is required",
            code="path_mode_incomplete", json_out=json_out,
        )
        raise typer.Exit(code=2)

    from ....i18n import resolve_lang, t
    lang = resolve_lang(lang)
    emit_progress(0.05, "start", "eda path mode init", enabled=ndjson_progress)
    if not json_out:
        rprint(f"[cyan]📊 {t('eda.cli.title', lang)}[/cyan] — {data_path}")
    emit_progress(0.2, "plugin_call", "invoking eda_run hook",
                  enabled=ndjson_progress)
    try:
        with _SilencePluginIO(json_out=json_out, quiet=quiet):
            hook_results, hook_errors = call_hook_isolated(
                get_pmgr(), "eda_run",
                snapshot_id="__path__",
                data_path=data_path,
                data_hash="",
                output_path=f"analysis/path_{Path(data_path).name}",
                invalidate_cache=invalidate_cache,
            )
    except Exception as e:
        _emit_error(f"plugin invocation failed: {e}", code="plugin_error", json_out=json_out)
        raise typer.Exit(code=1)
    if hook_errors and not hook_results:
        detail = "; ".join(f"{name}: {err}" for name, err in hook_errors)
        _emit_error(f"plugin invocation failed: {detail}", code="plugin_error", json_out=json_out)
        raise typer.Exit(code=1)
    for name, err in hook_errors:
        rprint(f"[yellow]plugin {name} failed and was skipped: {err}[/yellow]", file=sys.stderr)
    emit_progress(0.9, "merge", "merging plugin results",
                  enabled=ndjson_progress)
    # A plugin that looked and found nothing of its kind (an image plugin
    # given a CSV says files_analyzed: 0) has not profiled the data.
    hook_results = [r for r in hook_results or []
                    if r is not None and not (isinstance(r, dict) and r.get("files_analyzed") == 0)]
    if not hook_results:
        builtin = _builtin_profile(Path(data_path), sample_rows=sample_rows, max_rows=max_rows)
        if builtin is not None:
            hook_results = [builtin]
    merged = _finish_eda(hook_results, json_out=json_out, return_dict=report is not None,
                         lang=lang, report=report)
    if report is not None:
        if merged is None:  # the error was already printed; keep its exit code
            raise typer.Exit(code=1)
        _write_profile(merged, report, lang)
    emit_progress(1.0, "complete", "done", enabled=ndjson_progress)
    return None


def _write_profile(envelope, report: Path, lang: Optional[str]) -> None:
    """The data profile report next to the JSON (best-effort, like diff)."""
    from ....i18n import resolve_lang
    from ..diff_report import for_paged_media, write_pdf
    from ..profile_report import build_profile_html, build_profile_markdown

    lang = resolve_lang(lang)
    try:
        report.parent.mkdir(parents=True, exist_ok=True)
        if report.suffix.lower() == ".md":
            report.write_text(build_profile_markdown(envelope, lang=lang), encoding="utf-8")
        elif report.suffix.lower() == ".pdf":
            write_pdf(for_paged_media(build_profile_html(envelope, lang=lang)), report)
        else:
            report.write_text(build_profile_html(envelope, lang=lang), encoding="utf-8")
        rprint(f"[dim]report → {report}[/dim]", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 — the analysis already succeeded
        rprint(f"[yellow]report not written: {e}[/yellow]", file=sys.stderr)


def _builtin_profile(path: Path, *, sample_rows: int = 0, max_rows: Optional[int] = None):
    """The built-in CSV profile, or None when the path is not CSV."""
    from ..diff import sniff_modality
    from ....profile import profile_csv

    if not path.exists():
        return None
    kind, detail = sniff_modality(path)
    if kind != "csv":
        return None
    return profile_csv(detail.get("files") or [str(path)], sample_rows=sample_rows, max_rows=max_rows)


def _summary(res, lang: str, report: Optional[Path]) -> None:
    """A few lines for the built-in CSV profile: size, column kinds,
    period and the quality findings. The whole profile is in --json and
    the report."""
    from collections import Counter

    from ....i18n import MESSAGES, t
    from ....profile import finding_text

    prof, cols = res.get("profile") or {}, res.get("columns") or {}
    kinds = Counter(c.get("type") for c in cols.values())
    named = ", ".join(f"{t('kind.' + k, lang) if 'kind.' + k in MESSAGES['en'] else k} {n}"
                      for k, n in kinds.most_common())
    rows = f"{prof.get('rows', 0):,}"
    rprint(f"   {t('eda.cli.size', lang, rows=rows, cols=len(cols), kinds=named)}")
    period = prof.get("period") or {}
    if period.get("start"):
        rprint(f"   {t('eda.cli.period', lang, start=period['start'], end=period.get('end', ''))}")
    found = res.get("findings") or []
    if not found:
        rprint(f"   {t('eda.cli.no_findings', lang)}")
    else:
        rprint(f"   {t('eda.cli.findings', lang, n=len(found))}")
        for f in found:
            text = finding_text(f, lang)
            shown, rest = text["signals"][:2], len(text["signals"]) - 2
            more = f" {t('eda.cli.more', lang, n=rest)}" if rest > 0 else ""
            rprint(f"     · {text['title']}: {' · '.join(shown)}{more}")
    if report is None:
        rprint(f"[dim]   {t('eda.cli.hint', lang)}[/dim]")


def _finish_eda(hook_results, *, json_out: bool, return_dict: bool = False,
                lang: str = "en", report: Optional[Path] = None):
    """Shared post-hook handling. Returns the merged dict if ``return_dict``."""
    if not hook_results:
        _emit_error(
            f"No plugin available for EDA analysis. `ddoc doctor` shows what is installed; "
            f"install a plugin for your data, e.g. {plugin_command('tabular')}",
            code="no_plugin", json_out=json_out,
        )
        if return_dict:
            return None
        raise typer.Exit(code=1)

    valid = [r for r in hook_results if r is not None]
    if not valid:
        _emit_error(
            "No plugin returned a valid result.",
            code="empty_result", json_out=json_out,
        )
        if return_dict:
            return None
        raise typer.Exit(code=1)

    res = _merge_plugin_results(valid, hook_name="eda_run")
    from ....profile import is_profile
    if not json_out and is_profile(res):
        _summary(res, lang, report)
    else:
        _emit(res, json_out=json_out)
    return res if return_dict else None

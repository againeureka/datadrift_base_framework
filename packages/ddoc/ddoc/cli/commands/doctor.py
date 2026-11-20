"""``ddoc doctor`` and ``ddoc probe`` — look before running.

* ``doctor`` checks the environment: Python, the libraries the core
  needs, whether each installed plugin actually loads, and which optional
  features are available (PDF reports, ``ddoc serve``, embeddings).
* ``probe <ref> [<cur>]`` says what ``ddoc diff`` would make of the paths
  without judging anything: what kind of data each is, who would judge it,
  and whether the answer would be BLIND — and why.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich import print as rprint

from ...i18n import resolve_lang, t

# Which plugin judges which kind of input (entry-point names).
KIND_PLUGIN = {"categorical": "ddoc_categorical", "vision": "ddoc_vision",
               "text": "ddoc_text", "audio": "ddoc_audio"}
_MARK = {"ok": "[green]✓[/green]", "warn": "[yellow]![/yellow]", "fail": "[red]✕[/red]"}


def _try_import(module: str) -> Optional[str]:
    """None when ``module`` imports, else a one-line reason."""
    import logging

    logging.disable(logging.INFO)  # some libraries log while importing
    try:
        importlib.import_module(module)
        return None
    except Exception as e:  # noqa: BLE001 — any failure is the answer
        return f"{type(e).__name__}: {e}".splitlines()[0][:160]
    finally:
        logging.disable(logging.NOTSET)


def plugin_status() -> List[Dict[str, Any]]:
    """Every installed ddoc plugin and whether it loads."""
    from importlib import metadata as md

    try:
        eps = md.entry_points(group="ddoc")
    except TypeError:  # Python 3.8/3.9
        eps = md.entry_points().get("ddoc", [])
    out = []
    for ep in sorted(eps, key=lambda e: e.name):
        try:
            ep.load()
            error = None
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}".splitlines()[0][:160]
        dist = getattr(ep, "dist", None)
        out.append({"name": ep.name, "version": getattr(dist, "version", "?"), "error": error})
    return out


def environment(lang: str) -> List[Dict[str, Any]]:
    """The doctor's rows: (area, item, status, detail, fix)."""
    from ... import __version__
    from ...plugins.hookspecs import HOOKSPEC_VERSION

    rows: List[Dict[str, Any]] = []

    def row(area: str, item: str, status: str, detail: str = "", fix: str = "") -> None:
        rows.append({"area": area, "item": item, "status": status, "detail": detail, "fix": fix})

    py = sys.version_info
    row("core", "Python", "ok" if py >= (3, 8) else "fail", f"{py.major}.{py.minor}.{py.micro}",
        "" if py >= (3, 8) else t("doctor.fix_python", lang))
    row("core", "ddoc", "ok", f"{__version__} · hookspec {HOOKSPEC_VERSION}")
    for mod in ("numpy", "pandas", "pluggy", "typer", "jinja2"):
        err = _try_import(mod)
        row("core", mod, "ok" if err is None else "fail", err or "", t("doctor.fix_pip", lang, pkg=mod) if err else "")

    plugins = plugin_status()
    if not plugins:
        row("plugins", t("doctor.plugins", lang), "warn", t("doctor.no_plugins", lang), t("doctor.fix_plugins", lang))
    for p in plugins:
        row("plugins", p["name"], "ok" if p["error"] is None else "fail",
            p["version"] if p["error"] is None else f"{p['version']} — {p['error']}",
            t("doctor.fix_plugin_load", lang) if p["error"] else "")

    for item, mods, enables in (
        ("pdf", ["weasyprint"], "doctor.feat_pdf"),
        ("serve", ["fastapi", "uvicorn"], "doctor.feat_serve"),
        ("embeddings", ["torch"], "doctor.feat_embeddings"),
    ):
        errs = [f"{m}: {e}" for m in mods if (e := _try_import(m))]
        if item == "embeddings":
            # CLIP comes bundled with the vision and text plugins; one of them must load.
            carriers = [p["name"] for p in plugins if p["name"] in ("ddoc_vision", "ddoc_text") and not p["error"]]
            if not carriers:
                errs.append(t("doctor.no_embedding_plugin", lang))
        row("features", t(enables, lang), "ok" if not errs else "warn", "; ".join(errs),
            t(f"doctor.fix_{item}", lang) if errs else "")
    return rows


def doctor_command(
    json_out: bool = typer.Option(False, "--json", help="Emit the checks as one JSON object."),
    lang: Optional[str] = typer.Option(None, "--lang", help="en | ko (default: $DDOC_LANG, else en)."),
):
    """Check the environment: Python, core libraries, plugins (do they load?), optional features.

    Exit code 0 when the core works, 1 when something it needs is missing.
    """
    lang = resolve_lang(lang)
    rows = environment(lang)
    failed = [r for r in rows if r["status"] == "fail" and r["area"] == "core"]
    broken = [r for r in rows if r["status"] == "fail" and r["area"] == "plugins"]
    if json_out:
        sys.stdout.write(json.dumps({"ok": not failed, "checks": rows}, ensure_ascii=False) + "\n")
    else:
        area = None
        for r in rows:
            if r["area"] != area:
                area = r["area"]
                rprint(f"[bold]{t('doctor.area_' + area, lang)}[/bold]")
            line = f"  {_MARK[r['status']]} {r['item']}"
            if r["detail"]:
                line += f"  [dim]{r['detail']}[/dim]"
            rprint(line)
            if r["fix"]:
                rprint(f"      ⤷ {r['fix']}")
        summary = ("doctor.summary_fail" if failed else
                   "doctor.summary_plugins" if broken else "doctor.summary_ok")
        rprint("\n" + t(summary, lang, count=len(broken)))
    raise typer.Exit(code=1 if failed else 0)


# ── probe ─────────────────────────────────────────────────────────────


def _describe(path: Path) -> Dict[str, Any]:
    from .diff import sniff_modality

    if not path.exists():
        return {"path": str(path), "exists": False, "kind": None}
    kind, detail = sniff_modality(path)
    info: Dict[str, Any] = {"path": str(path), "exists": True, "kind": kind,
                            "counts": detail.get("counts"), "marker": detail.get("marker")}
    if kind == "csv":
        info["files"] = len(detail.get("files") or [])
    return info


def _csv_pair(ref_files: List[str], cur_files: List[str]) -> Dict[str, Any]:
    from .diff import _as_datetime, _load_csv_sampled

    # a sample is enough to find the columns and their kinds; the row counts are the totals
    (ref, ref_total), (cur, cur_total) = _load_csv_sampled(ref_files, 20_000), _load_csv_sampled(cur_files, 20_000)
    time_ref = {c for c in ref.columns if _as_datetime(ref[c]) is not None}
    time_cur = {c for c in cur.columns if _as_datetime(cur[c]) is not None}
    time = [str(c) for c in ref.columns if c in time_ref and c in time_cur]
    return {
        "rows": {"ref": ref_total, "cur": cur_total},
        "compared": [str(c) for c in ref.columns if c in cur.columns and str(c) not in time],
        "time": time,
        "only_ref": [str(c) for c in ref.columns if c not in cur.columns],
        "only_cur": [str(c) for c in cur.columns if c not in ref.columns],
    }


def probe(ref: Path, cur: Optional[Path], lang: str) -> Dict[str, Any]:
    """What `ddoc diff` would do with these paths (nothing is judged)."""
    from .diff import sniff_modality

    sides = [_describe(ref)] + ([_describe(cur)] if cur else [])
    loaded = {p["name"]: p["error"] is None for p in plugin_status()}
    out: Dict[str, Any] = {"paths": sides, "route": None, "blind": False, "reason": None, "hints": []}

    missing = [s["path"] for s in sides if not s["exists"]]
    if missing:
        out.update(blind=True, reason=t("probe.missing", lang, paths=", ".join(missing)))
        return out
    kinds = {s["kind"] for s in sides}
    if len(kinds) > 1:
        out.update(blind=True, reason=t("probe.mixed", lang, kinds=" / ".join(s["kind"] for s in sides)),
                   hints=[t("probe.hint_same_kind", lang)])
        return out
    kind = sides[0]["kind"]
    if kind == "unknown":
        out.update(blind=True, reason=t("probe.unknown", lang), hints=[t("hint.supported_inputs", lang)])
        return out
    if kind == "csv":
        out["route"] = t("probe.route_csv", lang)
        if cur:
            pair = _csv_pair(sniff_modality(ref)[1].get("files") or [str(ref)],
                             sniff_modality(cur)[1].get("files") or [str(cur)])
            out["csv"] = pair
            if not pair["compared"]:
                out.update(blind=True, reason=t("probe.no_shared", lang))
        return out
    plugin = KIND_PLUGIN.get(kind)
    if plugin and loaded.get(plugin):
        out["route"] = t("probe.route_plugin", lang, plugin=plugin)
    elif plugin and plugin in loaded:
        out.update(blind=True, reason=t("probe.plugin_broken", lang, plugin=plugin),
                   hints=[t("probe.hint_doctor", lang)])
    else:
        out.update(blind=True, reason=t("probe.no_plugin", lang, kind=kind),
                   hints=[t("hint.install_vision", lang) if kind == "vision"
                          else t("hint.install_plugin", lang, kind=kind)])
    return out


def probe_command(
    ref: Path = typer.Argument(..., help="Baseline dataset (directory or file)."),
    cur: Optional[Path] = typer.Argument(None, help="Current dataset (optional)."),
    json_out: bool = typer.Option(False, "--json", help="Emit one JSON object."),
    lang: Optional[str] = typer.Option(None, "--lang", help="en | ko (default: $DDOC_LANG, else en)."),
):
    """What would `ddoc diff` make of these paths? Kind of data, who judges it, BLIND or not — without judging.

    Exit code 0 when it can be judged, 2 when it would be BLIND.
    """
    lang = resolve_lang(lang)
    result = probe(ref, cur, lang)
    if json_out:
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
    else:
        for label, s in zip(("ref", "cur"), result["paths"]):
            if not s["exists"]:
                rprint(f"  {label}: {s['path']}  [red]{t('probe.not_found', lang)}[/red]")
                continue
            extra = (t("probe.files", lang, count=s["files"]) if s.get("files")
                     else (s.get("marker") or ", ".join(f"{k} {v}" for k, v in (s.get("counts") or {}).items())))
            rprint(f"  {label}: {s['path']}  → [bold]{s['kind']}[/bold]  [dim]{extra}[/dim]")
        pair = result.get("csv")
        if pair:
            rprint("  " + t("probe.csv_pair", lang, ref=f"{pair['rows']['ref']:,}", cur=f"{pair['rows']['cur']:,}",
                            compared=len(pair["compared"])))
            for key in ("only_ref", "only_cur", "time"):
                if pair[key]:
                    rprint("    · " + t(f"subject.{key}", lang, columns=", ".join(pair[key])))
        if result["blind"]:
            rprint(f"[yellow]◌ {t('probe.would_blind', lang)}[/yellow] — {result['reason']}")
        else:
            rprint(f"[green]✓ {t('probe.would_judge', lang)}[/green] — {result['route']}")
        for h in result["hints"]:
            rprint(f"   ⤷ {h}")
    raise typer.Exit(code=2 if result["blind"] else 0)

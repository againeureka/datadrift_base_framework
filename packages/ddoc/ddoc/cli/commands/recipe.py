"""``ddoc recipe`` — chain multiple ddoc steps via a YAML file.

Turns the GUI's one-shot dance (fetch → analyze →
report → export through 4 separate submits) into a single declarative
file. Useful for CI, scheduled jobs, and "I always run these 4 things
together" workflows.

``--narrate`` runs the same file for a reader: each step's title, note
and command line, then what that command prints when typed by hand.
``ddoc examples tour`` is built on it.
"""
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import typer
from rich import print as rprint

from ddoc.core.recipe import (
    Recipe, RecipeError, _STEP_KINDS, execute_recipe,
)


recipe_app = typer.Typer(help="Run a multi-step ddoc workflow defined in YAML.")


@recipe_app.command("validate")
def recipe_validate(
    path: Path = typer.Argument(..., help="Recipe YAML file."),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable envelope."),
):
    """Parse + validate a recipe without running any steps."""
    try:
        recipe = Recipe.load(path)
    except RecipeError as e:
        _emit_error(e, json_out=json_out)
        raise typer.Exit(code=2)
    issues = recipe.validate()
    body = {
        "status": "ok" if not issues else "error",
        "recipe": recipe.name or path.name,
        "step_count": len(recipe.steps),
        "issues": issues,
    }
    if json_out:
        sys.stdout.write(json.dumps(body, ensure_ascii=False, default=str) + "\n")
    else:
        if issues:
            rprint(f"[red]❌ {len(issues)} issue(s):[/red]")
            for i in issues:
                rprint(f"  • {i}")
        else:
            rprint(f"[green]✅ {recipe.name or path.name} — {len(recipe.steps)} step(s) OK[/green]")
    if issues:
        raise typer.Exit(code=2)


@recipe_app.command("run")
def recipe_run(
    path: Path = typer.Argument(..., help="Recipe YAML file."),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print the argv each step would run, don't actually execute.",
    ),
    json_out: bool = typer.Option(
        False, "--json",
        help="Machine-readable envelope to stdout. Otherwise pretty progress.",
    ),
    narrate: bool = typer.Option(
        False, "--narrate",
        help="For a reader: each step's title, note and command, then the command's own output. "
             "Steps can use earlier output paths, exit codes and verdicts, not JSON.",
    ),
    var: Optional[List[str]] = typer.Option(
        None, "--var", help="Set a recipe variable: --var ref=old.csv (repeatable).",
    ),
    lang: Optional[str] = typer.Option(
        None, "--lang", help="Language of the narration: en | ko (default: $DDOC_LANG, else en).",
    ),
):
    """Execute every step in the recipe in order.

    Stops at the first failing step and surfaces the error envelope.
    Each step's ``${steps.<id>.output}`` and ``${steps.<id>.json...}``
    are available to subsequent steps via the substitution machinery.
    """
    from ...i18n import resolve_lang

    try:
        recipe = Recipe.load(path)
        recipe.vars.update(_parse_vars(var or []))
    except RecipeError as e:
        _emit_error(e, json_out=json_out)
        raise typer.Exit(code=2)
    if narrate:
        if json_out or dry_run:
            _emit_error(RecipeError("--narrate runs the steps for a reader; "
                                    "it does not combine with --json or --dry-run"), json_out=json_out)
            raise typer.Exit(code=2)
        lang = resolve_lang(lang)
        import os
        os.environ["DDOC_LANG"] = lang  # the steps' own messages follow the narration
        try:
            result = narrate_recipe(recipe, lang=lang)
        except RecipeError as e:
            _emit_error(e, json_out=False)
            raise typer.Exit(code=2)
        print_ending(result, lang)
        raise typer.Exit(code=0 if result["status"] == "success" else 1)

    def _on_step(sr) -> None:
        if json_out:
            return
        if sr.skipped:
            rprint(f"[dim]→ {sr.id} ({sr.run}) — would run: ddoc {' '.join(sr.argv)}[/dim]")
        else:
            rprint(f"[cyan]✓ {sr.id} ({sr.run})[/cyan] [dim]({sr.elapsed_ms} ms)[/dim]")

    try:
        result = execute_recipe(recipe, dry_run=dry_run, on_step=_on_step)
    except RecipeError as e:
        _emit_error(e, json_out=json_out)
        raise typer.Exit(code=2)

    if json_out:
        sys.stdout.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
    else:
        if result["status"] == "error":
            rprint(f"[red]❌ recipe failed at step {result.get('failed_step')!r}: "
                   f"{result.get('error', {}).get('message')}[/red]")
        else:
            rprint(
                f"[green]✅ recipe '{result.get('recipe') or path.name}' "
                f"finished — {len(result['steps'])} step(s)[/green]"
            )

    if result["status"] != "success":
        raise typer.Exit(code=1)


def _parse_vars(pairs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise RecipeError(f"--var takes name=value, got {pair!r}", code="bad_var")
        out[key.strip()] = value
    return out


def _words(value: Any, lang: str) -> str:
    """A step's title or note: plain text, or {en: .., ko: ..}."""
    if isinstance(value, dict):
        return value.get(lang) or value.get("en") or next(iter(value.values()), "")
    return value or ""


def _leaf_steps(steps: List[Any]) -> Iterator[Dict[str, Any]]:
    for step in steps:
        if isinstance(step, dict) and "parallel" in step:
            yield from (c for c in step.get("parallel") or [] if isinstance(c, dict))
        elif isinstance(step, dict):
            yield step


def narrate_recipe(recipe: Recipe, *, lang: str, pause: bool = False) -> Dict[str, Any]:
    """Run ``recipe`` live for a reader. Before each step: its number,
    title, note and the command as one would type it; then whatever the
    command prints. ``pause`` waits for Enter before each command."""
    from rich.console import Console

    from ...i18n import t

    console = Console()
    leaves = {s.get("id"): s for s in _leaf_steps(recipe.steps)}
    order = list(leaves)

    def header(step_id: str) -> None:
        raw = leaves.get(step_id) or {}
        console.print()
        console.rule(f"[bold]{order.index(step_id) + 1}/{len(order)}[/bold]  "
                     f"{_words(raw.get('title'), lang) or step_id}", align="left", style="cyan")

    def on_start(raw: Dict[str, Any], argv: List[str]) -> None:
        header(raw["id"])
        note = _words(raw.get("note"), lang)
        if note:
            console.print(note, highlight=False, markup=False)
        console.print(f"[bold cyan]$ ddoc {shlex.join(argv)}[/bold cyan]", highlight=False)
        if pause and sys.stdin.isatty():
            console.input(f"[dim]{t('tour.pause', lang)}[/dim]")
        console.print()
        sys.stdout.flush()  # before the command writes to the same terminal

    def on_step(sr) -> None:
        if sr.skipped and sr.skipped_reason == "when":
            header(sr.id)
            console.print(f"[dim]{t('tour.skipped', lang, when=leaves[sr.id].get('when'))}[/dim]",
                          highlight=False)

    return execute_recipe(recipe, live=True, on_start=on_start, on_step=on_step)


_WRITES = ("diff", "analyze.eda", "report.render", "export.drift_report")


def print_ending(result: Dict[str, Any], lang: str, *, where: Optional[str] = None) -> None:
    """How many steps ran and which files they wrote, or where it stopped."""
    from rich.console import Console

    from ...i18n import t

    console = Console()
    steps = result.get("steps") or []
    skipped = sum(1 for s in steps if s.get("skipped"))
    console.print()
    console.rule(t("tour.end", lang), align="left", style="cyan")
    if result.get("status") != "success":
        message = (result.get("error") or {}).get("message", "")
        console.print(f"[red]❌ {t('tour.failed', lang, step=result.get('failed_step'), message=message)}[/red]")
        return
    console.print(f"✅ {t('tour.ran', lang, ran=len(steps) - skipped, total=len(steps), skipped=skipped)}")
    files = [s["output"] for s in steps if s.get("run") in _WRITES and not s.get("skipped") and s.get("output")]
    if files:
        import os
        near = os.path.relpath(where) if where else ""
        prefix = f"{(where if near.startswith('..') else near).rstrip('/')}/" if where else ""
        console.print(t("tour.files", lang, files=" · ".join(prefix + f for f in files)), highlight=False)


@recipe_app.command("kinds")
def recipe_kinds():
    """List the step kinds (`run` values) recipes can use."""
    rprint("[bold cyan]🍳 supported recipe step kinds[/bold cyan]\n")
    for kind, spec in _STEP_KINDS.items():
        argv = " ".join(spec["argv"])
        rprint(f"  [bold]{kind}[/bold]  → ddoc {argv}")
        opts = list(spec.get("options", {}).keys())
        if opts:
            rprint(f"     with: {', '.join(opts)}")


def _emit_error(e: RecipeError, *, json_out: bool) -> None:
    if json_out:
        sys.stdout.write(json.dumps(e.to_dict(), ensure_ascii=False, default=str) + "\n")
    else:
        rprint(f"[red]❌ {e}[/red]")

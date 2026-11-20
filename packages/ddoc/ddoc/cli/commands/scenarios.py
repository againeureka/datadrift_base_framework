"""``ddoc scenarios`` — find the command for a situation, and run it."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from typing import Dict, List, Optional

import typer
from rich import print as rprint
from rich.markup import escape

from ... import scenarios as sc
from ...i18n import resolve_lang, t

scenarios_app = typer.Typer(help="What to run, by situation: list, cheat sheet, run, menu",
                            invoke_without_command=True)


def _sets(pairs: List[str]) -> Dict[str, str]:
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise typer.BadParameter(f"--set expects name=value, got {pair!r}")
        name, value = pair.split("=", 1)
        out[name.strip()] = value
    return out


def _run(scenario: sc.Scenario, given: Dict[str, str], lang: str, dry: bool) -> int:
    try:
        cmds = sc.commands(scenario, given)
    except sc.CatalogError as e:
        rprint(f"[red]{scenario.id}: {e}[/red]")
        return 2
    code = 0
    env = dict(os.environ, DDOC_LANG=lang)  # the steps speak the same language
    for i, args in enumerate(cmds):
        rprint(f"[bold]$ ddoc {escape(shlex.join(args))}[/bold]")
        if dry:
            continue
        code = subprocess.call([sys.executable, "-m", "ddoc.cli.main", *args], env=env)
        if code != 0 and i < len(cmds) - 1:  # every step but the last must succeed
            rprint(f"[red]{t('scen.step_failed', lang, code=code)}[/red]")
            return code
    tips = [sc.fill(tip, scenario, given) for tip in scenario.tip_list(lang)]
    if tips and not dry:
        rprint("")
    for tip in tips:
        rprint(f"   ⤷ {tip}")
    return code


@scenarios_app.callback()
def scenarios_list(
    ctx: typer.Context,
    md: bool = typer.Option(False, "--md", help="Print the cheat sheet as Markdown."),
    json_out: bool = typer.Option(False, "--json", help="Emit the catalog as JSON."),
    lang: Optional[str] = typer.Option(None, "--lang", help="en | ko (default: $DDOC_LANG, else en)."),
):
    """List the scenarios by group (the default), or print them as a cheat sheet."""
    if ctx.invoked_subcommand is not None:
        return
    lang = resolve_lang(lang)
    if json_out:
        sys.stdout.write(json.dumps([sc.to_dict(s, lang) for s in sc.load()], ensure_ascii=False) + "\n")
        return
    if md:
        sys.stdout.write(sc.cheatsheet(lang) + "\n")
        return
    group = None
    for s in sc.load():
        if s.get("group", lang) != group:
            group = s.get("group", lang)
            rprint(f"\n[bold]{group}[/bold]")
        rprint(f"  [cyan]{s.id:<15}[/cyan] {s.get('title', lang)}  [dim]— {s.get('when', lang)}[/dim]")
    rprint(f"\n{t('scen.how', lang)}")


@scenarios_app.command("run")
def scenarios_run(
    scenario_id: str = typer.Argument(..., help="Scenario id (see `ddoc scenarios`)."),
    set_: List[str] = typer.Option([], "--set", help="name=value for a parameter (repeatable)."),
    dry_run: bool = typer.Option(False, "-n", "--dry-run", help="Print the commands; run nothing."),
    lang: Optional[str] = typer.Option(None, "--lang", help="en | ko (default: $DDOC_LANG, else en)."),
):
    """Run one scenario. Exit code is the last command's (diff: 0 OK, 1 DRIFT, 2 BLIND)."""
    lang = resolve_lang(lang)
    try:
        scenario = sc.find(scenario_id)
    except sc.CatalogError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(code=2)
    raise typer.Exit(code=_run(scenario, _sets(set_), lang, dry_run))


@scenarios_app.command("menu")
def scenarios_menu(
    lang: Optional[str] = typer.Option(None, "--lang", help="en | ko (default: $DDOC_LANG, else en)."),
):
    """Pick a scenario by number, fill in its values, check the command, run it."""
    lang = resolve_lang(lang)
    items = [s for s in sc.load() if s.run]
    group = None
    for i, s in enumerate(items, 1):
        if s.get("group", lang) != group:
            group = s.get("group", lang)
            rprint(f"\n[bold]{group}[/bold]")
        rprint(f"  {i:>2}. {s.get('title', lang)}  [dim]— {s.get('when', lang)}[/dim]")
    choice = typer.prompt("\n" + t("scen.pick", lang), default="", show_default=False).strip()
    if not choice.isdigit() or not 1 <= int(choice) <= len(items):
        raise typer.Exit(code=0)
    scenario = items[int(choice) - 1]
    given: Dict[str, str] = {}
    for p in scenario.params:
        label = f"{p.name} ({p.help.get(lang) or p.help['en']})"
        given[p.name] = typer.prompt(label, default=p.default or "", show_default=bool(p.default)).strip()
    try:
        cmds = sc.commands(scenario, given)
    except sc.CatalogError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(code=2)
    for args in cmds:
        rprint(f"[bold]$ ddoc {escape(shlex.join(args))}[/bold]")
    if not typer.confirm(t("scen.confirm", lang), default=True):
        raise typer.Exit(code=0)
    raise typer.Exit(code=_run(scenario, given, lang, dry=False))

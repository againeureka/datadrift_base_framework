"""``ddoc examples`` — generate ready-to-analyze toy datasets.

Exposes the :mod:`ddoc.examples.factories`
toy-data generators as a user-facing CLI so people learning ddoc can
get a first drift result in 5 minutes without producing real data:

    ddoc examples list
    ddoc examples generate weather --out /tmp/w --scenario monsoon
    ddoc diff /tmp/w/ref.csv /tmp/w/cur.csv
    ddoc examples tour weather          # the whole pipeline, step by step
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint


def _import_factories():
    """Return the toy-data factories module shipped with ddoc."""
    from ddoc.examples import factories
    return factories


examples_app = typer.Typer(help="Generate ready-to-analyze toy datasets for tutorials and quick smoke tests.")


@examples_app.command("list")
def list_examples():
    """List the supported (modality, scenario) combinations."""
    factories = _import_factories()
    rprint("[bold cyan]📚 ddoc example datasets[/bold cyan]\n")
    for name in factories.PAIR_BUILDERS:
        rprint(f"  • [bold]{name}[/bold]  scenarios: {', '.join(factories.scenarios_of(name))}")
    rprint("\n  shifted   — ref vs cur with a clear shift (weather: the rainy season)")
    rprint("  identical — the same data on both sides (drift ≈ 0; a sanity check)")
    rprint("  weather, dashcam, support and machine also take a story by name: a real weather-station")
    rprint("  log, a real dashcam drive, generated support tickets (text), generated machine sounds (audio).")
    rprint("\n[dim]💡 Try: ddoc examples generate weather --out /tmp/ex && "
           "ddoc diff /tmp/ex/ref.csv /tmp/ex/cur.csv — or the guided tour: ddoc examples tour[/dim]")


@examples_app.command("generate")
def generate_example(
    modality: str = typer.Argument(
        ..., help="One of: weather, dashcam, support, machine, timeseries, audio, text, vision, categorical "
                  "(run `ddoc examples list` to see)."
    ),
    out: Path = typer.Option(
        ..., "--out", "-o",
        help="Output directory: `ref.csv` and `cur.csv` for weather, `ref/` and `cur/` folders otherwise.",
    ),
    scenario: str = typer.Option(
        "shifted", "--scenario", "-s",
        help="`shifted` (default) for a drift scenario, `identical` for a sanity-check pair; "
             "weather, dashcam, support and machine also take a story name (`ddoc examples list`).",
    ),
):
    """Generate a (ref, cur) toy dataset pair for the given modality.

    Example:
        ddoc examples generate audio --out /tmp/audio --scenario shifted
        ddoc analyze drift --data-path-ref /tmp/audio/ref --data-path-cur /tmp/audio/cur --json
    """
    factories = _import_factories()
    if modality not in factories.PAIR_BUILDERS:
        rprint(f"[red]❌ unknown modality: {modality!r}[/red]")
        rprint(f"   supported: {', '.join(factories.PAIR_BUILDERS.keys())}")
        raise typer.Exit(code=2)
    if scenario not in factories.scenarios_of(modality):
        rprint(f"[red]❌ unknown scenario: {scenario!r}[/red]")
        rprint(f"   supported: {', '.join(factories.scenarios_of(modality))}")
        raise typer.Exit(code=2)

    from ...i18n import resolve_lang, t

    lang = resolve_lang(None)
    out.mkdir(parents=True, exist_ok=True)
    rprint(f"[cyan]📦 {t('examples.generating', lang, modality=modality, scenario=scenario, out=out)}[/cyan]")

    builder = factories.PAIR_BUILDERS[modality]
    try:
        ref_dir, cur_dir = builder(out, scenario=scenario)
    except ImportError as e:
        rprint(f"[red]❌ Missing dependency for {modality}: {e}[/red]")
        from ...plugin_install import plugin_command
        rprint(f"[yellow]   Install with: {plugin_command(modality)}[/yellow]")
        raise typer.Exit(code=1)
    except Exception as e:
        rprint(f"[red]❌ Failed to generate {modality}: {e}[/red]")
        raise typer.Exit(code=1)

    rprint(f"[green]✅ {t('examples.wrote', lang, side='ref', path=ref_dir)}[/green]")
    rprint(f"[green]✅ {t('examples.wrote', lang, side='cur', path=cur_dir)}[/green]")
    rprint(f"[dim]{t('examples.next', lang)}  ddoc diff {ref_dir} {cur_dir}[/dim]")


def _tours_dir() -> Path:
    from ddoc import examples
    return Path(examples.__file__).with_name("tours")


def _tour_info(path: Path) -> dict:
    import yaml
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("tour") or {}


@examples_app.command("tour")
def tour(
    name: Optional[str] = typer.Argument(None, help="Tour to run; leave out to list them."),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Working folder (default: ./ddoc-tour-<name>).",
    ),
    lang: Optional[str] = typer.Option(None, "--lang", help="en | ko (default: $DDOC_LANG, else en)."),
    pause: bool = typer.Option(False, "--pause", help="Wait for Enter before each step (for a live demo)."),
    save: Optional[Path] = typer.Option(
        None, "--save", help="Write the tour's recipe to this file and stop, to run it on your own data.",
    ),
):
    """A guided tour: one command runs a whole pipeline on real data,
    showing each step's purpose, the command and its output.

    Example:
        ddoc examples tour weather --lang ko
    """
    import os
    import shutil

    from ...core.recipe import Recipe, RecipeError
    from ...i18n import resolve_lang, t
    from .recipe import _words, narrate_recipe, print_ending

    from rich.console import Console

    console = Console()
    lang = resolve_lang(lang)
    tours = {p.stem: p for p in sorted(_tours_dir().glob("*.yaml"))}
    if name is None:
        console.print(f"[bold cyan]🧭 {t('tour.list', lang)}[/bold cyan]\n")
        for tour_name, path in tours.items():
            info = _tour_info(path)
            console.print(f"  • [bold]{tour_name}[/bold] — {_words(info.get('title'), lang)}")
            console.print(f"    {_words(info.get('about'), lang)}", highlight=False)
            if info.get("needs"):
                console.print(f"    [dim]{t('tour.needs', lang, needs=_words(info['needs'], lang))}[/dim]")
        console.print(f"\n[dim]ddoc examples tour {next(iter(tours), '<name>')} --lang {lang}[/dim]")
        return
    if name not in tours:
        console.print(f"[red]❌ {t('tour.unknown', lang, name=name, names=', '.join(tours))}[/red]")
        raise typer.Exit(code=2)
    path, info = tours[name], _tour_info(tours[name])
    own = info.get("own", "")
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, save)
        console.print(t("tour.saved", lang, path=save))
        console.print(f"  ddoc recipe run {save} --narrate {own}".rstrip(), highlight=False)
        return

    try:
        recipe = Recipe.load(path)
    except RecipeError as e:
        console.print(f"[red]❌ {e}[/red]")
        raise typer.Exit(code=2)
    recipe.vars["lang"] = lang
    os.environ["DDOC_LANG"] = lang  # the steps' own messages follow the tour
    out = out or Path(f"ddoc-tour-{name}")
    out.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold cyan]🧭 {_words(info.get('title'), lang)}[/bold cyan]")
    console.print(_words(info.get("about"), lang), highlight=False)
    console.print(f"[dim]{t('tour.folder', lang, out=out)}[/dim]")
    here = os.getcwd()
    os.chdir(out)
    try:
        result = narrate_recipe(recipe, lang=lang, pause=pause)
    except RecipeError as e:
        console.print(f"[red]❌ {e}[/red]")
        raise typer.Exit(code=2)
    finally:
        os.chdir(here)
    print_ending(result, lang, where=str(out))
    if result["status"] == "success":
        console.print(f"\n{t('tour.own', lang)}")
        console.print(f"  ddoc examples tour {name} --save my-check.yaml", highlight=False)
        console.print(f"  ddoc recipe run my-check.yaml --narrate {own}".rstrip(), highlight=False)
    raise typer.Exit(code=0 if result["status"] == "success" else 1)

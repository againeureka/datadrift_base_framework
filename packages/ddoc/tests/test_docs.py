"""The parts of the docs that come from code stay in step with it."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import typer

from ddoc.cli.main import app

PKG = Path(__file__).resolve().parent.parent


def _docs_script():
    spec = importlib.util.spec_from_file_location("ddoc_docs_script", PKG / "scripts" / "docs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generated_blocks_are_current():
    assert _docs_script().stale() == [], "run: python packages/ddoc/scripts/docs.py build"


def _sections(text: str):
    """`## \\`ddoc a b <arg>\\`` headings → (command words, section text)."""
    parts = re.split(r"^## `ddoc ([^`]*)`\s*$", text, flags=re.M)
    for i in range(1, len(parts), 2):
        words = [w for w in parts[i].split() if not w.startswith("<")]
        yield words, parts[i + 1]


def test_every_option_of_a_documented_command_is_in_its_section():
    root = typer.main.get_command(app)
    text = (PKG / "docs" / "reference" / "commands.md").read_text(encoding="utf-8")
    missing, checked = [], 0
    for words, body in _sections(text):
        cmd = root
        for w in words:
            cmd = cmd.commands[w]
        # typer ships its own copy of click, so test by shape, not by class
        if hasattr(cmd, "commands"):
            continue  # a group's page lists its subcommands, not their options
        for param in cmd.params:
            longs = [o for o in getattr(param, "opts", []) if o.startswith("--")]
            if param.param_type_name != "option" or not longs or longs[0] == "--help":
                continue
            checked += 1
            if not any(o in body for o in longs):
                missing.append(f"ddoc {' '.join(words)}: {longs[0]}")
    assert checked > 30  # the check itself must be looking at something
    assert missing == [], missing

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
        words = [w for w in parts[i].split() if not w.startswith(("<", "["))]
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


# ── Usage guides (Korean and English) ─────────────────────────────────

GUIDES = {"ko": PKG / "docs" / "guides" / "usage.md", "en": PKG / "docs" / "guides" / "usage.en.md"}
RUN_SH = PKG.parent.parent / "run.sh"


def _shell_commands(path: Path):
    """`ddoc …` and `./run.sh …` lines of the ```bash blocks, as words
    (comments, pipes and redirections cut off, leading VAR=value dropped)."""
    import shlex

    text = path.read_text(encoding="utf-8")
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.S):
        for line in block.replace("\\\n", " ").splitlines():
            line = re.split(r"\s+#\s", " " + line)[0]
            line = re.split(r"\s[|>]\s", line)[0].strip()
            words = shlex.split(line) if line else []
            while words and re.match(r"^[A-Z_]+=", words[0]):
                words.pop(0)
            if words and words[0] in ("ddoc", "./run.sh"):
                yield words


def _signature(words, root, run_sh_commands):
    """(command path, options) of one line; fails if either does not exist."""
    if words[0] == "./run.sh":
        assert words[1] in run_sh_commands, f"./run.sh has no `{words[1]}`"
        return ("./run.sh", words[1]), ()
    cmd, rest, path = root, words[1:], []
    while rest and hasattr(cmd, "commands") and rest[0] in cmd.commands:
        path.append(rest[0])
        cmd = cmd.commands[rest.pop(0)]
    known = {o for p in cmd.params for o in getattr(p, "opts", [])}
    opts = sorted({w.split("=", 1)[0] for w in rest if w.startswith("-")})
    unknown = [o for o in opts if o not in known]
    assert not unknown, f"`{' '.join(words)}`: no option {unknown}"
    return ("ddoc", *path), tuple(o for o in opts if o != "--lang")


def test_usage_guides_use_real_commands_and_say_the_same_in_both_languages():
    root = typer.main.get_command(app)
    run_sh = set(re.findall(r"^\s+([a-z]+)[|)]", RUN_SH.read_text(encoding="utf-8"), flags=re.M))
    sigs = {lang: [_signature(w, root, run_sh) for w in _shell_commands(p)] for lang, p in GUIDES.items()}
    assert len(sigs["ko"]) > 20  # the check itself must be looking at something
    assert sigs["ko"] == sigs["en"], "the Korean and English guides show different commands"


def test_usage_guide_links_resolve():
    for path in GUIDES.values():
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            assert (path.parent / target.split("#", 1)[0]).exists(), f"{path.name}: broken link {target}"

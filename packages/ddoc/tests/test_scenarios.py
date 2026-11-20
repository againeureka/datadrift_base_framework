"""The scenario catalog — one file for CLI, menu, GUI and cheat sheet."""
from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner

from ddoc import scenarios as sc
from ddoc.cli.main import app

runner = CliRunner()


def test_catalog_loads_and_is_bilingual():
    items = sc.load()
    assert len(items) >= 10 and len({s.id for s in items}) == len(items)
    for s in items:
        for key in ("group", "title", "when"):
            assert s.text[key].get("en") and s.text[key].get("ko"), (s.id, key)
        assert len(s.tips.get("en", [])) == len(s.tips.get("ko", [])), s.id
        assert s.run or s.tips, s.id


def _walk(root, args):
    """The click command that `ddoc <args>` reaches, and the args left over."""
    cmd, rest = root, list(args)
    while rest and hasattr(cmd, "commands") and rest[0] in cmd.commands:
        cmd = cmd.commands[rest.pop(0)]
    return cmd, rest


def test_every_command_and_option_in_the_catalog_exists():
    """A renamed option or command must not leave the catalog pointing nowhere."""
    root = typer.main.get_command(app)
    for s in sc.load():
        vals = {p.name: p.default or f"<{p.name}>" for p in s.params}
        for line in s.run:
            cmd, rest = _walk(root, sc.expand(line, vals))
            assert not hasattr(cmd, "commands"), (s.id, line)  # reached a real command
            known = {o for p in cmd.params for o in getattr(p, "opts", [])}
            for arg in rest:
                if arg.startswith("-"):
                    assert arg in known, (s.id, arg)


def test_gui_tabs_and_their_fields_exist():
    js = (sc.CATALOG.parent / "server" / "static" / "app.js").read_text(encoding="utf-8")
    for s in sc.load():
        if s.gui:
            assert f"    {s.gui}: {{" in js, (s.id, s.gui)


def test_expand_keeps_values_whole_and_drops_empty_brackets():
    s = sc.find("compare")
    assert sc.commands(s, {"ref": "a b.csv", "cur": "c.csv"}) == [
        ["diff", "a b.csv", "c.csv", "--report", "note.html"]]
    assert sc.commands(s, {"ref": "a", "cur": "c", "issuer": "Lab X"})[0][-2:] == ["--issuer", "Lab X"]
    with pytest.raises(sc.CatalogError, match="ref"):
        sc.commands(s, {"cur": "c"})


def test_cli_lists_prints_and_dry_runs():
    res = runner.invoke(app, ["scenarios", "--json", "--lang", "ko"])
    body = json.loads(res.output)
    assert any(x["id"] == "compare" and x["gui"] == "diff" for x in body)
    md = runner.invoke(app, ["scenarios", "--md", "--lang", "en"]).output
    assert "### Compare two datasets" in md and "`ddoc diff <ref> <cur> --report note.html`" in md
    res = runner.invoke(app, ["scenarios", "run", "compare", "--set", "ref=a.csv", "--set", "cur=b.csv", "-n"])
    assert res.exit_code == 0 and "ddoc diff a.csv b.csv --report note.html" in res.output
    res = runner.invoke(app, ["scenarios", "run", "compare"])
    assert res.exit_code == 2 and "--set ref=" in res.output
    assert runner.invoke(app, ["scenarios", "run", "nope"]).exit_code == 2


def test_run_executes_every_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ref = tmp_path / "r.csv"
    ref.write_text("a\n" + "\n".join(str(i % 3) for i in range(60)) + "\n")
    res = runner.invoke(app, ["scenarios", "run", "pr-comment", "--set", f"ref={ref}", "--set", f"cur={ref}",
                              "--set", "out=x.md"])
    assert res.exit_code == 0, res.output  # identical data: OK
    assert (tmp_path / "x.md").read_text(encoding="utf-8").startswith("# ")


def test_server_lists_the_catalog():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ddoc.server.app import create_app

    client = TestClient(create_app(bind_info="testclient:0"))
    r = client.get("/scenarios", params={"lang": "ko"})
    assert r.status_code == 200
    first = r.json()[0]
    assert first["id"] == "check-env" and first["commands"] == ["ddoc doctor"]
    assert "scenarios" in client.get("/").text

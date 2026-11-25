"""Guided tours and what they stand on: the diff/probe recipe steps, a
verdict that steers later steps, narrated (live) runs, the weather
example and the short profile summary."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ddoc.cli.main import app
from ddoc.core.recipe import Recipe, RecipeError, _step_to_argv, execute_recipe
from ddoc.examples import factories

runner = CliRunner()
TOURS = Path(factories.__file__).with_name("tours")


def _recipe(tmp_path: Path, body: str) -> Recipe:
    p = tmp_path / "recipe.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return Recipe.load(p)


# ── Recipe steps ───────────────────────────────────────────────────────


def test_diff_and_probe_steps_build_the_commands_one_would_type():
    assert _step_to_argv("diff", {"ref": "a.csv", "cur": "b.csv", "report": "n.html", "lang": "ko"},
                         json_flag=False) == ["diff", "a.csv", "b.csv", "--report", "n.html", "--lang", "ko"]
    assert _step_to_argv("probe", {"ref": "a.csv"}) == ["probe", "a.csv", "--json"]
    # an empty variable drops its option, so a command falls back to its default
    assert _step_to_argv("analyze.eda", {"data_path": "a.csv", "lang": ""}, json_flag=False) == \
        ["analyze", "eda", "--data-path", "a.csv"]


def test_a_drift_verdict_is_an_answer_and_steers_later_steps(tmp_path):
    ref, cur = factories.make_pair_weather(tmp_path / "data", scenario="monsoon")
    same, _ = factories.make_pair_weather(tmp_path / "same", scenario="identical")
    recipe = _recipe(tmp_path, f"""
        name: t
        workspace: {tmp_path / 'ws'}
        steps:
          - id: judge
            run: diff
            with: {{ref: "{ref}", cur: "{cur}"}}
          - id: on-drift
            when: '${{steps.judge.verdict}} == "DRIFT"'
            run: diff
            with: {{ref: "{same}", cur: "{same}"}}
          - id: on-ok
            when: '${{steps.on-drift.returncode}} == 1'
            run: probe
            with: {{ref: "{ref}"}}
    """)
    result = execute_recipe(recipe)
    assert result["status"] == "success"
    by = {s["id"]: s for s in result["steps"]}
    assert (by["judge"]["returncode"], by["judge"]["verdict"]) == (1, "DRIFT")
    assert by["on-drift"]["verdict"] == "OK" and not by["on-drift"]["skipped"]
    assert by["on-ok"]["skipped"] and by["on-ok"]["skipped_reason"] == "when"


def test_step_words_must_be_text_or_per_language(tmp_path):
    recipe = _recipe(tmp_path, """
        steps:
          - id: a
            run: probe
            title: {en: Look, ko: 보기}
            note: plain text is fine
            with: {ref: x}
          - id: b
            run: probe
            note: [not, words]
            with: {ref: x}
    """)
    issues = recipe.validate()
    assert len(issues) == 1 and "b" in issues[0] and "note" in issues[0]


def test_a_live_run_keeps_no_envelopes(tmp_path, monkeypatch):
    ref, cur = factories.make_pair_weather(tmp_path, scenario="monsoon")
    recipe = _recipe(tmp_path, f"""
        steps:
          - id: judge
            run: diff
            with: {{ref: "{ref}", cur: "{cur}"}}
    """)
    started = []
    result = execute_recipe(recipe, live=True, on_start=lambda raw, argv: started.append(argv))
    assert result["status"] == "success"
    assert started == [["diff", str(ref), str(cur)]]          # as typed: no --json
    assert result["steps"][0]["verdict"] == "DRIFT" and result["steps"][0]["json"] is None
    assert not (tmp_path / ".ddoc-recipe-out").exists()


def test_recipe_run_takes_variables(tmp_path):
    recipe = tmp_path / "r.yaml"
    recipe.write_text("vars: {ref: none}\nsteps:\n  - id: look\n    run: probe\n    with: {ref: '${vars.ref}'}\n")
    res = runner.invoke(app, ["recipe", "run", str(recipe), "--dry-run", "--var", "ref=elsewhere.csv"])
    assert res.exit_code == 0 and "probe elsewhere.csv" in res.output
    bad = runner.invoke(app, ["recipe", "run", str(recipe), "--var", "no-equals"])
    assert bad.exit_code == 2


# ── Example data and the profile summary ──────────────────────────────


def test_weather_example_writes_two_csvs_for_any_story(tmp_path):
    res = runner.invoke(app, ["examples", "generate", "weather", "--out", str(tmp_path), "--scenario", "season"])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "ref.csv").is_file() and (tmp_path / "cur.csv").is_file()
    assert factories.pair_paths("weather", tmp_path) == (tmp_path / "ref.csv", tmp_path / "cur.csv")
    assert "yearly-reset" in factories.scenarios_of("weather")
    assert "season" not in factories.scenarios_of("vision")
    bad = runner.invoke(app, ["examples", "generate", "weather", "--out", str(tmp_path), "--scenario", "nope"])
    assert bad.exit_code == 2


def test_profile_prints_a_summary_not_the_whole_envelope(tmp_path):
    _, cur = factories.make_pair_weather(tmp_path, scenario="monsoon")
    res = runner.invoke(app, ["analyze", "eda", "--data-path", str(cur), "--quiet", "--lang", "en"])
    assert res.exit_code == 0, res.output
    assert "2,016 rows · 17 columns" in res.output and "period 2026-07-05" in res.output
    assert '"profile"' not in res.output and len(res.output.splitlines()) < 15


# ── Tours ──────────────────────────────────────────────────────────────


def test_every_tour_is_a_valid_recipe_with_words_in_both_languages():
    tours = sorted(TOURS.glob("*.yaml"))
    assert tours
    for path in tours:
        recipe = Recipe.load(path)
        assert recipe.validate() == []
        for step in recipe.steps:
            for key in ("title", "note"):
                assert set(step[key]) == {"en", "ko"}, (path.name, step["id"], key)


def test_tour_lists_and_saves(tmp_path):
    listed = runner.invoke(app, ["examples", "tour", "--lang", "en"])
    assert listed.exit_code == 0 and "weather" in listed.output
    saved = tmp_path / "mine.yaml"
    res = runner.invoke(app, ["examples", "tour", "weather", "--save", str(saved)])
    assert res.exit_code == 0 and saved.read_bytes() == (TOURS / "weather.yaml").read_bytes()
    assert runner.invoke(app, ["examples", "tour", "nope"]).exit_code == 2


def test_weather_tour_runs_from_start_to_finish(tmp_path):
    # a real run: every step is its own ddoc process writing to this terminal
    env = {**os.environ, "DDOC_LANG": "en"}
    proc = subprocess.run([sys.executable, "-m", "ddoc.cli.main", "examples", "tour", "weather",
                           "--out", str(tmp_path / "t")], capture_output=True, text=True, env=env,
                          cwd=tmp_path, timeout=300)
    out = proc.stdout
    assert proc.returncode == 0, out + proc.stderr
    for line in ("1/7", "$ ddoc probe ref.csv cur.csv", "DRIFT", "$ ddoc diff reset/ref.csv reset/cur.csv",
                 "7 of 7 steps ran"):
        assert line in out, line
    assert out.index("2/7") < out.index("can be judged") < out.index("3/7")   # in order
    for name in ("note.html", "note.md", "profile.html", "reset.html"):
        assert (tmp_path / "t" / name).is_file(), name
    assert not (tmp_path / "t" / ".ddoc-recipe-out").exists()

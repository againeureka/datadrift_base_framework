"""Drift Zoo — 각 케이스는 튜토리얼이자 회귀 테스트다.

세 케이스가 세 가지 다른 판정(DRIFT high / DRIFT / OK)을 내는 것까지
고정한다: 모든 것을 DRIFT라 우기는 탐지기가 되지 않도록. weather_station 은
실제 기상관측소 기록이라, 실데이터에서 판정과 판독이 어떻게 나오는지도 고정한다.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ddoc.cli.main import app

ZOO = Path(__file__).parent.parent / "zoo"
runner = CliRunner()


def _make(case: str, out_dir: Path, *args):
    spec = importlib.util.spec_from_file_location(
        f"zoo_{case}_make", ZOO / case / "make.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make(out_dir, *args)


@pytest.mark.parametrize("case,verdict,exit_code", [
    ("regime_shift", "DRIFT", 1),
    ("creeping_degradation", "DRIFT", 1),
    ("rhythm_not_drift", "OK", 0),
    ("weather_station", "DRIFT", 1),
])
def test_zoo_case_verdict(tmp_path, case, verdict, exit_code):
    ref, cur = _make(case, tmp_path)
    result = runner.invoke(app, ["diff", str(ref), str(cur), "--json"])
    assert result.exit_code == exit_code, result.output
    payload = json.loads(result.stdout.strip())
    assert payload["verdict"] == verdict


def test_zoo_regime_shift_report_evidence(tmp_path):
    ref, cur = _make("regime_shift", tmp_path)
    note = tmp_path / "note.html"
    runner.invoke(app, ["diff", str(ref), str(cur), "--json",
                        "--report", str(note)])
    body = note.read_text(encoding="utf-8")
    assert "price" in body and "channel" in body and "Findings by attribute" in body


@pytest.mark.parametrize("case,expected", [
    # the whole world moved: price shifts and spreads, the channel mix flips
    ("regime_shift", {"regime_shift", "shift_and_spread", "mix_shift"}),
    # a sensor drifting up and getting noisier
    ("creeping_degradation", {"shift_and_spread"}),
    # a daily rhythm is not drift
    ("rhythm_not_drift", {"stable"}),
    # the rainy season: dew point rises and narrows, humidity steps up
    ("weather_station", {"shift_and_narrow", "level_shift", "shift_and_spread"}),
])
def test_zoo_case_reads_as_its_story(tmp_path, case, expected):
    ref, cur = _make(case, tmp_path)
    result = runner.invoke(app, ["diff", str(ref), str(cur), "--json"])
    cards = json.loads(result.stdout.strip())["diagnosis"]["archetypes"]
    assert {c["code"] for c in cards if c["match"] == "match"} == expected


@pytest.mark.parametrize("story,verdict,top", [
    ("monsoon", "DRIFT", "dew_point_c"),
    ("yearly-reset", "OK", None),        # the reset total is set aside, the weather held
    ("season", "DRIFT", "dew_point_c"),
    ("next-days", "DRIFT", "pressure_inhg"),  # weather moves on a scale of days
])
def test_weather_station_stories(tmp_path, story, verdict, top):
    ref, cur = _make("weather_station", tmp_path, story)
    payload = json.loads(runner.invoke(app, ["diff", str(ref), str(cur), "--json"]).stdout.strip())
    assert payload["verdict"] == verdict
    if top:
        assert payload["top_attributes"][0][0] == top
    running = payload["raw"]["profile"]["columns"]["running"]
    rain_totals = ["rain_event_mm", "rain_daily_mm", "rain_weekly_mm", "rain_monthly_mm", "rain_yearly_mm"]
    # in four dry days the totals barely move, so nothing gives them away
    assert running == (rain_totals if story != "next-days" else [])
    assert "rain_hourly_mm" not in running   # a rolling sum, not a running total

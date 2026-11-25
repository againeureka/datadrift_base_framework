"""Running totals and sequences are set aside like dates, never by accident."""
from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from ddoc import archetypes as ar
from ddoc.cli.commands.diff import _running_kind, csv_diff
from ddoc.cli.main import app

runner = CliRunner()
rng = np.random.default_rng(7)


def _daily_total(days: int, per_day: int = 24) -> np.ndarray:
    """Rain since midnight: grows on wet days, starts over each day."""
    out = []
    for _ in range(days):
        total = 0.0
        for _ in range(per_day):
            if rng.random() < 0.3:
                total += round(float(rng.uniform(0.2, 3.0)), 1)
            out.append(total)
    return np.array(out)


def _hour_of_day(days: int) -> np.ndarray:
    return np.array([h for _ in range(days) for h in range(24) for _ in range(3)], dtype=float)


def _rolling(n: int) -> np.ndarray:
    """Rain in the last hour: rises and falls back step by step."""
    tips = (rng.random(n) < 0.2) * 0.3
    return np.round(np.convolve(tips, np.ones(6))[:n], 1)


@pytest.mark.parametrize("values,kind", [
    (_daily_total(10), "grows"),
    (np.cumsum(rng.uniform(0, 1, 200)), "grows"),             # a lifetime total
    (np.arange(1000, 1200, dtype=float), "grows"),            # an id or sequence number
    (np.full(100, 277.9), "flat"),                            # a monthly total in a dry fortnight
    (_hour_of_day(10), None),                                 # cyclic: same peak before each reset
    (_rolling(500), None),                                    # rolling sum: falls back gradually
    (rng.normal(20, 3, 300), None),                           # a measurement
    (np.arange(-50, 150, dtype=float), None),                 # negative values: not a count
])
def test_kind(values, kind):
    assert _running_kind(values) == kind


def _write(path, header, rows):
    path.write_text("\n".join([header, *(",".join(str(x) for x in r) for r in rows)]) + "\n")


def test_running_totals_are_set_aside_and_named(tmp_path):
    days = 14
    times = [f"2026-06-{1 + i // 24:02d} {i % 24:02d}:00" for i in range(days * 24)]
    temp_r, temp_c = rng.normal(20, 2, days * 24), rng.normal(20, 2, days * 24)
    year_r = 1200 + np.cumsum(rng.uniform(0, 0.4, days * 24))
    year_c = np.cumsum(rng.uniform(0, 0.4, days * 24))        # the yearly total started over
    hour = [i % 24 for i in range(days * 24)]
    _write(tmp_path / "r.csv", "time,temp,rain_year,hour",
           zip(times, temp_r.round(2), year_r.round(1), hour))
    _write(tmp_path / "c.csv", "time,temp,rain_year,hour",
           zip(times, temp_c.round(2), year_c.round(1), hour))
    res = csv_diff([str(tmp_path / "r.csv")], [str(tmp_path / "c.csv")])
    cols = res["profile"]["columns"]
    assert cols["running"] == ["rain_year"] and cols["compared"] == ["temp", "hour"]
    assert "rain_year" not in res["attribute_drifts"]
    kinds = {c["name"]: c["type"] for c in res["profile"]["cur"]["columns"]}
    assert kinds["rain_year"] == "running"

    note = tmp_path / "n.html"
    out = runner.invoke(app, ["diff", str(tmp_path / "r.csv"), str(tmp_path / "c.csv"),
                              "--json", "--lang", "ko", "--report", str(note)])
    assert json.loads(out.output.strip().splitlines()[-1])["verdict"] == "OK"
    body = note.read_text(encoding="utf-8")
    assert "rain_year: 누적·순번 열" in body and "누적·순번" in body


def test_without_a_time_column_file_order_says_nothing(tmp_path):
    # a file sorted by price must not make price look like a running total
    price_r = np.sort(rng.normal(100, 10, 200)).round(2)
    price_c = np.sort(rng.normal(130, 10, 200)).round(2)
    _write(tmp_path / "r.csv", "price", [[p] for p in price_r])
    _write(tmp_path / "c.csv", "price", [[p] for p in price_c])
    res = csv_diff([str(tmp_path / "r.csv")], [str(tmp_path / "c.csv")])
    assert res["profile"]["columns"]["running"] == [] and "price" in res["attribute_drifts"]


def test_probe_names_running_totals(tmp_path):
    times = [f"2026-06-{1 + i // 24:02d} {i % 24:02d}:00" for i in range(240)]
    _write(tmp_path / "r.csv", "time,total", zip(times, np.cumsum(rng.uniform(0, 1, 240)).round(1)))
    _write(tmp_path / "c.csv", "time,total", zip(times, (500 + np.cumsum(rng.uniform(0, 1, 240))).round(1)))
    out = runner.invoke(app, ["probe", str(tmp_path / "r.csv"), str(tmp_path / "c.csv"), "--lang", "en"])
    assert "total: running total or sequence" in out.output


def test_stuck_on_the_usual_value_says_so(tmp_path):
    # a dry week: rain stuck on 0, which was already the baseline's usual value
    times = [f"2026-07-{1 + i // 24:02d} {i % 24:02d}:00" for i in range(168)]
    wet = [round(float(rng.uniform(0.5, 6)), 1) if rng.random() < 0.3 else 0.0 for _ in times]
    _write(tmp_path / "r.csv", "time,rain_hour", zip(times, wet))
    _write(tmp_path / "c.csv", "time,rain_hour", zip(times, [0.0] * len(times)))
    out = runner.invoke(app, ["diff", str(tmp_path / "r.csv"), str(tmp_path / "c.csv"), "--json"])
    cards = json.loads(out.output.strip().splitlines()[-1])["diagnosis"]["archetypes"]
    stuck = next(c for c in cards if c["code"] == "stuck_values")
    text = ar.card_text(stuck, "ko")
    assert "기준에서도 가장 흔한 값" in text["signals"][0]
    assert any("비가 오지 않은 주" in c for c in text["causes"])

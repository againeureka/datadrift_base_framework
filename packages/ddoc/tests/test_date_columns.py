"""Date/time text columns are found by name or by shape, never by accident."""
from __future__ import annotations

import pandas as pd
import pytest

from ddoc.cli.commands.diff import _as_datetime, csv_diff


def _col(name, values):
    return pd.Series(values, name=name)


@pytest.mark.parametrize("name,values", [
    ("when", [f"2026-07-{1 + i % 28:02d}" for i in range(50)]),
    ("logged", [f"2026-07-{1 + i % 28:02d}T10:{i % 60:02d}:00Z" for i in range(50)]),
    ("visit", [f"07/{1 + i % 28:02d}/2026" for i in range(50)]),
    ("day", [f"2026/07/{1 + i % 28:02d} 09:30" for i in range(50)]),
    ("order_date", ["2026-07-01"] * 49 + ["not a date"]),     # by name: 90 % must parse
])
def test_found(name, values):
    assert _as_datetime(_col(name, values)) is not None


@pytest.mark.parametrize("name,values", [
    ("code", [f"{i % 9}-{i % 7}" for i in range(50)]),           # would parse as month-day
    ("channel", ["store", "online", "app"] * 17),
    ("event", [f"2026-07-{1 + i % 28:02d}" for i in range(40)] + ["n/a"] * 10),  # only 80 % date-shaped
    ("version", [f"1.{i}.0" for i in range(50)]),
])
def test_not_found(name, values):
    assert _as_datetime(_col(name, values)) is None


def test_numbers_stay_numeric():
    assert _as_datetime(_col("timestamp", list(range(50)))) is None


def test_a_date_column_without_a_date_name_gives_the_period(tmp_path):
    ref, cur = tmp_path / "r.csv", tmp_path / "c.csv"
    ref.write_text("when,x\n" + "\n".join(f"2026-01-{1 + i % 28:02d},{i % 5}" for i in range(60)) + "\n")
    cur.write_text("when,x\n" + "\n".join(f"2026-07-{1 + i % 28:02d},{i % 5}" for i in range(60)) + "\n")
    result = csv_diff([str(ref)], [str(cur)])
    assert result["profile"]["columns"]["time"] == ["when"]
    assert "when" not in result["attribute_drifts"]          # two periods always differ
    assert result["profile"]["cur"]["period"]["start"] == "2026-07-01"

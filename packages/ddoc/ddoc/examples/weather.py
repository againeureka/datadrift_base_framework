"""A real weather-station log, for examples that should feel like real data.

Outdoor readings in ten-minute steps from one home weather station,
2026-04-23 to 2026-08-31: temperature, humidity, dew point, wind, pressure
(inHg, as recorded), light (lux), UV index and rain. The rain columns are the
station's own running figures: rain in the last hour, and totals for the
current rain event, day, week, month and year (the yearly total was reset
on 2026-06-01). August has the logger's gaps, left as missing time.
``zoo/weather_station/prepare.py`` made the file from the station's export.

    from ddoc.examples import weather
    ref, cur = weather.write_pair("monsoon", out_dir)
"""
from __future__ import annotations

import gzip
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

DATA = Path(__file__).with_name("data") / "weather_station.csv.gz"

# Pairs of [start, end) days. Each tells one story about real data.
STORIES: Dict[str, Tuple[Tuple[str, str], Tuple[str, str]]] = {
    # the rainy season set in: humid, wet, dimmer days (a real change)
    "monsoon": (("2026-06-21", "2026-07-05"), ("2026-07-05", "2026-07-19")),
    # the station's yearly rain total started over; the weather did not change
    "yearly-reset": (("2026-05-18", "2026-06-01"), ("2026-06-01", "2026-06-15")),
    # late spring against high summer
    "season": (("2026-05-01", "2026-05-15"), ("2026-07-24", "2026-08-07")),
    # four days against the next four: weather moves on a scale of days
    "next-days": (("2026-07-24", "2026-07-28"), ("2026-07-28", "2026-08-01")),
}

# Measurements, as opposed to the running rain totals (useful where a tool
# would otherwise compare those totals as levels).
MEASUREMENTS = ["temp_c", "humidity", "dew_point_c", "feels_like_c", "wind_ms", "gust_ms",
                "wind_dir_deg", "pressure_inhg", "solar_lux", "uvi", "rain_hourly_mm"]


@lru_cache(maxsize=1)
def _table() -> Tuple[List[str], Tuple[str, ...]]:
    with gzip.open(DATA, "rt", encoding="utf-8") as fh:
        header, *rows = fh.read().splitlines()
    return header.split(","), tuple(rows)


def header() -> List[str]:
    return list(_table()[0])


def window(start: str, end: str, columns: Optional[Sequence[str]] = None) -> List[str]:
    """CSV lines (no header) with ``start <= time < end`` (ISO dates or
    ``YYYY-MM-DD HH:MM``), optionally only ``columns`` (``time`` first)."""
    names, rows = _table()
    picked = [r for r in rows if start <= r[:16] < end]
    if columns is None:
        return picked
    idx = [names.index("time"), *(names.index(c) for c in columns if c != "time")]
    return [",".join(cells[i] for i in idx) for cells in (r.split(",") for r in picked)]


def write_window(path: Path, start: str, end: str, columns: Optional[Sequence[str]] = None) -> Path:
    names = ["time", *(c for c in columns if c != "time")] if columns else header()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([",".join(names), *window(start, end, columns)]) + "\n", encoding="utf-8")
    return path


def write_pair(story: str, out_dir: Path, columns: Optional[Sequence[str]] = None) -> Tuple[Path, Path]:
    """Write ``ref.csv`` and ``cur.csv`` of a story under ``out_dir``."""
    (r0, r1), (c0, c1) = STORIES[story]
    out_dir = Path(out_dir)
    return (write_window(out_dir / "ref.csv", r0, r1, columns),
            write_window(out_dir / "cur.csv", c0, c1, columns))

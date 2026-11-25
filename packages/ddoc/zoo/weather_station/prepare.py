"""Build the shipped weather-station series from the station's own export.

The raw file is the backup CSV a home weather station (Ecowitt-style console)
writes once a minute. It is not in the repository; this script documents how
``ddoc/examples/data/weather_station.csv.gz`` was made from it:

* outdoor readings only (indoor temperature and humidity describe a home);
* 10-minute steps: means for measurements, the peak for gusts, a circular
  mean for wind direction, and the last value for rain totals and the
  rolling hourly rain, which are running figures;
* units as the station recorded them (pressure in inHg, light in lux);
* steps with fewer than five readings are dropped, so the logger's gaps in
  August stay visible as missing time rather than invented values.

    python prepare.py /path/to/Backup-A.CSV
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "ddoc" / "examples" / "data" / "weather_station.csv.gz"
START, END = "2026-04-23", "2026-09-01"     # whole days, before the logger stopped

MEAN = {"Outdoor Temperature(℃)": ("temp_c", 1), "Outdoor Humidity(%)": ("humidity", 0),
        "Dew Point(℃)": ("dew_point_c", 1), "Feels Like(℃)": ("feels_like_c", 1),
        "Wind(m/s)": ("wind_ms", 1), "REL Pressure(inhg)": ("pressure_inhg", 2),
        "Solar Rad.(lux)": ("solar_lux", 0), "UVI": ("uvi", 0)}
PEAK = {"Gust(m/s)": ("gust_ms", 1)}
LAST = {"Hourly Rain(mm)": "rain_hourly_mm", "Event Rain(mm)": "rain_event_mm",
        "Daily Rain(mm)": "rain_daily_mm", "Weekly Rain(mm)": "rain_weekly_mm",
        "Monthly Rain(mm)": "rain_monthly_mm", "Yearly Rain(mm)": "rain_yearly_mm"}
ORDER = ["time", "temp_c", "humidity", "dew_point_c", "feels_like_c", "wind_ms", "gust_ms",
         "wind_dir_deg", "pressure_inhg", "solar_lux", "uvi", *LAST.values()]


def build(raw: Path) -> pd.DataFrame:
    df = pd.read_csv(raw, na_values=["--"])
    df["Time"] = pd.to_datetime(df["Time"], format="%Y/%m/%d %H:%M")
    df = df.drop_duplicates("Time").set_index("Time").sort_index().loc[START:END]
    df = df[df.index < END]
    bins = df.resample("10min")
    out = pd.DataFrame({"n": bins.size()})
    for col, (name, digits) in MEAN.items():
        out[name] = bins[col].mean().round(digits)
    for col, (name, digits) in PEAK.items():
        out[name] = bins[col].max().round(digits)
    rad = np.deg2rad(df["Wind Direction(°)"])
    sin, cos = np.sin(rad).resample("10min").mean(), np.cos(rad).resample("10min").mean()
    out["wind_dir_deg"] = (np.rad2deg(np.arctan2(sin, cos)) % 360).round(0)
    for col, name in LAST.items():
        out[name] = bins[col].last()
    out = out[out["n"] >= 5].drop(columns="n")
    for name, digits in [*MEAN.values(), *PEAK.values()]:
        if digits == 0:
            out[name] = out[name].astype(int)
    out["wind_dir_deg"] = out["wind_dir_deg"].astype(int) % 360
    out.insert(0, "time", out.index.strftime("%Y-%m-%d %H:%M"))
    return out[ORDER].reset_index(drop=True)


if __name__ == "__main__":
    frame = build(Path(sys.argv[1]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"{OUT.name}: {len(frame):,} rows, {frame['time'].iloc[0]} → {frame['time'].iloc[-1]}, "
          f"{OUT.stat().st_size // 1024} KB")

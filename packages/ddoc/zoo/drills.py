"""Drills — one small case per drift pattern, with the reading it must get.

The Drift Zoo tells stories; the drills check the pattern reading. Each
drill builds a baseline and a current table where exactly one pattern is
planted (columns that should not change are drawn again from the same
distribution, so they carry real sampling noise), and names the
archetypes `ddoc diff` must read and, for attribute patterns, on which
columns.

Attribute patterns are checked strictly: the planted one must be read, on
the right column, and no other attribute pattern may appear. Dataset-level
readings (local shift, hidden change, ...) follow from the scores and the
verdict; they are checked strictly only in the drills aimed at them, as is
the verdict (``verdict=None``: shown, not checked).

    python zoo/eval.py          # run every drill and every zoo story, print the table

The embedding pattern needs an image or text plugin and is not drilled here.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

N = 600


@dataclass
class Drill:
    id: str
    story: str                                   # what was planted (Korean, for the README)
    make: Callable[[np.random.Generator], Tuple[pd.DataFrame, pd.DataFrame]]
    expect: Dict[str, Optional[List[str]]]       # archetype → columns (None: dataset-level)
    verdict: Optional[str] = None                # checked only when given


def _base(rng, n: int = N) -> pd.DataFrame:
    """Columns that do not change: each call is a fresh sample."""
    return pd.DataFrame({
        "temp": np.round(rng.normal(20, 2, n), 2),
        "channel": rng.choice(["store", "online", "app"], n, p=[0.5, 0.3, 0.2]),
    })


def _numeric(ref_values, cur_values, rng):
    ref, cur = _base(rng, len(ref_values)), _base(rng, len(cur_values))
    ref["x"] = np.round(ref_values, 3)
    cur["x"] = np.round(cur_values, 3)
    return ref, cur


def _categorical(ref_p: Dict[str, float], cur_p: Dict[str, float], rng):
    ref, cur = _base(rng), _base(rng)
    ref["kind"] = rng.choice(list(ref_p), N, p=list(ref_p.values()))
    cur["kind"] = rng.choice(list(cur_p), N, p=list(cur_p.values()))
    return ref, cur


def _stable(rng):
    return _numeric(rng.normal(50, 5, N), rng.normal(50, 5, N), rng)


def _small(rng):
    return _base(rng, 20), _base(rng, 20)


def _mixed(rng, n: int) -> pd.DataFrame:
    """Ten columns of the kinds real tables have."""
    return pd.DataFrame({
        **{f"c{i}": np.round(rng.normal(10 * i, 2, n), 2) for i in range(1, 5)},
        "skew1": np.round(rng.lognormal(1, 0.6, n), 2), "skew2": np.round(rng.exponential(5, n), 2),
        "rating": rng.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.1, 0.25, 0.45, 0.15]),
        "count": rng.poisson(3, n),
        "channel": rng.choice(["store", "online", "app"], n, p=[0.5, 0.3, 0.2]),
        "region": rng.choice(list("NSEW"), n),
    })


def _small_unchanged(rng):
    return _mixed(rng, 30), _mixed(rng, 30)


def _schema(rng):
    ref, cur = _stable(rng)
    cur["coupon"] = rng.choice(["Y", "N"], N)
    return ref, cur


def _regime(rng):
    ref = _base(rng)
    ref["price"] = np.round(rng.normal(100, 10, N), 1)
    ref["qty"] = rng.integers(1, 5, N)
    cur = pd.DataFrame({
        "temp": np.round(rng.normal(20, 2, N), 2),
        "channel": rng.choice(["store", "online", "app"], N, p=[0.2, 0.3, 0.5]),
        "price": np.round(rng.normal(140, 10, N), 1),
        "qty": rng.integers(3, 7, N),
    })
    return ref, cur


def _local(rng):
    ref = pd.DataFrame({f"m{i}": np.round(rng.normal(10 * i, 2, N), 2) for i in range(1, 7)})
    cur = pd.DataFrame({f"m{i}": np.round(rng.normal(10 * i, 2, N), 2) for i in range(1, 7)})
    cur["m6"] = np.round(rng.normal(60 + 2 * 2, 2, N), 2)   # 2σ up, same spread
    return ref, cur


def _hidden(rng):
    # 40 rows a side, the mean exactly 0.6σ up: the level-shift conditions
    # hold, but on so few rows chance alone could make that difference
    n = 40
    q = np.array([NormalDist().inv_cdf((i + 0.5) / n) for i in range(n)])
    return _numeric(50 + 5 * q, rng.permutation(50 + 5 * (q + 0.6)), rng)


def _missing(rng):
    ref, cur = _numeric(rng.normal(50, 5, N), rng.normal(50, 5, N), rng)
    cur.loc[cur.index % 4 == 0, "x"] = np.nan
    return ref, cur


def _tail(rng):
    cur_values = rng.normal(50, 5, N)
    cur_values[::12] = 200.0                     # about 8 % far above the baseline range
    return _numeric(rng.normal(50, 5, N), cur_values, rng)


DRILLS: List[Drill] = [
    Drill("stable", "같은 분포의 다른 표본", _stable, {"stable": None}, "OK"),
    Drill("small-sample", "같은 분포, 쪽마다 20행", _small, {"small_sample": None, "hidden_change": None}, "OK"),
    Drill("schema-change", "같은 분포 + 현재에만 있는 열", _schema, {"schema_change": None}, "OK"),
    Drill("regime-shift", "4개 열 중 3개가 함께 이동 (가격대·수량·채널 구성)", _regime,
          {"regime_shift": None, "level_shift": ["price", "qty"], "mix_shift": ["channel"]}, "DRIFT"),
    Drill("local-shift", "6개 열 중 1개만 2σ 이동", _local,
          {"local_shift": None, "level_shift": ["m6"]}, "DRIFT"),
    Drill("hidden-change", "쪽마다 40행, 평균 0.6σ 이동 — 유형 조건은 채우지만 표본 잡음 범위 안", _hidden,
          {"hidden_change": None}, "OK"),
    Drill("unit-change", "값 ×1000 (단위 변경)",
          lambda r: _numeric(r.normal(50, 5, N), r.normal(50, 5, N) * 1000, r), {"unit_change": ["x"]}),
    Drill("stuck-values", "값이 7.0 하나로 고정",
          lambda r: _numeric(r.normal(50, 5, N), np.full(N, 7.0), r), {"stuck_values": ["x"]}),
    Drill("shift-and-spread", "평균 +1σ, 산포 ×1.8",
          lambda r: _numeric(r.normal(50, 5, N), r.normal(55, 9, N), r), {"shift_and_spread": ["x"]}),
    Drill("level-shift", "평균 +1σ, 산포 그대로",
          lambda r: _numeric(r.normal(50, 5, N), r.normal(55, 5, N), r), {"level_shift": ["x"]}),
    Drill("spread-change", "평균 그대로, 산포 ×2",
          lambda r: _numeric(r.normal(50, 5, N), r.normal(50, 10, N), r), {"spread_change": ["x"]}),
    Drill("tail-outliers", "몸통은 그대로, 약 8%가 기준 범위 훨씬 밖", _tail, {"tail_outliers": ["x"]}),
    Drill("missing-surge", "값 분포는 그대로, 결측 25%", _missing,
          {"missing_surge": ["x"]}),
    Drill("new-category", "새 값 app 이 20% 로 등장",
          lambda r: _categorical({"store": 0.6, "online": 0.4}, {"store": 0.48, "online": 0.32, "app": 0.2}, r),
          {"new_category": ["kind"]}),
    Drill("vanished-category", "값 app (20%) 이 사라짐",
          lambda r: _categorical({"store": 0.48, "online": 0.32, "app": 0.2}, {"store": 0.6, "online": 0.4}, r),
          {"vanished_category": ["kind"]}),
    Drill("mix-shift", "같은 값들의 비중이 70/30 → 40/60",
          lambda r: _categorical({"store": 0.7, "online": 0.3}, {"store": 0.4, "online": 0.6}, r),
          {"mix_shift": ["kind"]}),
    # (appended: a drill's seed is its position, so the ones above keep their data)
    Drill("shift-and-narrow", "평균 +1σ, 산포 ×0.5",
          lambda r: _numeric(r.normal(50, 5, N), r.normal(55, 2.5, N), r), {"shift_and_narrow": ["x"]}),
    Drill("small-unchanged", "같은 분포에서 따로 뽑은 쪽마다 30행 · 10열 (오경보 확인)", _small_unchanged, {}, "OK"),
]

# The Drift Zoo stories, read the same way (see each make.py).
STORIES: Dict[str, Dict[str, Optional[List[str]]]] = {
    "regime_shift": {"regime_shift": None, "shift_and_spread": ["price"], "mix_shift": ["channel"]},
    "creeping_degradation": {"shift_and_spread": ["reading"]},
    "rhythm_not_drift": {"stable": None},
    # real data: the rainy season. Dew point, the column that moved most, rises
    # and narrows as the air nears saturation; most of the weather moved with it.
    "weather_station": {"regime_shift": None, "shift_and_narrow": ["dew_point_c"],
                        "level_shift": ["humidity", "rain_hourly_mm", "temp_c"],
                        "shift_and_spread": ["feels_like_c", "pressure_inhg"]},
}

"""Verdict sensitivity — how candidate scoring rules judge.

`ddoc diff` used to score each column with the Jensen–Shannon divergence
and call DRIFT when one column reached 0.25 (or the mean 0.15). The drills
showed that rule missing clear single changes, and small samples raising
alarms on noise. This script compared it with alternatives; ddoc now uses
the last one (``reading``, :mod:`ddoc.scoring`), and the script keeps the
comparison reproducible:

* ``js``       — before: JS divergence (base 2), 0.15 / 0.25
* ``js_dist``  — JS distance (square root of the divergence), same thresholds
* ``psi``      — population stability index on baseline deciles, 0.10 / 0.25
* ``gated``    — JS distance, but a column whose difference sample noise
  explains (G-test p ≥ 0.001 on the same bins) scores 0
* ``reading``  — ``gated``, and a column whose pattern reading matches
  (level shift, missing surge, ...) is flagged too, under the same noise test

and measures three things:

1. the verdict on every drill and Drift Zoo story;
2. false alarms — two independent samples of the *same* distribution, by
   sample size;
3. the smallest change each rule catches in 90 % of runs, in plain units
   (σ of a mean shift, spread ratio, %p of a mix shift, ...).

A fourth part checks rows in time order, where neighbours resemble each
other: the noise test counting rows one by one against counting them by
the day (:func:`ddoc.scoring.time_blocks`), on generated ten-minute
series whose days resemble the days before them to a chosen degree, and
on the real weather station's stories.

    python zoo/sensitivity.py           # Markdown tables (docs/sensitivity.md)

Every comparison goes through the built-in comparator (`csv_diff`) on CSV
files, so the summaries and readings are the ones `ddoc diff` produces.
Seeds are fixed; the tables are reproducible.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from drills import DRILLS  # noqa: E402

from ddoc import archetypes as ar  # noqa: E402
from ddoc import scoring  # noqa: E402
from ddoc.cli.commands.diff import compute_verdict, csv_diff  # noqa: E402
from ddoc.examples import weather  # noqa: E402
from ddoc.scoring import ALPHA, g_test_p, js_divergence  # noqa: E402

BINS = 20
PSI_FLOOR = 1e-4


# ── One comparison, the way `ddoc diff` makes it ────────────────────────


def histograms(r: pd.Series, c: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Aligned ref/cur counts, binned exactly as the built-in comparator does."""
    r, c = r.dropna(), c.dropna()
    if pd.api.types.is_numeric_dtype(r) and pd.api.types.is_numeric_dtype(c):
        lo, hi = float(min(r.min(), c.min())), float(max(r.max(), c.max()))
        if hi == lo:
            return np.array([len(r)]), np.array([len(c)])
        if bool((r % 1 == 0).all() and (c % 1 == 0).all()) and hi - lo + 1 <= BINS:
            values = range(int(lo), int(hi) + 1)
            rvc, cvc = r.value_counts(), c.value_counts()
            return (np.array([rvc.get(v, 0) for v in values]),
                    np.array([cvc.get(v, 0) for v in values]))
        edges = np.linspace(lo, hi, BINS + 1)
        return np.histogram(r, bins=edges)[0], np.histogram(c, bins=edges)[0]
    rvc, cvc = r.astype(str).value_counts(), c.astype(str).value_counts()
    keys = sorted(set(rvc.index) | set(cvc.index))
    return np.array([rvc.get(k, 0) for k in keys]), np.array([cvc.get(k, 0) for k in keys])


def psi(r: pd.Series, c: pd.Series) -> float:
    """PSI on baseline deciles (numeric) or on the categories."""
    r, c = r.dropna(), c.dropna()
    if pd.api.types.is_numeric_dtype(r) and pd.api.types.is_numeric_dtype(c) and r.nunique() > 10:
        inner = np.unique(np.quantile(r, np.linspace(0.1, 0.9, 9)))
        edges = np.concatenate(([-np.inf], inner, [np.inf]))
        rh, ch = np.histogram(r, bins=edges)[0], np.histogram(c, bins=edges)[0]
    else:
        rh, ch = histograms(r.astype(str), c.astype(str))
    p = np.maximum(rh / max(rh.sum(), 1), PSI_FLOOR)
    q = np.maximum(ch / max(ch.sum(), 1), PSI_FLOOR)
    return float(((q - p) * np.log(q / p)).sum())


@dataclass
class Column:
    js: float               # JS divergence on the comparator's bins (the old score)
    p: float                # G-test on the same bins
    psi: float
    patterns: List[str]     # matched attribute archetypes
    missing_p: float        # 2×2 test on the missing counts


def compare(ref: pd.DataFrame, cur: pd.DataFrame, tmp: Path) -> Dict[str, Column]:
    ref.to_csv(tmp / "ref.csv", index=False)
    cur.to_csv(tmp / "cur.csv", index=False)
    payload = csv_diff([str(tmp / "ref.csv")], [str(tmp / "cur.csv")])
    r_df, c_df = pd.read_csv(tmp / "ref.csv"), pd.read_csv(tmp / "cur.csv")
    out = {}
    for col, score in payload["attribute_drifts"].items():
        rh, ch = histograms(r_df[col], c_df[col])
        js = js_divergence(rh, ch)
        p = g_test_p(rh, ch)
        assert abs((math.sqrt(js) if p < ALPHA else 0.0) - score) < 1e-4   # what ddoc diff reports
        summary = payload["summaries"][col]
        cards = ar._attr_cards(col, ar._measure(summary, payload["distributions"].get(col)))
        n, miss = summary["n"], summary["missing"]
        mr, mc = round(miss["ref"] * n["ref"]), round(miss["cur"] * n["cur"])
        out[col] = Column(js=js, p=p, psi=psi(r_df[col], c_df[col]),
                          patterns=[c["code"] for c in cards if c["match"] == "match"],
                          missing_p=g_test_p(np.array([mr, n["ref"] - mr]), np.array([mc, n["cur"] - mc])))
    return out


# ── The rules ───────────────────────────────────────────────────────────


@dataclass
class Rule:
    key: str
    label: str
    score: Callable[[Column], float]
    overall: float
    attr: float
    reading: bool = False

    def flagged(self, col: Column) -> bool:
        """This column alone makes the verdict DRIFT."""
        if self.score(col) >= self.attr:
            return True
        if not self.reading or not col.patterns:
            return False
        other = [p for p in col.patterns if p != "missing_surge"]
        return ("missing_surge" in col.patterns and col.missing_p < ALPHA) or (bool(other) and col.p < ALPHA)

    def verdict(self, cols: Dict[str, Column]) -> str:
        if not cols:
            return "OK"
        mean = sum(self.score(c) for c in cols.values()) / len(cols)
        return "DRIFT" if mean >= self.overall or any(self.flagged(c) for c in cols.values()) else "OK"


def _gated(col: Column) -> float:
    return math.sqrt(col.js) if col.p < ALPHA else 0.0


RULES = [
    Rule("js", "이전: JS 발산", lambda c: c.js, 0.15, 0.25),
    Rule("js_dist", "JS 거리", lambda c: math.sqrt(c.js), 0.15, 0.25),
    Rule("psi", "PSI", lambda c: c.psi, 0.10, 0.25),
    Rule("gated", "JS 거리 + 잡음 거름", _gated, 0.15, 0.25),
    Rule("reading", "+ 판독 연동 (현행)", _gated, 0.15, 0.25, reading=True),
]
SWEEP = [Rule(f"gated{t}", f"잡음 거름, 단일 속성 {t}", _gated, 0.15, t) for t in (0.15, 0.20, 0.25, 0.30)]


# ── 1. drills and stories ───────────────────────────────────────────────

STORY_EXPECT = {"regime_shift": "DRIFT", "creeping_degradation": "DRIFT", "rhythm_not_drift": "OK"}
UNCHANGED = ("stable", "small-sample", "schema-change")


def cases(tmp: Path) -> List[Dict[str, Any]]:
    rows = []
    for i, drill in enumerate(DRILLS):
        ref, cur = drill.make(np.random.default_rng(1000 + i))   # the seeds zoo/eval.py uses
        rows.append({"id": drill.id, "story": drill.story, "cols": compare(ref, cur, tmp),
                     "expect": drill.verdict or ("OK" if drill.id in UNCHANGED else "DRIFT")})
    for case, expect in STORY_EXPECT.items():
        spec = importlib.util.spec_from_file_location(f"zoo_{case}", HERE / case / "make.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ref_p, cur_p = mod.make(tmp / case)
        rows.append({"id": case, "story": "이야기 케이스", "expect": expect,
                     "cols": compare(pd.read_csv(ref_p), pd.read_csv(cur_p), tmp)})
    return rows


# ── 2. false alarms on unchanged data ───────────────────────────────────

SIZES = [30, 100, 300, 1000, 5000]
NULL_REPS = 200


def _null_table(rng, n: int) -> pd.DataFrame:
    """Ten columns of everyday kinds: continuous, skewed, rating, count, categories."""
    cols: Dict[str, Any] = {}
    for i in range(4):
        cols[f"num{i}"] = np.round(rng.normal(50, 5, n), 2)
    for i in range(2):
        cols[f"skew{i}"] = np.round(rng.lognormal(3, 0.6, n), 2)
    cols["rating"] = rng.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.1, 0.25, 0.45, 0.15])
    cols["count"] = rng.poisson(3, n)
    cols["channel"] = rng.choice(["store", "online", "app"], n, p=[0.5, 0.3, 0.2])
    cols["region"] = rng.choice([f"r{k}" for k in range(8)], n, p=[.3, .2, .15, .1, .1, .05, .05, .05])
    return pd.DataFrame(cols)


def false_alarms(rules: List[Rule], tmp: Path) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """rule → n → (share of columns flagged, share of 10-column tables called DRIFT)."""
    out: Dict[str, Dict[int, Tuple[float, float]]] = {r.key: {} for r in rules}
    for n in SIZES:
        rng = np.random.default_rng(7 + n)
        flagged = {r.key: 0 for r in rules}
        drift = {r.key: 0 for r in rules}
        n_cols = 0
        for _ in range(NULL_REPS):
            cols = compare(_null_table(rng, n), _null_table(rng, n), tmp)
            n_cols += len(cols)
            for rule in rules:
                flagged[rule.key] += sum(rule.flagged(c) for c in cols.values())
                drift[rule.key] += rule.verdict(cols) == "DRIFT"
        for rule in rules:
            out[rule.key][n] = (flagged[rule.key] / n_cols, drift[rule.key] / NULL_REPS)
    return out


# ── 3. the smallest change each rule catches ───────────────────────────

DETECT_REPS = 40


def _cats(rng, n, p: Dict[str, float]) -> np.ndarray:
    return rng.choice(list(p), n, p=list(p.values()))


def _with_missing(rng, n, share):
    x = rng.normal(0, 1, n)
    x[rng.random(n) < share / 100] = np.nan
    return x


CHANGES: List[Tuple[str, str, List[float], Callable]] = [
    ("평균 이동", "σ", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0],
     lambda rng, n, d: (rng.normal(0, 1, n), rng.normal(d, 1, n))),
    ("산포 변화", "배", [1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
     lambda rng, n, k: (rng.normal(0, 1, n), rng.normal(0, k, n))),
    ("구성비 이동 (70/30에서)", "%p", [5, 10, 15, 20, 25, 30, 35, 40],
     lambda rng, n, d: (_cats(rng, n, {"a": .7, "b": .3}), _cats(rng, n, {"a": .7 - d / 100, "b": .3 + d / 100}))),
    ("새 범주 등장", "%", [1, 2, 5, 10, 15, 20, 30],
     lambda rng, n, s: (_cats(rng, n, {"a": .6, "b": .4}),
                        _cats(rng, n, {"a": .6 * (1 - s / 100), "b": .4 * (1 - s / 100), "new": s / 100}))),
    ("극단값 (+30σ)", "%", [1, 2, 5, 8, 15, 25],
     lambda rng, n, s: (rng.normal(0, 1, n), np.where(rng.random(n) < s / 100, 30.0, rng.normal(0, 1, n)))),
    ("결측 증가", "%p", [2, 5, 10, 25, 50],
     lambda rng, n, s: (rng.normal(0, 1, n), _with_missing(rng, n, s))),
]


def detection(rules: List[Rule], n: int, tmp: Path) -> Dict[str, Dict[str, Optional[float]]]:
    """change → rule → smallest size flagged in ≥ 90 % of runs."""
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for name, _unit, sizes, make in CHANGES:
        hits: Dict[float, Dict[str, int]] = {}
        for size in sizes:
            rng = np.random.default_rng(int(size * 1000) + n)
            hits[size] = {r.key: 0 for r in rules}
            for _ in range(DETECT_REPS):
                ref, cur = make(rng, n, size)
                col = compare(pd.DataFrame({"x": ref}), pd.DataFrame({"x": cur}), tmp)["x"]
                for rule in rules:
                    hits[size][rule.key] += rule.flagged(col)
        out[name] = {r.key: next((s for s in sizes if hits[s][r.key] >= 0.9 * DETECT_REPS), None)
                     for r in rules}
    return out


# ── Output ──────────────────────────────────────────────────────────────


# ── Rows in time order ─────────────────────────────────────────────────

TS_DAYS = 14
TS_REPS = 100
TS_RHO = (0.0, 0.3, 0.5, 0.7)     # how much a day's weather resembles the day before
TS_SHIFTS = (0.0, 1.0, 2.0)       # the current period's level, in σ of the daily level


def _ten_minute_days(rng, rho: float, shift: float, start: str) -> pd.DataFrame:
    """``TS_DAYS`` days of ten-minute readings: a daily cycle, each day's
    own level (σ = 2, correlated ``rho`` with the day before), and a
    wobble that carries over from one reading to the next."""
    level = rng.normal(0, 2)
    values = []
    for day in range(TS_DAYS):
        if day:
            level = rho * level + rng.normal(0, 2 * math.sqrt(1 - rho * rho))
        wobble = 0.0
        for step in range(144):
            wobble = 0.95 * wobble + rng.normal(0, 0.3)
            values.append(level + 2 * shift + 4 * math.sin(2 * math.pi * step / 144) + wobble)
    times = pd.date_range(start, periods=len(values), freq="10min")
    return pd.DataFrame({"time": times.strftime("%Y-%m-%d %H:%M"), "reading": np.round(values, 2)})


class _RowsOneByOne:
    """The noise test as before: every row on its own, time or not."""

    def __enter__(self):
        self.saved, scoring.time_blocks = scoring.time_blocks, lambda *_: None

    def __exit__(self, *_):
        scoring.time_blocks = self.saved


def _verdict(ref_csv: Path, cur_csv: Path) -> Tuple[str, Dict[str, Any]]:
    payload = csv_diff([str(ref_csv)], [str(cur_csv)])
    verdict = compute_verdict(payload, threshold=0.15, attr_threshold=0.25,
                              evidence=payload["distributions"])[0]
    return verdict, payload


def time_series(tmp: Path) -> Tuple[Dict[Tuple[float, float], Tuple[float, float]], List[Dict[str, Any]]]:
    """DRIFT rates (rows one by one, by the day) on generated series, and
    the weather station's stories both ways."""
    rng = np.random.default_rng(31)
    rates = {}
    for rho in TS_RHO:
        for shift in TS_SHIFTS:
            hits = [0, 0]
            for _ in range(TS_REPS):
                _ten_minute_days(rng, rho, 0.0, "2026-06-01").to_csv(tmp / "ref.csv", index=False)
                _ten_minute_days(rng, rho, shift, "2026-07-01").to_csv(tmp / "cur.csv", index=False)
                with _RowsOneByOne():
                    hits[0] += _verdict(tmp / "ref.csv", tmp / "cur.csv")[0] == "DRIFT"
                hits[1] += _verdict(tmp / "ref.csv", tmp / "cur.csv")[0] == "DRIFT"
            rates[(rho, shift)] = (hits[0] / TS_REPS, hits[1] / TS_REPS)
    stories = []
    for story in weather.STORIES:
        ref, cur = weather.write_pair(story, tmp / story)
        with _RowsOneByOne():
            before, plain = _verdict(ref, cur)
        after, payload = _verdict(ref, cur)
        stories.append({"story": story, "before": (before, plain["overall_score"]),
                        "after": (after, payload["overall_score"]),
                        "kept": sorted(k for k, v in payload["attribute_drifts"].items() if v > 0)})
    return rates, stories


def _pct(x: float) -> str:
    return "0" if x == 0 else ("<0.1%" if x < 0.001 else f"{x * 100:.1f}%")


def _head(rules: List[Rule], first: str) -> List[str]:
    return [f"| {first} | " + " | ".join(r.label for r in rules) + " |", "|---|" + "---|" * len(rules)]


def markdown() -> str:
    lines: List[str] = []
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        rows = cases(tmp)
        lines += ["### 1. 드릴과 이야기 케이스", "",
                  "칸마다 판정과 최대 속성 점수(판독 연동은 판정만). 굵게 = 기대와 다름. 변화를 심은 케이스는 "
                  "DRIFT, 나머지는 OK 가 기대입니다.", "",
                  f"| 케이스 | 심은 변화 | 기대 | " + " | ".join(r.label for r in RULES) + " |",
                  "|---|---|---|" + "---|" * len(RULES)]
        agree = {r.key: 0 for r in RULES}
        for row in rows:
            cells = []
            for rule in RULES:
                v = rule.verdict(row["cols"])
                agree[rule.key] += v == row["expect"]
                top = max((rule.score(c) for c in row["cols"].values()), default=0.0)
                text = v if rule.reading else f"{v} ({top:.3f})"
                cells.append(text if v == row["expect"] else f"**{text}**")
            lines.append(f"| `{row['id']}` | {row['story']} | {row['expect']} | " + " | ".join(cells) + " |")
        lines.append("| **기대와 일치** | | | " + " | ".join(f"**{agree[r.key]}/{len(rows)}**" for r in RULES) + " |")

        fa = false_alarms(RULES, tmp)
        lines += ["", "### 2. 변하지 않은 데이터에서의 오경보", "",
                  f"같은 분포에서 따로 뽑은 두 표본(쪽마다 n 행, 10개 열: 연속 4 · 치우친 연속 2 · 1–5 점수 · 횟수 · "
                  f"범주 2)을 {NULL_REPS}번 비교했습니다. 앞 숫자는 혼자서 판정을 DRIFT 로 만든 열의 비율, 뒤 숫자는 "
                  "표가 DRIFT 로 판정된 비율입니다. 여기서 DRIFT 는 모두 오경보입니다.", ""]
        lines += _head(RULES, "쪽마다 행 수")
        for n in SIZES:
            lines.append(f"| {n:,} | " + " | ".join(f"{_pct(fa[r.key][n][0])} · {_pct(fa[r.key][n][1])}"
                                                   for r in RULES) + " |")

        det = {n: detection(RULES + SWEEP, n, tmp) for n in (100, 1000)}
        lines += ["", "### 3. 잡아내는 가장 작은 변화", "",
                  f"열 하나만 바꿨을 때 {DETECT_REPS}번 중 90% 이상 그 열이 판정을 DRIFT 로 만든 가장 작은 크기. "
                  "칸은 쪽마다 100행 / 1,000행. — 는 시험한 범위에서 잡지 못함.", ""]
        lines += _head(RULES, "변화 (시험 범위)")
        for name, unit, sizes, _ in CHANGES:
            def cell(key: str) -> str:
                vals = [det[n][name][key] for n in (100, 1000)]
                return " / ".join("—" if v is None else f"{v:g}{unit}" for v in vals)
            lines.append(f"| {name} ({sizes[0]:g}–{sizes[-1]:g}{unit}) | "
                         + " | ".join(cell(r.key) for r in RULES) + " |")

        lines += ["", "### 4. 잡음 거름에서 단일 속성 기준값을 바꾸면", "",
                  "3과 같은 방식, 쪽마다 1,000행. 잡음 거름이 있으면 기준값은 오경보가 아니라 \"얼마나 큰 변화부터 "
                  "알릴지\"만 정합니다.", ""]
        lines += _head(SWEEP, "변화")
        for name, unit, _sizes, _ in CHANGES:
            lines.append(f"| {name} | " + " | ".join(
                "—" if det[1000][name][r.key] is None else f"{det[1000][name][r.key]:g}{unit}" for r in SWEEP) + " |")
        sweep_agree = [sum(r.verdict(row["cols"]) == row["expect"] for row in rows) for r in SWEEP]
        lines.append("| 드릴·이야기 일치 | " + " | ".join(f"{a}/{len(rows)}" for a in sweep_agree) + " |")

        rates, stories = time_series(tmp)
        lines += ["", "### 5. 시간 순서 기록", "",
                  f"쪽마다 {TS_DAYS}일의 10분 간격 기록(하루 주기 + 날마다 다른 수준 + 이웃 값끼리 닮은 흔들림)을 "
                  f"{TS_REPS}번 비교했습니다. ρ 는 하루의 수준이 전날과 닮은 정도(상관), 이동은 현재 기간의 수준을 "
                  "날마다의 수준 표준편차 단위로 올린 것입니다. 칸은 DRIFT 로 판정된 비율, 행 단위 → 하루 단위. "
                  "이동 0 의 DRIFT 는 모두 오경보입니다.", "",
                  "| 하루끼리 닮은 정도 | " + " | ".join("변화 없음" if s == 0 else f"수준 +{s:g}σ" for s in TS_SHIFTS)
                  + " |", "|---|" + "---|" * len(TS_SHIFTS)]
        for rho in TS_RHO:
            lines.append(f"| ρ = {rho:g} | " + " | ".join(
                f"{_pct(rates[(rho, s)][0])} → {_pct(rates[(rho, s)][1])}" for s in TS_SHIFTS) + " |")
        lines += ["", "실측 기상관측소의 이야기([weather_station](../zoo/weather_station/)), 판정(점수):", "",
                  "| 이야기 | 행 단위 | 하루 단위 (현행) | 점수가 남은 열 |", "|---|---|---|---|"]
        for row in stories:
            (b, bs), (a, as_) = row["before"], row["after"]
            lines.append(f"| `{row['story']}` | {b} ({bs:.3f}) | {a} ({as_:.3f}) | "
                         + (", ".join(row["kept"]) or "—") + " |")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    print(markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

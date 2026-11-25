"""Example cases — small stories with a known answer, for the GUI's front page.

Each case writes a baseline and a current CSV and says what happened and
what the verdict should be, so a first-time user can press one button and
see a real problem being diagnosed. They ship inside the package (the
Drift Zoo's make.py scripts live in the repository only) and carry a date
column, so the report also shows the periods compared. The first two are
real: windows of a home weather station's log (:mod:`ddoc.examples.weather`);
the rest are generated stories, each with one planted change.

    from ddoc.examples import cases
    ref, cur = cases.build("price-regime", out_dir)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from . import weather

Rows = List[str]


@dataclass(frozen=True)
class Case:
    id: str
    icon: str                      # a glyph for the card
    verdict: str                   # what `ddoc diff` should answer
    text: Dict[str, Dict[str, str]]  # lang → title, story, look (what to look at in the result)
    header: str
    make: Callable[[np.random.Generator], Tuple[Rows, Rows]]
    real: bool = False             # measured data rather than a generated story


def _days(start: date, n: int) -> List[str]:
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


def _weather(story: str) -> Callable[[np.random.Generator], Tuple[Rows, Rows]]:
    (r0, r1), (c0, c1) = weather.STORIES[story]
    return lambda rng: (weather.window(r0, r1), weather.window(c0, c1))


def _price_regime(rng) -> Tuple[Rows, Rows]:
    def period(start: date, price_mu: float, online_p: float, app_p: float) -> Rows:
        rows = []
        for day in _days(start, 45):
            for _ in range(int(rng.integers(25, 40))):
                price = max(1.0, rng.normal(price_mu, price_mu * 0.08))
                u = rng.random()
                channel = "app" if u < app_p else "online" if u < app_p + online_p else "store"
                rows.append(f"{day},{price:.0f},{channel},{int(rng.integers(1, 5))}")
        return rows
    return period(date(2026, 3, 1), 100, 0.20, 0.05), period(date(2026, 7, 1), 165, 0.45, 0.35)


def _sensor_decay(rng) -> Tuple[Rows, Rows]:
    def window(start: date, days: int, bias_end: float, noise_end: float, spike_p: float, frac0: float) -> Rows:
        rows = []
        for i, day in enumerate(_days(start, days)):
            frac = frac0 + (1.0 - frac0) * i / max(1, days - 1)
            for hour in range(24):
                for _ in range(3):
                    reading = (20.0 + 6.0 * np.sin((hour - 6) / 24 * 2 * np.pi) + bias_end * frac
                               + rng.normal(0, 0.6 + (noise_end - 0.6) * frac))
                    if rng.random() < spike_p * frac:
                        reading += rng.choice([-1, 1]) * rng.uniform(10, 18)
                    humidity = float(np.clip(rng.normal(45, 6), 5, 95))
                    rows.append(f"{day},{hour},{reading:.2f},{humidity:.1f}")
        return rows
    return (window(date(2026, 5, 1), 30, 0.0, 0.6, 0.0, 0.0),
            window(date(2026, 6, 15), 14, 8.0, 3.2, 0.08, 0.55))


def _unit_change(rng) -> Tuple[Rows, Rows]:
    def batch(start: date, scale: float) -> Rows:
        rows = []
        for day in _days(start, 30):
            for _ in range(int(rng.integers(30, 45))):
                weight = max(0.05, rng.normal(2.4, 0.5)) * scale
                carrier = rng.choice(["road", "air", "sea"], p=[0.6, 0.25, 0.15])
                rows.append(f"{day},{weight:.2f},{carrier},{float(rng.normal(3.2, 0.9)):.1f}")
        return rows
    return batch(date(2026, 4, 1), 1.0), batch(date(2026, 5, 1), 1000.0)   # kg → g upstream


def _weekday_rhythm(rng) -> Tuple[Rows, Rows]:
    def week(start: date) -> Rows:
        rows = []
        for day in _days(start, 7):
            for hour in range(24):
                base = (60 * np.exp(-((hour - 8) ** 2) / 4) + 80 * np.exp(-((hour - 12.5) ** 2) / 3)
                        + 70 * np.exp(-((hour - 18.5) ** 2) / 5) + 5)
                for _ in range(max(0, int(rng.normal(base, base * 0.07) / 6))):
                    rows.append(f"{day},{hour},{max(1.0, rng.normal(12, 2.5)):.1f},{max(0.0, rng.normal(28, 9)):.2f}")
        return rows
    return week(date(2026, 6, 1)), week(date(2026, 6, 8))


CASES: List[Case] = [
    Case("monsoon", "☂", "DRIFT", {
        "en": {"title": "A weather station as the rainy season sets in",
               "story": "Ten-minute readings from a real home weather station, the last two weeks of June against the first two of July. From about 5 July, humid and rainy days set in.",
               "look": "Dew point and humidity step up, dew point also narrowing (shift with narrower spread): not a fault but a new season. The verdict is DRIFT and the job is a new baseline. The rain totals only grow, so they are set aside."},
        "ko": {"title": "장마가 시작된 기상관측소",
               "story": "실제 가정용 기상관측소의 10분 기록, 6월 말 2주와 7월 초 2주. 7월 5일 무렵부터 습하고 비 오는 날이 이어졌습니다.",
               "look": "이슬점과 습도가 한 단계 올라섰고, 이슬점은 폭까지 좁아졌습니다(이동·수축형). 고장이 아니라 계절이 바뀐 것이라 판정은 DRIFT, 할 일은 기준 데이터 갱신입니다. 늘어나기만 하는 누적 강우 열은 비교에서 빠집니다."}},
         ",".join(weather.header()), _weather("monsoon"), real=True),
    Case("yearly-reset", "↺", "OK", {
        "en": {"title": "The day a yearly rain total started over",
               "story": "The same station, the second half of May against the first half of June. On 1 June the station reset its yearly rain total from 1,294 mm to 0.",
               "look": "Compared as levels, that one column would make it DRIFT. ddoc finds the totals that only grow, sets them aside and says why; the weather itself reads OK."},
        "ko": {"title": "연간 강우 누적값이 초기화된 날",
               "story": "같은 관측소, 5월 후반 2주와 6월 전반 2주. 6월 1일에 관측소의 연간 강우 누적값이 1,294mm에서 0으로 초기화되었습니다.",
               "look": "누적값을 수준으로 비교하면 이 열 하나로 DRIFT가 됩니다. ddoc은 늘어나기만 하는 누적 열을 찾아 비교에서 빼고 이유를 적습니다. 날씨 자체는 OK입니다."}},
         ",".join(weather.header()), _weather("yearly-reset"), real=True),
    Case("price-regime", "↗", "DRIFT", {
        "en": {"title": "A shop whose prices and channels moved",
               "story": "Daily sales, March against July. Unit prices stepped up and sales moved from the store to online and the app.",
               "look": "Several columns move at once: the reading says regime shift, a new normal rather than one broken column."},
        "ko": {"title": "가격대와 채널이 통째로 바뀐 매장",
               "story": "일별 판매 기록, 3월과 7월. 단가가 한 단계 오르고 판매가 매장에서 온라인·앱으로 옮겨 갔습니다.",
               "look": "여러 열이 함께 움직입니다. 열 하나의 고장이 아니라 새 기준이 필요한 '레짐 전환형'으로 읽힙니다."}},
         "date,price,channel,qty", _price_regime),
    Case("sensor-decay", "∿", "DRIFT", {
        "en": {"title": "A sensor that is slowly failing",
               "story": "Temperature readings, last month against the last two weeks. Each day looks fine, yet the bias and noise keep growing.",
               "look": "The reading column shifts and spreads while humidity stays put: one instrument, not the environment."},
        "ko": {"title": "서서히 죽어가는 센서",
               "story": "온도 측정값, 지난달과 최근 2주. 하루하루는 멀쩡해 보이지만 편향과 잡음이 계속 커지고 있습니다.",
               "look": "측정값 열만 이동하고 퍼지며 습도는 그대로입니다. 환경이 아니라 계기 하나의 문제입니다."}},
         "date,hour,reading,humidity", _sensor_decay),
    Case("unit-change", "×", "DRIFT", {
        "en": {"title": "A column that changed its unit",
               "story": "Shipment records, April against May. An upstream system started sending weight in grams instead of kilograms.",
               "look": "Mean and spread grew by the same factor of 1000: the reading names a unit change, a data error to fix, not a change in the world."},
        "ko": {"title": "단위가 바뀐 열",
               "story": "배송 기록, 4월과 5월. 상류 시스템이 무게를 kg 대신 g 으로 보내기 시작했습니다.",
               "look": "평균과 산포가 똑같이 1000배가 됩니다. 세상의 변화가 아니라 고쳐야 할 데이터 오류인 '단위 변경형'으로 읽힙니다."}},
         "date,weight,carrier,lead_days", _unit_change),
    Case("weekday-rhythm", "≈", "OK", {
        "en": {"title": "Two weeks that only look different",
               "story": "Store visits, one week against the next. Every day has the same morning, lunch and evening peaks; only the noise differs.",
               "look": "A detector must be able to say no. The verdict is OK and the reading says stable."},
        "ko": {"title": "달라 보이지만 같은 두 주",
               "story": "매장 방문 기록, 한 주와 그다음 주. 매일 아침·점심·저녁 피크가 같고 잡음만 다릅니다.",
               "look": "탐지기는 '아니오'라고 말할 줄도 알아야 합니다. 판정은 OK, 판독은 '안정형'입니다."}},
         "date,hour,dwell_min,basket", _weekday_rhythm),
]
BY_ID = {c.id: c for c in CASES}


def build(case_id: str, out_dir: Path, *, seed: int = 5) -> Tuple[Path, Path]:
    """Write ``ref.csv`` and ``cur.csv`` of a case under ``out_dir``."""
    case = BY_ID[case_id]
    ref_rows, cur_rows = case.make(np.random.default_rng(seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    ref, cur = out_dir / "ref.csv", out_dir / "cur.csv"
    ref.write_text("\n".join([case.header, *ref_rows]) + "\n", encoding="utf-8")
    cur.write_text("\n".join([case.header, *cur_rows]) + "\n", encoding="utf-8")
    return ref, cur


def to_dict(case: Case, lang: str) -> Dict[str, Any]:
    text = case.text.get(lang) or case.text["en"]
    return {"id": case.id, "icon": case.icon, "verdict": case.verdict, "real": case.real, **text}

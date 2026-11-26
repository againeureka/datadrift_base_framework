"""Example cases — small stories with a known answer, for the GUI's front page.

Each case writes a baseline and a current dataset and says what happened
and what the verdict should be, so a first-time user can press one button
and see a real problem being diagnosed. They ship inside the package (the
Drift Zoo's make.py scripts live in the repository only). The CSV cases
carry a date column, so the report also shows the periods compared. Some
are real: windows of a home weather station's log (:mod:`ddoc.examples.weather`)
and frames of a dashcam drive (:mod:`ddoc.examples.dashcam`, two folders
of images, judged by the vision plugin); the rest are generated stories,
each with one planted change.

    from ddoc.examples import cases
    ref, cur = cases.build("price-regime", out_dir)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import dashcam, machine, support, weather

Rows = List[str]


@dataclass(frozen=True)
class Case:
    id: str
    icon: str                      # a glyph for the card
    verdict: str                   # what `ddoc diff` should answer
    text: Dict[str, Dict[str, str]]  # lang → title, story, look (what to look at in the result)
    header: str = ""
    make: Optional[Callable[[np.random.Generator], Tuple[Rows, Rows]]] = None
    real: bool = False             # measured data rather than a generated story
    # cases that are not two CSVs write their own data: out_dir → (ref, cur)
    folders: Optional[Callable[[Path], Tuple[Path, Path]]] = None
    needs: Optional[str] = None    # the plugin that judges it (images: vision)
    options: Dict[str, Any] = field(default_factory=dict)  # diff options for this case


def _days(start: date, n: int) -> List[str]:
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


def _dashcam(story: str) -> Callable[[Path], Tuple[Path, Path]]:
    return lambda out_dir: dashcam.write_pair(story, out_dir)


def _folders(module, story: str) -> Callable[[Path], Tuple[Path, Path]]:
    return lambda out_dir: module.write_pair(story, out_dir)


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
    # images: frames of a real dashcam drive, judged by the vision plugin
    Case("camera-swap", "⇆", "DRIFT", {
        "en": {"title": "The rear camera's footage came in",
               "story": "Frames of a real dashcam drive, every 6 s: the front camera against the rear one at the same moments, as if the wrong camera's footage reached a model trained on the front.",
               "look": "Exposure and colour move most, and on the embedding map the two cameras' frames sit apart (content shift in embeddings). Under the map the report names frames to open."},
        "ko": {"title": "후방 카메라 영상이 들어왔다",
               "story": "실제 블랙박스 주행의 6초 간격 프레임. 같은 순간의 전방 카메라와 후방 카메라로, 전방 영상으로 학습한 모델에 후방 영상이 잘못 들어온 상황입니다.",
               "look": "노출과 색이 가장 크게 바뀌고, 임베딩 지도에서 두 카메라의 프레임이 갈라집니다(임베딩 분리형). 지도 아래에 열어 볼 프레임이 나옵니다."}},
         real=True, folders=_dashcam("camera-swap"), needs="vision", options={"with_embeddings": True}),
    Case("next-stretch", "⟶", "DRIFT", {
        "en": {"title": "Same camera, the next stretch of road",
               "story": "The front camera, the first six minutes of the drive against the next six: expressway and open sky first, then more city streets.",
               "look": "Nothing broke, and it is still DRIFT: the scenery changed. A baseline has to cover the roads, hours and weather the camera will meet."},
        "ko": {"title": "같은 카메라, 다음 구간",
               "story": "전방 카메라, 주행의 앞 6분과 뒤 6분. 고가도로와 트인 하늘에서 시내 도로로 바뀝니다.",
               "look": "고장은 없는데도 DRIFT입니다. 풍경이 바뀐 것입니다. 기준 데이터는 카메라가 만날 도로·시간대·날씨를 덮어야 합니다."}},
         real=True, folders=_dashcam("next-stretch"), needs="vision", options={"with_embeddings": True}),
    Case("blurry-lens", "◌", "DRIFT", {
        "en": {"title": "A smudged lens, planted on real frames",
               "story": "The same stretch of the drive, frames three seconds apart; the current ones blurred as by a smudged or fogged lens.",
               "look": "Sharpness, noise and dynamic range move far more than anything else, while the scenery stayed the same. Sharpness reads as a shift with narrower spread: the blurred frames bunch up at low sharpness."},
        "ko": {"title": "실제 프레임에 심은 렌즈 흐림",
               "story": "주행의 같은 구간에서 3초씩 어긋난 프레임. 현재 쪽은 렌즈에 얼룩이나 김이 서린 것처럼 흐리게 했습니다.",
               "look": "선명도·잡음·명암 범위가 다른 무엇보다 크게 움직입니다. 풍경은 그대로입니다. 선명도는 이동·수축형으로 읽힙니다. 흐린 프레임들이 낮은 선명도 근처로 몰렸기 때문입니다."}},
         folders=_dashcam("blurry-lens"), needs="vision"),
    # text and audio: generated stories, judged by their plugins
    Case("app-update", "✉", "DRIFT", {
        "en": {"title": "Support tickets after an app update",
               "story": "Tickets to a note-taking app, an ordinary week against the week after version 5 shipped: sign-in and sync complaints crowd in.",
               "look": "Messages grow longer and angrier. Only part of the topics changed, so the weeks still mix on the embedding map, yet the report's current tickets least like the baseline are nearly all about version 5."},
        "ko": {"title": "앱 업데이트 뒤의 고객 문의",
               "story": "메모 앱에 들어온 문의, 평범한 한 주와 버전 5 출시 다음 주. 로그인·동기화 불만이 몰려옵니다.",
               "look": "글이 길어지고 격해집니다. 주제는 일부만 바뀌어 임베딩 지도에서 두 주가 섞이지만, 진단서가 꼽은 '기준과 가장 다른 현재 문의'는 거의 모두 버전 5 이야기입니다."}},
         folders=_folders(support, "app-update"), needs="text", options={"with_embeddings": True}),
    Case("bearing-wear", "⚙", "DRIFT", {
        "en": {"title": "A motor whose bearing starts to wear",
               "story": "One-second clips of an electric motor: healthy, then with knocks at the bearing's defect frequency ringing a high resonance.",
               "look": "The spectral centroid and zero-crossing rate rise: high frequencies the motor never made. Each measure is graphed, baseline against current."},
        "ko": {"title": "베어링 마모가 시작된 모터",
               "story": "전기 모터 소리의 1초 클립. 정상 상태와, 베어링 결함 주파수마다 충격음이 높은 공진을 울리는 상태.",
               "look": "스펙트럼 중심과 영교차율이 올라갑니다. 모터가 내지 않던 고주파가 생겼다는 뜻입니다. 값마다 기준과 현재의 그래프가 나옵니다."}},
         folders=_folders(machine, "bearing-wear"), needs="audio"),
    Case("louder-mic", "♪", "DRIFT", {
        "en": {"title": "Same motor, microphone moved closer",
               "story": "The same healthy motor recorded 2.2 times louder, as if the microphone was moved or its gain turned up.",
               "look": "The reading says unit or scale change: loudness and its spread grew by the same factor, the pitch measures stayed. Check the microphone before the machine."},
        "ko": {"title": "같은 모터, 가까이 옮긴 마이크",
               "story": "같은 정상 모터를 2.2배 크게 녹음했습니다. 마이크를 옮겼거나 이득을 올린 상황입니다.",
               "look": "판독은 단위·스케일 변경형입니다. 크기와 그 퍼짐이 같은 배율로 커졌고 음높이 쪽 값은 그대로입니다. 기계보다 마이크를 먼저 보세요."}},
         folders=_folders(machine, "louder-mic"), needs="audio"),
]
BY_ID = {c.id: c for c in CASES}


def build(case_id: str, out_dir: Path, *, seed: int = 5) -> Tuple[Path, Path]:
    """Write a case's data under ``out_dir``: ``ref.csv`` and ``cur.csv``,
    or for an image case ``ref/`` and ``cur/`` folders."""
    case = BY_ID[case_id]
    if case.folders is not None:
        return case.folders(out_dir)
    ref_rows, cur_rows = case.make(np.random.default_rng(seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    ref, cur = out_dir / "ref.csv", out_dir / "cur.csv"
    ref.write_text("\n".join([case.header, *ref_rows]) + "\n", encoding="utf-8")
    cur.write_text("\n".join([case.header, *cur_rows]) + "\n", encoding="utf-8")
    return ref, cur


def to_dict(case: Case, lang: str) -> Dict[str, Any]:
    text = case.text.get(lang) or case.text["en"]
    return {"id": case.id, "icon": case.icon, "verdict": case.verdict, "real": case.real,
            "needs": case.needs, "options": dict(case.options), **text}

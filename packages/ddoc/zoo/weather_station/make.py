"""Weather station — 실제 기상관측소 기록에서 고른 두 기간.

가정용 기상관측소의 10분 간격 실외 기록(2026-04-23 ~ 08-31)에서 이야기가 있는
두 기간을 잘라 data/ref.csv, data/cur.csv 로 씁니다. 데이터는 패키지에 들어 있어
(ddoc/examples/data/weather_station.csv.gz) 따로 내려받지 않습니다.

    python make.py                 # 장마 (기본)
    python make.py yearly-reset    # 연간 강우 누적값이 초기화된 날
    python make.py season          # 늦봄과 한여름
    python make.py next-days       # 나흘과 그다음 나흘

기대 판정: monsoon DRIFT · yearly-reset OK · season DRIFT · next-days OK
"""
from __future__ import annotations

import sys
from pathlib import Path

from ddoc.examples import weather


def make(out_dir: Path, story: str = "monsoon") -> tuple[Path, Path]:
    return weather.write_pair(story, Path(out_dir))


if __name__ == "__main__":
    story = sys.argv[1] if len(sys.argv) > 1 else "monsoon"
    if story not in weather.STORIES:
        raise SystemExit(f"unknown story {story!r}; one of: {', '.join(weather.STORIES)}")
    ref, cur = make(Path(__file__).parent / "data", story)
    print(f"{story}: {ref} · {cur}")

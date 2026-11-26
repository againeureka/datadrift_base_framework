"""Machine sound — 전기 모터 소리를 정답이 있게 만든 두 묶음 (오디오).

data/ref/, data/cur/ 에 각각 machine/clip_*.wav 40개(1초, 16kHz)와 ddoc.yaml 을 씁니다.
판정에는 audio 플러그인(librosa)이 필요합니다.

    python make.py                 # 베어링 마모 (기본): 정상 → 결함 충격음
    python make.py louder-mic      # 같은 소리, 마이크를 가까이 (2.2배 크게)
    python make.py two-shifts      # 주간과 야간 (대조군)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from ddoc.examples import machine


def make(out_dir: Path, story: str = "bearing-wear") -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    for side in ("ref", "cur"):
        shutil.rmtree(out_dir / side, ignore_errors=True)
    return machine.write_pair(story, out_dir)


if __name__ == "__main__":
    story = sys.argv[1] if len(sys.argv) > 1 else "bearing-wear"
    if story not in machine.STORIES:
        raise SystemExit(f"unknown story {story!r}; one of: {', '.join(machine.STORIES)}")
    ref, cur = make(Path(__file__).parent / "data", story)
    print(f"{story}: {ref} · {cur}")

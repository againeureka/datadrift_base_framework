"""Dashcam — 실제 블랙박스 주행의 프레임에서 고른 두 묶음.

2채널 블랙박스로 찍은 12분 주행에서 뽑아 줄인 프레임(전방·후방)으로 이야기가 있는 두 묶음을
data/ref/, data/cur/ 에 씁니다. 데이터는 패키지에 들어 있어
(ddoc/examples/data/dashcam.zip) 따로 내려받지 않습니다. 판정에는 vision 플러그인이 필요합니다.

    python make.py                 # 다른 카메라 (기본): 전방 → 같은 순간의 후방
    python make.py next-stretch    # 같은 카메라, 다음 구간
    python make.py same-stretch    # 같은 구간의 3초 뒤 프레임 (잡음 수준)
    python make.py blurry-lens     # 심은 변화: 렌즈 흐림
    python make.py low-quality     # 심은 변화: 절반 해상도·강한 압축
    python make.py overexposed     # 심은 변화: 밝은 노출
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from ddoc.examples import dashcam


def make(out_dir: Path, story: str = "camera-swap") -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    for side in ("ref", "cur"):
        shutil.rmtree(out_dir / side, ignore_errors=True)
    return dashcam.write_pair(story, out_dir)


if __name__ == "__main__":
    story = sys.argv[1] if len(sys.argv) > 1 else "camera-swap"
    if story not in dashcam.STORIES:
        raise SystemExit(f"unknown story {story!r}; one of: {', '.join(dashcam.STORIES)}")
    ref, cur = make(Path(__file__).parent / "data", story)
    print(f"{story}: {ref} · {cur}")

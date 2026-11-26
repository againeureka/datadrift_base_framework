"""Support tickets — 메모 앱 고객 문의를 정답이 있게 만든 두 묶음 (텍스트).

data/ref/, data/cur/ 에 각각 tickets/data.csv(id, text)와 ddoc.yaml 을 씁니다.
판정에는 text 플러그인(torch)이 필요합니다.

    python make.py                 # 앱 업데이트 (기본): 평범한 주 → 버전 5 출시 다음 주
    python make.py chat-widget     # 같은 주제, 채팅 창 말투
    python make.py two-weeks       # 평범한 두 주 (대조군)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from ddoc.examples import support


def make(out_dir: Path, story: str = "app-update") -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    for side in ("ref", "cur"):
        shutil.rmtree(out_dir / side, ignore_errors=True)
    return support.write_pair(story, out_dir)


if __name__ == "__main__":
    story = sys.argv[1] if len(sys.argv) > 1 else "app-update"
    if story not in support.STORIES:
        raise SystemExit(f"unknown story {story!r}; one of: {', '.join(support.STORIES)}")
    ref, cur = make(Path(__file__).parent / "data", story)
    print(f"{story}: {ref} · {cur}")

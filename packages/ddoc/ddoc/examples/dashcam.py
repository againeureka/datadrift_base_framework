"""Frames from a real dashcam drive, for image examples that feel real.

One afternoon drive of about 12 minutes on city expressways, recorded by
a two-camera dashcam: the front camera, and a rear one looking out of
the back window (warmer, purplish colour; following traffic; autumn
trees). ``zoo/dashcam/prepare.py`` made ``data/dashcam.zip`` from the
recordings: frames cut above the dashboard (no date, time or speed
overlay), shrunk to 320 × 140, named by seconds since the drive started.

    front/t0000.jpg … t0360.jpg   every 3 s, first half of the drive
    front/t0366.jpg … t0720.jpg   every 6 s, second half
    rear/t0000.jpg  … t0360.jpg   every 6 s, first half

A story is a baseline and a current set of frames; some plant a change
on real frames (a blurred lens, a lower-quality recording, a brighter
exposure), so their answer is known.

    from ddoc.examples import dashcam
    ref_dir, cur_dir = dashcam.write_pair("camera-swap", out_dir)
"""
from __future__ import annotations

import io
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

DATA = Path(__file__).with_name("data") / "dashcam.zip"
HALF = 360  # seconds: where the second half of the drive starts


def _second(name: str) -> int:
    return int(Path(name).stem[1:])


# The sets of frames the stories compare.
SETS: Dict[str, Callable[[str], bool]] = {
    # the front camera, first half, every 6 s
    "front": lambda n: n.startswith("front/") and _second(n) <= HALF and _second(n) % 6 == 0,
    # the same stretch, the frames 3 s later (interleaved with "front")
    "front-between": lambda n: n.startswith("front/") and _second(n) < HALF and _second(n) % 6 == 3,
    # the front camera, second half
    "front-later": lambda n: n.startswith("front/") and _second(n) > HALF,
    # the rear camera, first half: the same moments as "front"
    "rear": lambda n: n.startswith("rear/"),
}

# story → (baseline set, current set, change planted on the current frames)
STORIES: Dict[str, Tuple[str, str, Optional[str]]] = {
    # another camera's frames arrive: the rear one, at the same moments
    "camera-swap": ("front", "rear", None),
    # the same camera, the next six minutes of the drive
    "next-stretch": ("front", "front-later", None),
    # the same camera and stretch, the frames in between: nothing changed
    "same-stretch": ("front", "front-between", None),
    # planted on the in-between frames: a smudged or fogged lens
    "blurry-lens": ("front", "front-between", "blur"),
    # planted: a lower-quality recording (half the detail, heavy compression)
    "low-quality": ("front", "front-between", "low-quality"),
    # planted: a brighter exposure setting
    "overexposed": ("front", "front-between", "bright"),
}


@lru_cache(maxsize=1)
def _frames() -> Dict[str, bytes]:
    with zipfile.ZipFile(DATA) as zf:
        return {n: zf.read(n) for n in sorted(zf.namelist()) if n.endswith(".jpg")}


def frames(set_name: str) -> List[Tuple[str, bytes]]:
    """``(name, jpeg bytes)`` of one set, in drive order."""
    keep = SETS[set_name]
    return [(n, b) for n, b in _frames().items() if keep(n)]


def _plant(jpeg: bytes, change: str) -> bytes:
    from PIL import Image, ImageEnhance, ImageFilter

    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    quality = 90
    if change == "blur":
        img = img.filter(ImageFilter.GaussianBlur(2.2))
    elif change == "low-quality":
        small = img.resize((img.width // 2, img.height // 2), Image.BILINEAR)
        img, quality = small.resize(img.size, Image.BILINEAR), 12
    elif change == "bright":
        img = ImageEnhance.Brightness(img).enhance(1.6)
    else:
        raise ValueError(f"unknown change: {change!r}")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def write_set(folder: Path, set_name: str, change: Optional[str] = None) -> Path:
    """Write one set of frames into ``folder`` (``front-t0012.jpg`` …)."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in frames(set_name):
        target = folder / name.replace("/", "-")
        target.write_bytes(_plant(data, change) if change else data)
    return folder


def write_pair(story: str, out_dir: Path) -> Tuple[Path, Path]:
    """Write ``ref/`` and ``cur/`` folders of a story under ``out_dir``."""
    ref_set, cur_set, change = STORIES[story]
    out_dir = Path(out_dir)
    return write_set(out_dir / "ref", ref_set), write_set(out_dir / "cur", cur_set, change)

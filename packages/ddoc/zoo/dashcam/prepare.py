"""Make ``ddoc/examples/data/dashcam.zip`` from a dashcam's own recordings.

The source is one afternoon drive (about 12 minutes) recorded by a
two-camera dashcam as ``.jdr`` files: a front camera (1080p, 30 fps) and
a rear one looking out of the back window (1080p, 2 fps), interleaved
with GPS and motion-sensor records in chunks of

    tag (4: "00VI" "00VP" front · "01VI" "01VP" rear · "00GP" GPS · "00SE" sensor)
    size (4, little endian) · 20-byte header (…, year, month, weekday, day,
    hour, minute, second, millisecond as u16) · payload (H.264 for video)

with padding between some chunks. Only the two video streams are read;
the GPS and sensor records are never decoded.

Each kept frame is cut to its top 840 rows (the dashboard, the camera's
overlay of date, time and speed, and anything hanging there are gone),
shrunk to 320 × 140 and saved as JPEG without metadata. At that size
number plates and faces are not legible. Frames are named by seconds
since the drive started, not by the time of day.

Kept: the front camera every 3 s in the first half of the drive and
every 6 s in the second, the rear camera every 6 s in the first half,
enough for the stories in ``ddoc.examples.dashcam``.

    python prepare.py <folder of .jdr files>     # needs ffmpeg and Pillow
"""
from __future__ import annotations

import io
import struct
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

OUT = Path(__file__).resolve().parents[2] / "ddoc" / "examples" / "data" / "dashcam.zip"
FPS = {"00": 30, "01": 2}           # front, rear
KEEP_ROWS = 840                      # of 1080: above the dashboard
SIZE = (320, 140)
QUALITY = 70


def chunks(data: bytes) -> Iterator[Tuple[str, str, datetime, bytes]]:
    """(stream, kind, time, payload) for each chunk, skipping padding."""
    i = data.find(b"00VI")
    while 0 <= i <= len(data) - 28:
        tag = data[i:i + 4]
        size = struct.unpack("<I", data[i + 4:i + 8])[0]
        year, month, _, day, hour, minute, second, ms = struct.unpack("<8H", data[i + 12:i + 28])
        if tag[:2].isdigit() and tag[2:3].isalpha() and year == 2026 and 0 < size < 8_000_000 \
                and i + 28 + size <= len(data):
            when = datetime(year, month, day, hour, minute, second, ms * 1000)
            yield tag[:2].decode(), tag[2:3].decode(), when, data[i + 28:i + 28 + size]
            i += 28 + size
            continue
        # padding: on to the next video chunk
        found = [j for j in (data.find(t, i + 1) for t in (b"00VI", b"00VP", b"01VI", b"01VP")) if j > 0]
        i = min(found) if found else -1


def frames(path: Path, stream: str, pick: List[int], tmp: Path) -> List[bytes]:
    """Decode frames ``pick`` (indices) of one stream as PNG bytes."""
    raw = tmp / f"{path.stem}-{stream}.h264"
    with raw.open("wb") as fh:
        for s, kind, _, body in chunks(path.read_bytes()):
            if s == stream and kind == "V":
                fh.write(body)
    if not pick:
        return []
    expr = "+".join(f"eq(n\\,{n})" for n in pick)
    pattern = tmp / f"{path.stem}-{stream}-%04d.png"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "h264",
                    "-framerate", str(FPS[stream]), "-i", str(raw),
                    "-vf", f"select='{expr}'", "-fps_mode", "vfr", str(pattern)], check=True)
    out = sorted(tmp.glob(f"{path.stem}-{stream}-*.png"))
    return [p.read_bytes() for p in out]


def shrink(png: bytes) -> bytes:
    from PIL import Image

    img = Image.open(io.BytesIO(png)).convert("RGB")
    img = img.crop((0, 0, img.width, KEEP_ROWS)).resize(SIZE, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=QUALITY, optimize=True)
    return buf.getvalue()


def main(src: Path) -> None:
    files = sorted(src.glob("*.jdr"))
    times: Dict[Tuple[Path, str], List[datetime]] = {}
    for f in files:
        for s, kind, when, _ in chunks(f.read_bytes()):
            if kind == "V":
                times.setdefault((f, s), []).append(when)
    start = min(t[0] for t in times.values())
    end = max(t[-1] for t in times.values())
    half = (end - start).total_seconds() / 2
    print(f"{len(files)} files, {start:%H:%M:%S} – {end:%H:%M:%S}, half at {half:.0f} s")

    def wanted(stream: str, sec: float) -> int:
        """The sampling step for this stream at this moment, 0 = none."""
        if stream == "00":
            return 3 if sec < half else 6
        return 6 if sec < half else 0

    with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as zf:
        tmp = Path(td)
        kept = {"00": 0, "01": 0}
        seen: Dict[str, set] = {"00": set(), "01": set()}   # seconds taken, across files
        for (f, stream), stamps in sorted(times.items()):
            pick, names = [], []
            for n, when in enumerate(stamps):
                sec = (when - start).total_seconds()
                step = wanted(stream, sec)
                slot = int(sec // step) if step else None
                if slot is None or slot * step in seen[stream]:
                    continue
                seen[stream].add(slot * step)
                pick.append(n)
                names.append(f"{'front' if stream == '00' else 'rear'}/t{slot * step:04d}.jpg")
            for name, png in zip(names, frames(f, stream, pick, tmp)):
                zf.writestr(name, shrink(png))
                kept[stream] += 1
    print(f"front {kept['00']} · rear {kept['01']} frames → {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python prepare.py <folder of .jdr files>")
    main(Path(sys.argv[1]).expanduser())

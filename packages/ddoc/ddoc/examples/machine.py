"""Machine sounds, generated: an electric motor running, and what can
change in what a microphone hears of it.

Each clip is one second at 16 kHz: the motor's mains hum (120 Hz and its
harmonics) and rotation (about 29.5 Hz), slightly different from clip to
clip as the load moves, over a floor of broadband noise. A worn bearing
adds sharp knocks at its defect frequency, each ringing a high
resonance of the housing; a microphone moved closer or set to a higher
gain makes everything louder and nothing else. The stories are built
from a fixed seed, so their answers are known.

    from ddoc.examples import machine
    ref_dir, cur_dir = machine.write_pair("bearing-wear", out_dir)

Clips are written as 16-bit WAV with the standard library; judging them
needs the audio plugin (librosa).
"""
from __future__ import annotations

import wave
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

RATE = 16_000
CLIPS = 40

# story → (baseline condition, current condition)
STORIES: Dict[str, Tuple[str, str]] = {
    # a bearing starts to wear: knocks ring a high resonance
    "bearing-wear": ("healthy", "worn"),
    # the microphone was moved closer (or its gain turned up): same sound, louder
    "louder-mic": ("healthy", "louder"),
    # the day shift against the night shift, same machine, nothing changed
    "two-shifts": ("healthy", "healthy"),
}


def _clip(rng: np.random.Generator, condition: str) -> np.ndarray:
    t = np.arange(RATE) / RATE
    load = rng.uniform(0.9, 1.1)
    rot = 29.5 * rng.uniform(0.995, 1.005)
    y = np.zeros_like(t)
    for k, amp in ((1, 0.20), (2, 0.08), (3, 0.04), (4, 0.02)):        # mains hum, 120 Hz harmonics
        y += amp * load * rng.uniform(0.85, 1.15) * np.sin(2 * np.pi * 120 * k * t + rng.uniform(0, 2 * np.pi))
    y += 0.06 * np.sin(2 * np.pi * rot * t + rng.uniform(0, 2 * np.pi))   # rotation
    y += 0.03 * rng.standard_normal(t.size)                               # broadband floor
    if condition == "worn":
        # knocks at the outer-race defect frequency (about 3.6 x rotation),
        # each ringing the housing near 3.2 kHz for a few milliseconds
        period = 1 / (3.58 * rot)
        ring_t = np.arange(int(0.006 * RATE)) / RATE
        ring = np.exp(-ring_t / 0.0015) * np.sin(2 * np.pi * 3200 * ring_t)
        for start in np.arange(rng.uniform(0, period), 1.0, period):
            i = int(start * RATE)
            seg = ring[: max(0, min(ring.size, t.size - i))]
            y[i:i + seg.size] += 0.35 * rng.uniform(0.7, 1.3) * seg
        y += 0.015 * rng.standard_normal(t.size)
    elif condition == "louder":
        y *= 2.2
    elif condition != "healthy":
        raise ValueError(f"unknown condition: {condition!r}")
    return np.clip(y, -1.0, 1.0)


def _write_wav(path: Path, y: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((y * 32767).astype("<i2").tobytes())


def write_set(folder: Path, condition: str, *, seed: int, n: int = CLIPS) -> Path:
    """``n`` clips of one condition under ``folder/machine/`` with the
    ``ddoc.yaml`` that tells ddoc they are audio."""
    data = Path(folder) / "machine"
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        _write_wav(data / f"clip_{i:03d}.wav", _clip(rng, condition))
    (data / "ddoc.yaml").write_text("modality: audio\nformats: [.wav]\n", encoding="utf-8")
    return Path(folder)


def write_pair(story: str, out_dir: Path, *, seed: Optional[int] = None) -> Tuple[Path, Path]:
    """Write ``ref/`` and ``cur/`` of a story under ``out_dir``."""
    before, after = STORIES[story]
    seed = 11 if seed is None else seed
    out_dir = Path(out_dir)
    return (write_set(out_dir / "ref", before, seed=seed),
            write_set(out_dir / "cur", after, seed=seed + 1))

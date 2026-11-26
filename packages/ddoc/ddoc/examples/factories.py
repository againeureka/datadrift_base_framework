"""Toy-data factories — single source for both pytest fixtures and the
``ddoc examples generate`` user-facing CLI.

These datasets used to be generated ad-hoc in ``/tmp`` by hand during
early development; this module promotes them to permanent factories
so the same code path drives:

* the regression test suite (through ``tests/fixtures/factories.py``), and
* the user onboarding command (``ddoc examples generate <modality>``).

Each factory writes a single dataset directory containing the file(s)
the relevant ddoc plugin expects plus a ``ddoc.yaml`` declaring
modality. Pair-of-(ref, cur) construction lives one level up
(``conftest.py`` fixtures and the ``ddoc examples`` command), keeping
factories themselves narrow and reusable.
"""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _write_yaml(path: Path, fields: Dict[str, Any]) -> None:
    """Tiny YAML writer — stays free of pyyaml so factories work even in
    the lightest install (PyYAML is in core deps but tests run before
    it's necessarily imported)."""
    lines = []
    for key, value in fields.items():
        if isinstance(value, list):
            inner = ", ".join(str(v) for v in value)
            lines.append(f"{key}: [{inner}]")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n")


# ── Timeseries ────────────────────────────────────────────────────────


def make_toy_timeseries(
    out_dir: str | Path,
    *,
    start: str = "2026-06-21",
    end: str = "2026-07-05",
    dataset_name: str = "weather",
) -> Path:
    """Write a timeseries dataset (CSV + ddoc.yaml) under
    ``<out_dir>/<dataset_name>/``: the days ``[start, end)`` of a real home
    weather station's log (:mod:`ddoc.examples.weather`), ten-minute steps.

    Schema: ``time`` plus the station's measurements (temperature,
    humidity, dew point, wind, pressure, light, UV, rain in the last
    hour). Its running rain totals are left out: they only grow, so they
    would differ between any two periods. Returns the dataset directory.
    """
    from . import weather

    dataset_dir = Path(out_dir) / dataset_name
    weather.write_window(dataset_dir / "data.csv", start, end, weather.MEASUREMENTS)
    _write_yaml(
        dataset_dir / "ddoc.yaml",
        {
            "modality": "timeseries",
            "csv_file": "data.csv",
            "timestamp_column": "time",
            "numeric_columns": list(weather.MEASUREMENTS),
        },
    )
    return dataset_dir


# ── Audio ─────────────────────────────────────────────────────────────


def make_toy_audio(
    out_dir: str | Path,
    *,
    freq: float = 440.0,
    sr: int = 8000,
    duration: float = 1.0,
    amplitude: float = 0.3,
    dataset_name: str = "toy_audio",
) -> Path:
    """Write a tiny single-tone WAV (sine at ``freq`` Hz) + ddoc.yaml
    under ``<out_dir>/<dataset_name>/``.

    Soundfile and numpy are imported lazily because the audio plugin
    isn't always installed; callers needing audio fixtures install
    ``ddoc-plugin-audio`` first (which already pulls librosa/numpy).
    """
    import numpy as np
    import soundfile as sf

    base = Path(out_dir)
    dataset_dir = base / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    n_samples = int(sr * duration)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)
    wave = amplitude * np.sin(2.0 * math.pi * freq * t)
    sf.write(str(dataset_dir / "tone.wav"), wave, sr)

    _write_yaml(
        dataset_dir / "ddoc.yaml",
        {"modality": "audio", "formats": [".wav"]},
    )
    return dataset_dir


# ── Text ──────────────────────────────────────────────────────────────


_VOCAB_SHORT = ["yes", "ok", "good", "fine", "thanks", "great", "nice", "sure"]
_VOCAB_LONG = [
    "The quick brown fox jumps over the lazy dog repeatedly throughout the day in the meadow.",
    "Comprehensive analysis indicates significant divergence between observed and expected outcomes here.",
    "Implementation of the algorithm requires careful consideration of memory constraints and threading.",
    "Subsequently, the proposed methodology demonstrates substantial improvements over baseline approaches.",
]


def make_toy_text(
    out_dir: str | Path,
    *,
    vocab: str = "short",
    n: int = 20,
    seed: int = 42,
    dataset_name: str = "toy_text",
) -> Path:
    """Write a tiny text dataset (CSV with id,text columns) +
    ddoc.yaml under ``<out_dir>/<dataset_name>/``.

    ``vocab`` selects the pool: ``"short"`` for terse one-word strings
    (low avg length, low vocabulary diversity) or ``"long"`` for
    multi-clause sentences. Pairing short-vs-long is the canonical
    drift scenario for the text plugin.
    """
    if vocab not in ("short", "long"):
        raise ValueError(f"vocab must be 'short' or 'long', got {vocab!r}")
    pool = _VOCAB_SHORT if vocab == "short" else _VOCAB_LONG

    rng = random.Random(seed)
    base = Path(out_dir)
    dataset_dir = base / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    csv_path = dataset_dir / "data.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "text"])
        for i in range(n):
            w.writerow([i, rng.choice(pool)])

    _write_yaml(
        dataset_dir / "ddoc.yaml",
        {
            "modality": "text",
            "text_column": "text",
            "id_column": "id",
            "language": "english",
        },
    )
    return dataset_dir


# ── Vision ────────────────────────────────────────────────────────────


def make_toy_vision(
    out_dir: str | Path,
    *,
    n: int = 40,
    seed: int = 1,
    shifted: bool = False,
    size: int = 96,
    color: Optional[Tuple[int, int, int]] = None,
    dataset_name: str = "toy_vision",
) -> Path:
    """Write ``n`` PNGs + ddoc.yaml under ``<out_dir>/<dataset_name>/``.

    Each image is a light background with 2–5 rectangles and circles of
    random colours, so the set has real variety (brightness, colour,
    edges, embeddings). ``shifted=True`` darkens the images and adds a
    blue cast — the kind of change a new camera or lighting brings.
    ``color`` keeps the old behaviour: ``n`` identical solid images.

    Pillow is imported lazily; callers needing vision fixtures install
    ``ddoc-plugin-vision`` first (which pulls Pillow and friends).
    """
    from PIL import Image, ImageDraw, ImageEnhance

    base = Path(out_dir)
    dataset_dir = base / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    for i in range(n):
        if color is not None:
            im = Image.new("RGB", (size, size), color)
        else:
            im = Image.new("RGB", (size, size), tuple(rng.randint(170, 255) for _ in range(3)))
            draw = ImageDraw.Draw(im)
            for _ in range(rng.randint(2, 5)):
                x, y, w = rng.randint(0, size - 20), rng.randint(0, size - 20), rng.randint(8, size // 3)
                fill = tuple(rng.randint(0, 255) for _ in range(3))
                (draw.ellipse if rng.random() < 0.5 else draw.rectangle)([x, y, x + w, y + w], fill=fill)
            if shifted:
                im = ImageEnhance.Brightness(im).enhance(0.45)
                r, g, b = im.split()
                im = Image.merge("RGB", (r, g, b.point(lambda v: min(255, v + 60))))
        im.save(dataset_dir / f"img_{i:03d}.png")

    _write_yaml(
        dataset_dir / "ddoc.yaml",
        {"modality": "image", "formats": [".png"]},
    )
    return dataset_dir


# ── Categorical (demo path) ───────────────────────────────────────────


# Default distributions used by the categorical pair builder. Shape
# matches a typical camera-stats-window dict emitted by a monitoring
# application — the demo doubles as a regression check on the
# categorical shape.
_CATEGORICAL_REF = {
    "color_distribution":  {"red": 30, "blue": 10, "white": 5},
    "type_distribution":   {"sedan": 60, "suv": 30, "truck": 10},
    "hourly_distribution": {str(h): 0 for h in range(24)} | {
        "8": 12, "9": 18, "12": 25, "17": 22, "18": 30, "19": 15,
    },
    "confidence_stats":    {"mean": 0.85, "std": 0.07},
}

_CATEGORICAL_CUR_SHIFTED = {
    "color_distribution":  {"red": 5, "blue": 25, "white": 20},
    "type_distribution":   {"sedan": 50, "suv": 30, "truck": 20},
    "hourly_distribution": {str(h): 0 for h in range(24)} | {
        "8": 8, "9": 12, "12": 30, "17": 14, "18": 20, "19": 10,
        "21": 18, "22": 15,
    },
    "confidence_stats":    {"mean": 0.78, "std": 0.10},
}


def make_toy_categorical(
    out_dir: str | Path,
    *,
    distributions: Dict[str, Any] | None = None,
) -> Path:
    """Write a categorical-drift dataset directly under ``out_dir``.

    The plugin (``ddoc-plugin-categorical``) reads
    ``distributions.json`` directly from ``data_path`` — no
    sub-directory walking. So ``out_dir`` IS the dataset dir.
    ``distributions`` defaults to the canonical "ref" shape — pass an
    alternative for the "current" side.
    """
    import json as _json
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    payload = distributions if distributions is not None else _CATEGORICAL_REF
    (base / "distributions.json").write_text(
        _json.dumps(payload, indent=2), encoding="utf-8",
    )
    _write_yaml(
        base / "ddoc.yaml",
        {"modality": "categorical",
         "distributions_file": "distributions.json"},
    )
    return base


# ── Pair builders (the user-facing scenario level) ────────────────────


_REF_DIR = "ref"
_CUR_DIR = "cur"


def make_pair_timeseries(out_dir: str | Path, *, scenario: str = "shifted") -> Tuple[Path, Path]:
    """Return ``(ref_dir, cur_dir)`` for a timeseries drift scenario, cut
    from a real weather station's log.

    ``scenario="shifted"``: the two weeks before the rainy season against
    the two weeks after it set in — dew point and humidity step up.
    ``scenario="identical"``: the same two weeks on both sides — drift
    score should be near-zero (sanity check).
    """
    from .weather import STORIES

    base = Path(out_dir)
    (r0, r1), (c0, c1) = STORIES["monsoon"]
    if scenario == "shifted":
        make_toy_timeseries(base / _REF_DIR, start=r0, end=r1)
        make_toy_timeseries(base / _CUR_DIR, start=c0, end=c1)
    elif scenario == "identical":
        make_toy_timeseries(base / _REF_DIR, start=r0, end=r1)
        make_toy_timeseries(base / _CUR_DIR, start=r0, end=r1)
    else:
        raise ValueError(f"unsupported scenario: {scenario!r}")
    return base / _REF_DIR, base / _CUR_DIR


def make_pair_audio(out_dir: str | Path, *, scenario: str = "shifted") -> Tuple[Path, Path]:
    """Return ``(ref_dir, cur_dir)`` for an audio drift scenario.

    ``scenario="shifted"``: 440 Hz vs 880 Hz (octave apart — clear
    spectral drift). ``scenario="identical"``: both 440 Hz.
    """
    base = Path(out_dir)
    if scenario == "shifted":
        make_toy_audio(base / _REF_DIR, freq=440.0)
        make_toy_audio(base / _CUR_DIR, freq=880.0)
    elif scenario == "identical":
        make_toy_audio(base / _REF_DIR, freq=440.0)
        make_toy_audio(base / _CUR_DIR, freq=440.0)
    else:
        raise ValueError(f"unsupported scenario: {scenario!r}")
    return base / _REF_DIR, base / _CUR_DIR


def make_pair_text(out_dir: str | Path, *, scenario: str = "shifted") -> Tuple[Path, Path]:
    """Return ``(ref_dir, cur_dir)`` for a text drift scenario.

    ``scenario="shifted"``: ref uses the short vocab pool, cur uses
    the long sentences pool. ``scenario="identical"``: both short.
    """
    base = Path(out_dir)
    if scenario == "shifted":
        make_toy_text(base / _REF_DIR, vocab="short", seed=42)
        make_toy_text(base / _CUR_DIR, vocab="long", seed=43)
    elif scenario == "identical":
        make_toy_text(base / _REF_DIR, vocab="short", seed=42)
        make_toy_text(base / _CUR_DIR, vocab="short", seed=42)
    else:
        raise ValueError(f"unsupported scenario: {scenario!r}")
    return base / _REF_DIR, base / _CUR_DIR


def make_pair_vision(out_dir: str | Path, *, scenario: str = "shifted") -> Tuple[Path, Path]:
    """Return ``(ref_dir, cur_dir)`` for a vision drift scenario.

    ``scenario="shifted"``: 40 varied images vs 40 others, darker with a
    blue cast. ``scenario="identical"``: the same 40 images on both sides.
    """
    base = Path(out_dir)
    if scenario == "shifted":
        make_toy_vision(base / _REF_DIR, seed=1)
        make_toy_vision(base / _CUR_DIR, seed=2, shifted=True)
    elif scenario == "identical":
        make_toy_vision(base / _REF_DIR, seed=1)
        make_toy_vision(base / _CUR_DIR, seed=1)
    else:
        raise ValueError(f"unsupported scenario: {scenario!r}")
    return base / _REF_DIR, base / _CUR_DIR


def make_pair_categorical(out_dir: str | Path, *, scenario: str = "shifted") -> Tuple[Path, Path]:
    """Return ``(ref_dir, cur_dir)`` for a categorical-distribution
    drift scenario.

    ``scenario="shifted"``: ref vs cur with clear shifts in
    ``color_distribution`` / ``type_distribution`` / ``hourly_distribution``
    — a typical vehicle-fingerprint shape as emitted by a
    camera-stats monitoring application.

    ``scenario="identical"``: ref and cur identical — drift score
    should be near-zero.
    """
    base = Path(out_dir)
    if scenario == "shifted":
        make_toy_categorical(base / _REF_DIR, distributions=_CATEGORICAL_REF)
        make_toy_categorical(base / _CUR_DIR, distributions=_CATEGORICAL_CUR_SHIFTED)
    elif scenario == "identical":
        make_toy_categorical(base / _REF_DIR, distributions=_CATEGORICAL_REF)
        make_toy_categorical(base / _CUR_DIR, distributions=_CATEGORICAL_REF)
    else:
        raise ValueError(f"unsupported scenario: {scenario!r}")
    return base / _REF_DIR, base / _CUR_DIR


def make_pair_weather(out_dir: str | Path, *, scenario: str = "shifted") -> Tuple[Path, Path]:
    """Return ``(ref.csv, cur.csv)``: two windows of the real
    weather-station log with every column, rain totals included.

    ``scenario`` is one of the stories in :data:`ddoc.examples.weather.STORIES`
    (``monsoon``, ``yearly-reset``, ``season``, ``next-days``);
    ``shifted`` is ``monsoon``; ``identical`` is monsoon's baseline twice.
    """
    from . import weather

    story = "monsoon" if scenario in ("shifted", "identical") else scenario
    if story not in weather.STORIES:
        raise ValueError(f"unsupported scenario: {scenario!r}")
    base = Path(out_dir)
    if scenario == "identical":
        (r0, r1), _ = weather.STORIES[story]
        return (weather.write_window(base / "ref.csv", r0, r1),
                weather.write_window(base / "cur.csv", r0, r1))
    return weather.write_pair(story, base)


def make_pair_dashcam(out_dir: str | Path, *, scenario: str = "shifted") -> Tuple[Path, Path]:
    """Return ``(ref_dir, cur_dir)``: two folders of frames from a real
    dashcam drive.

    ``scenario`` is one of the stories in :data:`ddoc.examples.dashcam.STORIES`
    (``camera-swap``, ``next-stretch``, ``same-stretch``, ``blurry-lens``,
    ``low-quality``, ``overexposed``); ``shifted`` is ``camera-swap``;
    ``identical`` is ``same-stretch``.
    """
    from . import dashcam

    story = {"shifted": "camera-swap", "identical": "same-stretch"}.get(scenario, scenario)
    if story not in dashcam.STORIES:
        raise ValueError(f"unsupported scenario: {scenario!r}")
    return dashcam.write_pair(story, Path(out_dir))


def make_pair_support(out_dir: str | Path, *, scenario: str = "shifted") -> Tuple[Path, Path]:
    """Return ``(ref_dir, cur_dir)``: two folders of generated support
    tickets (text). ``scenario`` is a story in
    :data:`ddoc.examples.support.STORIES` (``app-update``, ``chat-widget``,
    ``two-weeks``); ``shifted`` is ``app-update``, ``identical`` ``two-weeks``."""
    from . import support

    story = {"shifted": "app-update", "identical": "two-weeks"}.get(scenario, scenario)
    if story not in support.STORIES:
        raise ValueError(f"unsupported scenario: {scenario!r}")
    return support.write_pair(story, Path(out_dir))


def make_pair_machine(out_dir: str | Path, *, scenario: str = "shifted") -> Tuple[Path, Path]:
    """Return ``(ref_dir, cur_dir)``: two folders of generated machine
    sounds (audio). ``scenario`` is a story in
    :data:`ddoc.examples.machine.STORIES` (``bearing-wear``, ``louder-mic``,
    ``two-shifts``); ``shifted`` is ``bearing-wear``, ``identical`` ``two-shifts``."""
    from . import machine

    story = {"shifted": "bearing-wear", "identical": "two-shifts"}.get(scenario, scenario)
    if story not in machine.STORIES:
        raise ValueError(f"unsupported scenario: {scenario!r}")
    return machine.write_pair(story, Path(out_dir))


# Index for the ``ddoc examples`` CLI to enumerate.
PAIR_BUILDERS = {
    "weather": make_pair_weather,
    "dashcam": make_pair_dashcam,
    "support": make_pair_support,
    "machine": make_pair_machine,
    "timeseries": make_pair_timeseries,
    "audio": make_pair_audio,
    "text": make_pair_text,
    "vision": make_pair_vision,
    "categorical": make_pair_categorical,
}

SUPPORTED_SCENARIOS = ("shifted", "identical")


def scenarios_of(modality: str) -> Tuple[str, ...]:
    """The scenarios one modality's builder takes."""
    if modality == "weather":
        from .weather import STORIES
        return (*SUPPORTED_SCENARIOS, *STORIES)
    stories = {"dashcam": "dashcam", "support": "support", "machine": "machine"}.get(modality)
    if stories:
        import importlib
        return (*SUPPORTED_SCENARIOS, *importlib.import_module(f".{stories}", __package__).STORIES)
    return SUPPORTED_SCENARIOS


def pair_paths(modality: str, out_dir: str | Path) -> Tuple[Path, Path]:
    """Where a builder puts its pair: two CSV files for ``weather``,
    ``ref/`` and ``cur/`` folders for the others."""
    base = Path(out_dir)
    if modality == "weather":
        return base / "ref.csv", base / "cur.csv"
    return base / _REF_DIR, base / _CUR_DIR

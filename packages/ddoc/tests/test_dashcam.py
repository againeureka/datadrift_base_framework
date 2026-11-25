"""The dashcam example: what the shipped frames are (and are not), the
stories built from them, and how ddoc reads them when the vision plugin
is there."""
from __future__ import annotations

import importlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from ddoc.cli.main import app
from ddoc.examples import cases, dashcam, factories

runner = CliRunner()


def _vision_ready() -> bool:
    try:
        import importlib.metadata as md
        if not any(ep.name == "ddoc_vision" for ep in md.entry_points(group="ddoc")):
            return False
        importlib.import_module("ddoc_plugin_vision")
        return True
    except Exception:  # installed but torch or another dependency missing
        return False


def _sharpness(path: Path) -> float:
    g = np.asarray(Image.open(path).convert("L"), dtype=float)
    return float(np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean())


# ── The shipped frames ─────────────────────────────────────────────────


def test_frames_are_small_bare_and_named_by_drive_time():
    with zipfile.ZipFile(dashcam.DATA) as zf:
        names = [n for n in zf.namelist() if n.endswith(".jpg")]
        assert sum(n.startswith("front/") for n in names) == 181
        assert sum(n.startswith("rear/") for n in names) == 61
        for name in names[::20]:
            img = Image.open(io.BytesIO(zf.read(name)))
            assert img.size == (320, 140)                 # too small for plates or faces
            assert not img.getexif() and "exif" not in img.info
    # seconds since the drive started, never a time of day
    assert all(Path(n).stem[0] == "t" and Path(n).stem[1:].isdigit() and int(Path(n).stem[1:]) <= 726
               for n in names)


@pytest.mark.parametrize("story,n_ref,n_cur", [
    ("camera-swap", 61, 61), ("next-stretch", 61, 60), ("same-stretch", 61, 60), ("blurry-lens", 61, 60),
])
def test_stories_write_two_folders(tmp_path, story, n_ref, n_cur):
    ref, cur = factories.make_pair_dashcam(tmp_path, scenario=story)
    assert (len(list(ref.glob("*.jpg"))), len(list(cur.glob("*.jpg")))) == (n_ref, n_cur)
    assert factories.pair_paths("dashcam", tmp_path) == (ref, cur)


def test_planted_changes_land_on_the_same_frames(tmp_path):
    _, plain = dashcam.write_pair("same-stretch", tmp_path / "plain")
    _, blurred = dashcam.write_pair("blurry-lens", tmp_path / "blur")
    _, bright = dashcam.write_pair("overexposed", tmp_path / "bright")
    name = sorted(p.name for p in plain.glob("*.jpg"))[10]
    assert _sharpness(blurred / name) < 0.6 * _sharpness(plain / name)
    lum = lambda p: float(np.asarray(Image.open(p).convert("L")).mean())  # noqa: E731
    assert lum(bright / name) > 1.2 * lum(plain / name)


def test_image_cases_build_folders_and_carry_their_options(tmp_path):
    swap = cases.BY_ID["camera-swap"]
    ref, cur = cases.build("camera-swap", tmp_path)
    assert ref.is_dir() and len(list(cur.glob("*.jpg"))) == 61
    d = cases.to_dict(swap, "ko")
    assert d["needs"] == "vision" and d["options"] == {"with_embeddings": True} and d["real"]
    assert cases.to_dict(cases.BY_ID["monsoon"], "en")["needs"] is None


def test_without_the_vision_plugin_probe_says_blind_first(tmp_path, monkeypatch):
    if _vision_ready():
        pytest.skip("the vision plugin is here; this checks the other case")
    ref, cur = factories.make_pair_dashcam(tmp_path, scenario="camera-swap")
    res = runner.invoke(app, ["probe", str(ref), str(cur), "--json"])
    assert res.exit_code == 2 and json.loads(res.output.strip().splitlines()[-1])["blind"]


# ── Read by the vision plugin ──────────────────────────────────────────

vision = pytest.mark.skipif(not _vision_ready(), reason="needs the vision plugin with torch")


def _diff(ref: Path, cur: Path, *extra: str) -> dict:
    res = runner.invoke(app, ["diff", str(ref), str(cur), "--json", *extra])
    return json.loads(res.output.strip().splitlines()[-1])


@vision
def test_another_camera_separates_on_the_embedding_map(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ref, cur = factories.make_pair_dashcam(tmp_path / "d", scenario="camera-swap")
    out = _diff(ref, cur, "--with-embeddings")
    assert out["verdict"] == "DRIFT"
    codes = {c["code"] for c in out["diagnosis"]["archetypes"] if c["match"] == "match"}
    assert "embedding_shift" in codes
    assert out["top_attributes"][0][0] in ("exposure", "colorfulness")


@vision
def test_a_planted_blur_stands_far_above_the_noise(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    same = _diff(*factories.make_pair_dashcam(tmp_path / "same", scenario="same-stretch"))
    blur = _diff(*factories.make_pair_dashcam(tmp_path / "blur", scenario="blurry-lens"))
    assert same["verdict"] == "OK" and blur["verdict"] == "DRIFT"
    assert blur["overall_score"] > 5 * same["overall_score"]
    assert {a for a, _ in blur["top_attributes"][:3]} & {"sharpness", "gaussian_noise_level", "dynamic_range"}
    # the plugin hands over its values per image, so attributes get a reading too
    narrow = [c for c in blur["diagnosis"]["archetypes"] if c["code"] == "shift_and_narrow" and c["match"] == "match"]
    assert narrow and "sharpness" in narrow[0]["attrs"]
    assert "attribute_values" not in blur.get("raw", {})


def test_zoo_make_writes_a_story_and_starts_clean(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("zoo_dashcam", Path(__file__).parents[1] / "zoo" / "dashcam" / "make.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.make(tmp_path, "camera-swap")
    ref, cur = mod.make(tmp_path, "next-stretch")       # a second story replaces the first
    assert len(list(cur.glob("*.jpg"))) == 60 and not list(cur.glob("rear-*.jpg"))


@vision
@pytest.mark.parametrize("case", [c for c in cases.CASES if c.folders is not None], ids=lambda c: c.id)
def test_image_cases_get_the_verdict_they_promise(case, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ref, cur = cases.build(case.id, tmp_path / "d")
    extra = ["--with-embeddings"] if case.options.get("with_embeddings") else []
    assert _diff(ref, cur, *extra)["verdict"] == case.verdict
    for lang in ("en", "ko"):
        assert all(case.text[lang][k] for k in ("title", "story", "look"))


@vision
def test_comparing_images_leaves_the_folders_and_the_working_directory_clean(tmp_path, monkeypatch):
    ref, cur = factories.make_pair_dashcam(tmp_path / "d", scenario="blurry-lens")
    before = {p: sorted(x.name for x in p.iterdir()) for p in (ref, cur)}
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    _diff(ref, cur)
    assert {p: sorted(x.name for x in p.iterdir()) for p in (ref, cur)} == before
    assert list(work.iterdir()) == []

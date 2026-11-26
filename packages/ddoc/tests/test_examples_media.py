"""The generated text and audio examples: support tickets and machine
sounds. What they are is checked everywhere; how ddoc reads them, where
the text or audio plugin is installed."""
from __future__ import annotations

import csv
import json
import wave
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from ddoc.cli.main import app
from ddoc.examples import factories, machine, support

from test_dashcam import plugin_ready

runner = CliRunner()


def _texts(folder: Path):
    with (folder / "tickets" / "data.csv").open(encoding="utf-8") as fh:
        return [r["text"] for r in csv.DictReader(fh)]


def _clips(folder: Path):
    out = []
    for path in sorted((folder / "machine").glob("*.wav")):
        with wave.open(str(path)) as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, machine.RATE)
            out.append(np.frombuffer(w.readframes(w.getnframes()), dtype="<i2") / 32767)
    return out


# ── What the examples are ──────────────────────────────────────────────


def test_tickets_change_one_thing_per_story(tmp_path):
    words = lambda xs: np.mean([len(x.split()) for x in xs])  # noqa: E731
    ref, update = factories.make_pair_support(tmp_path / "u", scenario="app-update")
    _, chat = factories.make_pair_support(tmp_path / "c", scenario="chat-widget")
    _, week = factories.make_pair_support(tmp_path / "w", scenario="two-weeks")
    base, upd, cht, wk = _texts(ref), _texts(update), _texts(chat), _texts(week)
    assert len(base) == len(upd) == support.TICKETS
    assert (ref / "tickets" / "ddoc.yaml").read_text().startswith("modality: text")
    assert words(upd) > 1.3 * words(base) and words(cht) < 0.8 * words(base)
    assert abs(words(wk) - words(base)) < 0.1 * words(base)
    assert sum("v5" in t or "version 5" in t for t in upd) > 0.3 * len(upd)
    assert not any("v5" in t or "version 5" in t for t in base + wk)
    assert sum(t[:1].islower() for t in cht) > 0.8 * len(cht)


def test_machine_sounds_change_one_thing_per_story(tmp_path):
    def zcr(y): return float(np.mean(np.abs(np.diff(np.sign(y))) > 0))
    def rms(y): return float(np.sqrt(np.mean(y ** 2)))
    ref, worn = factories.make_pair_machine(tmp_path / "w", scenario="bearing-wear")
    _, louder = factories.make_pair_machine(tmp_path / "l", scenario="louder-mic")
    healthy, worn, louder = _clips(ref), _clips(worn), _clips(louder)
    assert len(healthy) == len(worn) == machine.CLIPS
    assert np.mean([zcr(y) for y in worn]) > 1.3 * np.mean([zcr(y) for y in healthy])
    ratio = np.mean([rms(y) for y in louder]) / np.mean([rms(y) for y in healthy])
    assert 2.0 < ratio < 2.4
    assert (ref / "machine" / "ddoc.yaml").read_text().startswith("modality: audio")


# ── How ddoc reads them ────────────────────────────────────────────────


def _diff(tmp_path, monkeypatch, make, story, *extra):
    ref, cur = make(tmp_path / story, scenario=story)
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(app, ["diff", str(ref), str(cur), "--json", *extra])
    return json.loads(res.output.strip().splitlines()[-1])


def _read(out, code):
    return {a for c in out["diagnosis"]["archetypes"] if c["code"] == code and c["match"] == "match"
            for a in c.get("attrs") or [None]}


@pytest.mark.skipif(not plugin_ready("text"), reason="needs the text plugin with torch")
def test_tickets_as_read_by_the_text_plugin(tmp_path, monkeypatch):
    update = _diff(tmp_path, monkeypatch, factories.make_pair_support, "app-update")
    assert update["verdict"] == "DRIFT" and update["top_attributes"][0][0] in ("length_chars", "length_words")
    chat = _diff(tmp_path, monkeypatch, factories.make_pair_support, "chat-widget", "--with-embeddings")
    assert chat["verdict"] == "DRIFT" and _read(chat, "embedding_shift")
    assert "length_words" in _read(chat, "shift_and_narrow")
    weeks = _diff(tmp_path, monkeypatch, factories.make_pair_support, "two-weeks")
    assert weeks["verdict"] == "OK" and _read(weeks, "stable")


@pytest.mark.skipif(not plugin_ready("audio"), reason="needs the audio plugin (librosa)")
def test_machine_sounds_as_read_by_the_audio_plugin(tmp_path, monkeypatch):
    worn = _diff(tmp_path, monkeypatch, factories.make_pair_machine, "bearing-wear")
    assert "spectral_centroid_mean" in _read(worn, "shift_and_narrow") | _read(worn, "level_shift")
    louder = _diff(tmp_path, monkeypatch, factories.make_pair_machine, "louder-mic")
    assert _read(louder, "unit_change") == {"rms_energy_mean"}
    shifts = _diff(tmp_path, monkeypatch, factories.make_pair_machine, "two-shifts")
    attr_codes = {c["code"] for c in shifts["diagnosis"]["archetypes"] if c["match"] == "match" and "attrs" in c}
    assert not attr_codes                      # nothing changed shape, whatever the score says

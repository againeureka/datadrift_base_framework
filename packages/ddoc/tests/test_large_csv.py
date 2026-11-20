"""Large CSVs are compared through a uniform random sample, and say so."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ddoc.cli.commands import diff as df_mod
from ddoc.cli.commands.diff import _load_csv_sampled, csv_diff
from ddoc.cli.commands.diff_report import build_consultation_html
from ddoc import profile as pf


def _write(path, n, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pd.DataFrame({"x": np.round(rng.normal(50 + shift, 5, n), 3),
                  "kind": rng.choice(["a", "b", "c"], n)}).to_csv(path, index=False)
    return str(path)


def test_small_files_are_read_whole(tmp_path):
    frame, total = _load_csv_sampled([_write(tmp_path / "s.csv", 300)], max_rows=1000)
    assert len(frame) == total == 300


def test_large_files_are_sampled_in_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(df_mod, "_CHUNK", 7_000)          # force several chunks
    path = _write(tmp_path / "big.csv", 30_000)
    frame, total = _load_csv_sampled([path], max_rows=5_000)
    assert total == 30_000 and len(frame) == 5_000
    assert frame["x"].dtype.kind == "f"                    # still numbers after the chunks
    again, _ = _load_csv_sampled([path], max_rows=5_000)
    assert frame.equals(again)                             # same data, same sample
    full = pd.read_csv(path)
    assert abs(frame["x"].mean() - full["x"].mean()) < 0.5  # a fair sample


def test_sampled_comparison_reports_it(tmp_path, monkeypatch):
    monkeypatch.setattr(df_mod, "_CHUNK", 7_000)
    ref, cur = _write(tmp_path / "r.csv", 30_000, seed=1), _write(tmp_path / "c.csv", 30_000, 5.0, seed=2)
    sampled = csv_diff([ref], [cur], max_rows=4_000)
    whole = csv_diff([ref], [cur], max_rows=0)
    assert sampled["profile"]["cur"]["total_rows"] == 30_000 and sampled["rows"]["cur"] == 4_000
    assert "total_rows" not in whole["profile"]["cur"]
    assert abs(sampled["attribute_drifts"]["x"] - whole["attribute_drifts"]["x"]) < 0.1
    payload = {"verdict": "DRIFT", "severity": "high", "overall_score": 0.4, "threshold": 0.15,
               "attr_threshold": 0.25, "top_attributes": [["x", 0.4]], "ref": "r", "cur": "c",
               "modality": "tabular", "engine": "builtin", "detector": "csv", "raw": sampled, "hints": []}
    assert "30,000행 중 4,000행 무작위 추출" in build_consultation_html(payload, lang="ko")


def test_profile_samples_too(tmp_path, monkeypatch):
    monkeypatch.setattr(df_mod, "_CHUNK", 7_000)
    env = pf.profile_csv([_write(tmp_path / "big.csv", 20_000)], max_rows=3_000)
    assert env["profile"]["rows"] == 3_000 and env["profile"]["total_rows"] == 20_000

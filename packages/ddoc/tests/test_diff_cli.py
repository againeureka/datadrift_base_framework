"""``ddoc diff`` — verdict facade tests.

Covers the three contract pillars: modality sniffing, the built-in CSV
fallback, and the verdict/exit-code mapping (0=OK, 1=DRIFT, 2=BLIND).
"""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from ddoc.cli.commands.diff import (
    compute_verdict,
    csv_diff,
    sniff_modality,
)
from ddoc.cli.main import app

runner = CliRunner()


# ── sniffing ──────────────────────────────────────────────────────────


def test_sniff_csv_file(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("x\n1\n")
    kind, detail = sniff_modality(f)
    assert kind == "csv"
    assert detail["files"] == [str(f)]


def test_sniff_image_dir(tmp_path):
    (tmp_path / "one.jpg").write_bytes(b"\xff\xd8")
    (tmp_path / "two.png").write_bytes(b"\x89PNG")
    kind, _ = sniff_modality(tmp_path)
    assert kind == "vision"


def test_sniff_categorical_marker_wins(tmp_path):
    (tmp_path / "distributions.json").write_text("{}")
    (tmp_path / "extra.csv").write_text("x\n1\n")
    kind, detail = sniff_modality(tmp_path)
    assert kind == "categorical"
    assert detail["marker"] == "distributions.json"


def test_sniff_empty_dir_unknown(tmp_path):
    kind, _ = sniff_modality(tmp_path)
    assert kind == "unknown"


# ── builtin csv comparator ────────────────────────────────────────────


def _write_csv(path, rows):
    path.write_text("\n".join(rows) + "\n")
    return str(path)


# Prices around 105 against prices around 300: 40 rows a side, enough that
# the difference cannot be chance (three rows a side could be).
LOW = ["price"] + [str(100 + i % 11) for i in range(40)]
HIGH = ["price"] + [str(290 + i % 21) for i in range(40)]


def test_csv_diff_identical_is_zero(tmp_path):
    rows = ["price,color", "100,red", "120,blue", "110,red"]
    a = _write_csv(tmp_path / "a.csv", rows)
    b = _write_csv(tmp_path / "b.csv", rows)
    res = csv_diff([a], [b])
    assert res["overall_score"] == 0.0
    assert set(res["columns_compared"]) == {"price", "color"}


def test_csv_diff_shifted_scores_high(tmp_path):
    a = _write_csv(tmp_path / "a.csv", LOW)
    b = _write_csv(tmp_path / "b.csv", HIGH)
    res = csv_diff([a], [b])
    assert res["overall_score"] == 1.0                   # no overlap: the JS distance is 1
    assert res["summaries"]["price"]["noise_p"] < 1e-6
    assert res["detector"] == "builtin:csv_js"


def test_three_rows_a_side_cannot_tell_change_from_chance(tmp_path):
    """Fully separated, but three against three splits that way by chance
    one time in twenty: the score is gated, the verdict OK, and the
    reading says the pattern is there but within noise."""
    ref, cur = _mk_pair(tmp_path, ["price", "100", "110", "105"], ["price", "300", "290", "310"])
    payload = json.loads(runner.invoke(app, ["diff", str(ref), str(cur), "--json"]).stdout.strip())
    assert payload["verdict"] == "OK" and payload["overall_score"] == 0.0
    codes = {c["code"] for c in payload["diagnosis"]["archetypes"] if c["match"] == "match"}
    assert {"hidden_change", "small_sample"} <= codes


def test_csv_diff_no_shared_columns_raises(tmp_path):
    a = _write_csv(tmp_path / "a.csv", ["x", "1"])
    b = _write_csv(tmp_path / "b.csv", ["y", "1"])
    with pytest.raises(ValueError, match="no shared columns"):
        csv_diff([a], [b])


# ── verdict mapping ───────────────────────────────────────────────────


def test_verdict_ok():
    v, sev, _ = compute_verdict(
        {"overall_score": 0.01, "attribute_drifts": {"a": 0.02}},
        threshold=0.15, attr_threshold=0.25,
    )
    assert (v, sev) == ("OK", "none")


def test_verdict_drift_by_overall():
    v, sev, _ = compute_verdict(
        {"overall_score": 0.2, "attribute_drifts": {"a": 0.2}},
        threshold=0.15, attr_threshold=0.25,
    )
    assert (v, sev) == ("DRIFT", "medium")


def test_verdict_drift_by_single_attribute():
    # 평균은 조용한데 한 축만 강하게 드리프트 — 평균에 묻히면 안 된다.
    v, sev, top = compute_verdict(
        {"overall_score": 0.08, "attribute_drifts": {"a": 0.4, "b": 0.01}},
        threshold=0.15, attr_threshold=0.25,
    )
    assert (v, sev) == ("DRIFT", "high")
    assert top[0][0] == "a"


# ── e2e: exit codes + json purity ─────────────────────────────────────


def _mk_pair(tmp_path, ref_rows, cur_rows):
    ref, cur = tmp_path / "ref", tmp_path / "cur"
    ref.mkdir(); cur.mkdir()
    _write_csv(ref / "data.csv", ref_rows)
    _write_csv(cur / "data.csv", cur_rows)
    return ref, cur


def test_diff_e2e_ok_exit_0(tmp_path):
    rows = ["price", "100", "110", "105"]
    ref, cur = _mk_pair(tmp_path, rows, rows)
    result = runner.invoke(app, ["diff", str(ref), str(cur), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["verdict"] == "OK"
    assert payload["engine"] == "builtin"


def test_diff_e2e_drift_exit_1_and_pure_json(tmp_path):
    ref, cur = _mk_pair(tmp_path, LOW, HIGH)
    result = runner.invoke(app, ["diff", str(ref), str(cur), "--json"])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout.strip())  # 전체 stdout이 JSON 하나여야 한다
    assert payload["verdict"] == "DRIFT"
    assert payload["top_attributes"][0][0] == "price"


def test_diff_e2e_missing_path_blind_exit_2(tmp_path):
    result = runner.invoke(app, ["diff", str(tmp_path / "nope"), str(tmp_path), "--json"])
    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout.strip())
    assert payload["verdict"] == "BLIND"


def test_diff_e2e_unknown_kind_blind_exit_2(tmp_path):
    ref, cur = tmp_path / "ref", tmp_path / "cur"
    ref.mkdir(); cur.mkdir()
    (ref / "weird.xyz").write_text("?")
    (cur / "weird.xyz").write_text("?")
    result = runner.invoke(app, ["diff", str(ref), str(cur), "--json"])
    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout.strip())["verdict"] == "BLIND"


# ── consultation report (--report) ────────────────────────────────────


def test_diff_report_csv_evidence(tmp_path):
    ref, cur = _mk_pair(
        tmp_path,
        ["price,color"] + [f"{p},{'red' if i % 3 else 'blue'}" for i, p in enumerate(LOW[1:])],
        ["price,color"] + [f"{p},green" for p in HIGH[1:]],
    )
    note = tmp_path / "note.html"
    result = runner.invoke(app, ["diff", str(ref), str(cur), "--json",
                                 "--report", str(note)])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout.strip())
    assert payload["report_path"] == str(note)
    body = note.read_text(encoding="utf-8")
    assert "Data Drift Diagnostic Report" in body
    assert "DRIFT" in body
    assert "price" in body and "color" in body          # findings bars
    assert "Findings by attribute" in body and "green" in body  # per-attribute panels
    assert "Items to check" in body                       # checks for the drifted columns



def test_a_report_that_fails_is_reported_not_swallowed(tmp_path, monkeypatch):
    from ddoc.cli.commands import diff_report

    def broken(*a, **k):
        raise KeyError("threshold")
    monkeypatch.setattr(diff_report, "build_consultation_html", broken)
    ref, cur = _mk_pair(tmp_path, LOW, HIGH)
    note = tmp_path / "note.html"
    out = runner.invoke(app, ["diff", str(ref), str(cur), "--report", str(note)])
    assert out.exit_code == 1                      # the verdict still stands
    assert not note.exists()
    assert "could not be written" in out.output and "KeyError" in out.output
    res = runner.invoke(app, ["diff", str(ref), str(cur), "--json", "--report", str(note)])
    payload = json.loads(res.stdout.strip())
    assert payload["report_error"].startswith("KeyError") and "report_path" not in payload

def test_diff_report_categorical_evidence(tmp_path):
    import sys as _sys
    _sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from fixtures.factories import make_pair_categorical
    ref, cur = make_pair_categorical(tmp_path, scenario="shifted")
    note = tmp_path / "note.html"
    result = runner.invoke(app, ["diff", str(ref), str(cur), "--json",
                                 "--report", str(note)])
    assert result.exit_code == 1, result.output
    body = note.read_text(encoding="utf-8")
    # Evidence pulled straight from the two distributions.json files.
    assert "color_distribution" in body
    assert "Findings by attribute" in body


def test_diff_report_blind(tmp_path):
    note = tmp_path / "note.html"
    result = runner.invoke(app, ["diff", str(tmp_path / "nope"), str(tmp_path),
                                 "--json", "--report", str(note)])
    assert result.exit_code == 2
    body = note.read_text(encoding="utf-8")
    assert "BLIND" in body and "Reason" in body


def test_build_consultation_html_minimal():
    from ddoc.cli.commands.diff_report import build_consultation_html
    html_out = build_consultation_html({
        "verdict": "OK", "severity": "none", "overall_score": 0.01,
        "threshold": 0.15, "attr_threshold": 0.25,
        "top_attributes": [["a", 0.01]], "ref": "/r", "cur": "/c",
        "modality": "tabular", "engine": "builtin", "detector": "x",
        "raw": {"attribute_drifts": {"a": 0.01}}, "hints": [],
    })
    assert html_out.startswith("<!doctype html><html") and "</html>" in html_out
    assert "OK" in html_out


# ── plugin isolation ──────────────────────────────────────────────────


def _pm_with(*plugins):
    import pluggy

    from ddoc.plugins import hookspecs

    pm = pluggy.PluginManager("ddoc")
    pm.add_hookspecs(hookspecs)
    for plugin in plugins:
        pm.register(plugin)
    return pm


_DRIFT_KWARGS = dict(
    snapshot_id_ref="__path__", snapshot_id_cur="__path__",
    data_path_ref="model.pth", data_path_cur="cur",
    data_hash_ref="", data_hash_cur="",
    detector="default", cfg={}, output_path="analysis/diff",
)


def test_failing_plugin_does_not_silence_the_others():
    """One plugin raising on an input it does not understand must not
    hide the answer of another plugin that handled it."""
    import pluggy

    from ddoc.cli.commands.utils import call_hook_isolated

    hookimpl = pluggy.HookimplMarker("ddoc")

    class Broken:
        @hookimpl
        def drift_detect(self, data_path_ref, **_):
            raise NotADirectoryError(data_path_ref)

    class Declines:
        @hookimpl
        def drift_detect(self, **_):
            return None

    class Works:
        @hookimpl
        def drift_detect(self, **_):
            return {"modality": "csv", "overall_score": 0.4}

    pm = _pm_with(Broken(), Declines(), Works())
    results, errors = call_hook_isolated(pm, "drift_detect", **_DRIFT_KWARGS)
    assert results == [{"modality": "csv", "overall_score": 0.4}]
    assert [name for name, _ in errors] == [pm.get_name(p) for p in pm.get_plugins()
                                           if type(p).__name__ == "Broken"]


def test_isolated_call_matches_pluggy_convention():
    """Plugins may declare only the hook arguments they use, and results
    come back in the same order as a normal pluggy fan-out."""
    import pluggy

    from ddoc.cli.commands.utils import call_hook_isolated

    hookimpl = pluggy.HookimplMarker("ddoc")

    class Narrow:
        @hookimpl
        def drift_detect(self, data_path_ref, detector):
            return {"modality": "narrow", "seen": [data_path_ref, detector]}

    class Full:
        @hookimpl
        def drift_detect(self, snapshot_id_ref, snapshot_id_cur, data_path_ref,
                         data_path_cur, data_hash_ref, data_hash_cur, detector,
                         cfg, output_path):
            return {"modality": "full"}

    pm = _pm_with(Narrow(), Full())
    results, errors = call_hook_isolated(pm, "drift_detect", **_DRIFT_KWARGS)
    assert errors == []
    assert results == pm.hook.drift_detect(**_DRIFT_KWARGS)
    assert {"modality": "narrow", "seen": ["model.pth", "default"]} in results


def test_pdf_report_keeps_json_stdout_clean(tmp_path):
    """Writing a PDF must not leak the PDF engine's logging: stdout stays
    exactly one JSON object, and the terminal is not flooded either."""
    pytest.importorskip("weasyprint")
    import subprocess
    import sys

    ref, cur = tmp_path / "ref.csv", tmp_path / "cur.csv"
    ref.write_text("x,kind\n" + "\n".join(f"{i % 10},{'ab'[i % 2]}" for i in range(200)) + "\n")
    cur.write_text("x,kind\n" + "\n".join(f"{i % 10 + 5},{'ab'[i % 3 > 0]}" for i in range(200)) + "\n")
    proc = subprocess.run([sys.executable, "-m", "ddoc.cli.main", "diff", str(ref), str(cur),
                           "--json", "--report", str(tmp_path / "note.pdf")],
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode in (0, 1), proc.stderr
    assert json.loads(proc.stdout)["report_path"].endswith("note.pdf")
    assert (tmp_path / "note.pdf").stat().st_size > 10_000
    assert len(proc.stderr.splitlines()) < 5, proc.stderr[:500]


def test_diff_leaves_nothing_in_the_working_directory(tmp_path, monkeypatch):
    """Plugins write a metrics copy to their output folder; `ddoc diff`
    must not leave it behind in the caller's directory."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "distributions.json").write_text(json.dumps({"label": {"a": 50, "b": 50}}))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    runner.invoke(app, ["diff", str(data), str(data), "--json"])
    assert list(work.iterdir()) == []


def test_looking_for_a_cache_leaves_no_trace(tmp_path, monkeypatch):
    """The core cache is made on the first write, and only for real data
    hashes; comparing plain paths has none."""
    from ddoc.core.cache_service import CacheService

    monkeypatch.chdir(tmp_path)
    cache = CacheService()
    assert cache.load_analysis_cache(snapshot_id="__path__", data_hash="", cache_type="attributes_text") is None
    assert cache.list_caches() == {"success": True, "caches": [], "count": 0}
    assert cache.save_analysis_cache("__path__", "", "attributes_text", {"a": 1})["skipped"]
    assert list(tmp_path.iterdir()) == []
    assert cache.save_analysis_cache("s1", "abc123", "attributes_text", {"a": 1})["success"]
    assert cache.load_analysis_cache(data_hash="abc123", cache_type="attributes_text") == {"a": 1}


def test_profile_of_a_csv_leaves_nothing_behind(tmp_path, monkeypatch):
    """Plugins that find nothing of their kind in a CSV may still write an
    analysis folder; the built-in profile takes over and it goes."""
    from ddoc.examples import factories

    _, cur = factories.make_pair_weather(tmp_path / "data", scenario="monsoon")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    res = runner.invoke(app, ["analyze", "eda", "--data-path", str(cur), "--quiet"])
    assert res.exit_code == 0, res.output
    assert list(work.iterdir()) == []


def test_values_from_a_plugin_are_read_like_csv_columns():
    """A plugin that hands over what it measured per item gets summaries
    and evidence built exactly as for a CSV column."""
    from ddoc.cli.commands.diff import attribute_evidence

    ref = [10.0 + (i % 7) for i in range(60)]
    summaries, dists, scores = attribute_evidence({
        "sharpness": {"ref": ref, "cur": [3.0 + (i % 3) * 0.5 for i in range(60)]},
        "level": {"ref": [1, 2, 3] * 20, "cur": [2, 3, 4] * 20},     # few integers: a bar each
        "empty": {"ref": [], "cur": [1.0]},
        "same": {"ref": [5.0] * 10, "cur": [5.0] * 10},
    })
    assert set(summaries) == {"sharpness", "level", "same"}
    s = summaries["sharpness"]
    assert s["kind"] == "numeric" and s["n"] == {"ref": 60, "cur": 60} and s["mean"]["cur"] < s["mean"]["ref"]
    assert sum(dists["sharpness"]["ref"].values()) == 60
    assert summaries["level"]["discrete"] and dists["level"]["cur"] == {"1": 0, "2": 20, "3": 20, "4": 20}
    assert "same" not in dists                                      # nothing to draw
    # scored like CSV columns: a clear move scores, no difference scores 0
    assert scores["sharpness"] == pytest.approx(1.0) and scores["same"] == 0.0


def _ten_minute_log(path, days, start, level_of_day):
    """``days`` days of ten-minute readings: each day its own level (and a
    daily cycle), so rows of one day resemble each other."""
    import math
    from datetime import datetime, timedelta

    rows = ["time,reading,kind"]
    t0 = datetime.fromisoformat(start)
    for d in range(days):
        for step in range(144):
            t = t0 + timedelta(days=d, minutes=10 * step)
            value = level_of_day(d) + 3 * math.sin(2 * math.pi * step / 144)
            kind = "wet" if (d + step // 72) % 3 == 0 else "dry"
            rows.append(f"{t:%Y-%m-%d %H:%M},{value:.2f},{kind}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(path)


def test_rows_in_time_order_are_tested_by_the_day(tmp_path):
    """Two fortnights of the same weather: each day has its own level, so
    rows one by one would call them different; by the day they are not."""
    import random

    rng = random.Random(4)
    levels = [rng.gauss(0, 2) for _ in range(28)]
    ref = _ten_minute_log(tmp_path / "ref.csv", 14, "2026-06-01", lambda d: levels[d])
    cur = _ten_minute_log(tmp_path / "cur.csv", 14, "2026-06-15", lambda d: levels[14 + d])
    payload = csv_diff([ref], [cur])
    assert payload["profile"]["noise_blocks"] == {
        "column": "time", "ref": {"unit": "day", "blocks": 14}, "cur": {"unit": "day", "blocks": 14}}
    reading = payload["summaries"]["reading"]
    assert payload["attribute_drifts"]["reading"] == 0.0 and reading["design_effect"] > 5
    assert payload["summaries"]["kind"]["kind"] == "categorical" and "design_effect" in payload["summaries"]["kind"]
    # the same rows without their times: every row counted on its own
    for name in ("ref.csv", "cur.csv"):
        lines = (tmp_path / name).read_text(encoding="utf-8").splitlines()
        (tmp_path / f"plain-{name}").write_text(
            "\n".join(",".join(line.split(",")[1:]) for line in lines) + "\n", encoding="utf-8")
    plain = csv_diff([str(tmp_path / "plain-ref.csv")], [str(tmp_path / "plain-cur.csv")])
    assert "noise_blocks" not in plain["profile"] and "design_effect" not in plain["summaries"]["reading"]
    assert plain["attribute_drifts"]["reading"] > 0.15     # beyond noise, row by row
    # a change bigger than the days' own swings still shows
    moved = _ten_minute_log(tmp_path / "moved.csv", 14, "2026-06-15", lambda d: levels[14 + d] + 8)
    assert csv_diff([ref], [moved])["attribute_drifts"]["reading"] > 0.5


def test_plugin_values_with_times_are_tested_by_the_day():
    from ddoc.cli.commands.diff import attribute_evidence

    times = [f"2026-06-{1 + i // 24:02d}T{i % 24:02d}:00" for i in range(24 * 6)]
    day = [i // 24 for i in range(24 * 6)]
    values = {"x": {"ref": [float(d % 3) + 0.01 * i for i, d in enumerate(day)],
                    "cur": [float((d + 1) % 3) + 0.01 * i for i, d in enumerate(day)],
                    "ref_time": times, "cur_time": times}}
    summaries, _, _ = attribute_evidence(values)
    assert summaries["x"]["noise_blocks"]["ref"] == {"unit": "day", "blocks": 6}
    assert summaries["x"]["design_effect"] > 1
    # times that do not line up with the values are ignored
    values["x"]["cur_time"] = times[:-1]
    summaries, _, _ = attribute_evidence(values)
    assert "noise_blocks" not in summaries["x"]

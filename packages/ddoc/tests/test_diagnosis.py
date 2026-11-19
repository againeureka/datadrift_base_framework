"""ddoc.diagnosis — rationale, per-attribute patterns, samples, checks."""
from __future__ import annotations

import json

from typer.testing import CliRunner

from ddoc import diagnosis as dx
from ddoc.cli.commands.diff import csv_diff
from ddoc.cli.commands.diff_report import build_consultation_html
from ddoc.cli.main import app

runner = CliRunner()


def _codes(patterns):
    return [p["code"] for p in patterns]


# ── rationale ─────────────────────────────────────────────────────────


def test_rationale_names_the_rule_that_fired():
    d = dx.build(verdict="DRIFT", overall=0.11, top=[("color", 0.27), ("hour", 0.15)],
                 threshold=0.15, attr_threshold=0.25)
    assert d["rationale"] == [{"code": "attr", "attr": "color", "score": 0.27, "threshold": 0.25}]

    d = dx.build(verdict="DRIFT", overall=0.2, top=[("a", 0.2)], threshold=0.15, attr_threshold=0.25)
    assert [r["code"] for r in d["rationale"]] == ["overall"]


def test_rationale_for_ok_reports_highest_attribute():
    d = dx.build(verdict="OK", overall=0.02, top=[("a", 0.05), ("b", 0.01)],
                 threshold=0.15, attr_threshold=0.25)
    assert d["rationale"][0]["code"] == "ok"
    assert d["rationale"][0]["max_attr"] == 0.05
    assert d["checks"] == []


# ── categorical patterns ──────────────────────────────────────────────


def test_new_and_vanished_values_and_share_shift():
    evidence = {
        "color": {"ref": {"red": 60, "blue": 40, "green": 0},
                  "cur": {"red": 10, "blue": 40, "green": 0, "pink": 50}},
        "shape": {"ref": {"a": 50, "b": 50}, "cur": {"a": 100, "b": 0}},
    }
    d = dx.build(verdict="DRIFT", overall=0.4, top=[("color", 0.4), ("shape", 0.3)],
                 threshold=0.15, attr_threshold=0.25, evidence=evidence)
    by_name = {a["name"]: a for a in d["attributes"]}
    # pink appeared with 50 %; red's 60 % → 10 % drop is the same event
    # seen from the other side, so it is not reported a second time.
    color = by_name["color"]["patterns"]
    assert _codes(color) == ["new_values"]
    assert color[0]["values"] == [["pink", 0.5]]
    assert _codes(by_name["shape"]["patterns"]) == ["vanished_values"]


def test_share_shift_not_repeated_when_the_top_mover_is_new():
    evidence = {"hour": {"ref": {"8": 50, "9": 50, "21": 0},
                         "cur": {"8": 45, "9": 40, "21": 15}}}
    d = dx.build(verdict="DRIFT", overall=0.3, top=[("hour", 0.3)],
                 threshold=0.15, attr_threshold=0.25, evidence=evidence)
    assert _codes(d["attributes"][0]["patterns"]) == ["new_values"]


def test_share_shift_when_no_value_appears_or_vanishes():
    evidence = {"channel": {"ref": {"store": 75, "online": 25}, "cur": {"store": 20, "online": 80}}}
    d = dx.build(verdict="DRIFT", overall=0.3, top=[("channel", 0.3)],
                 threshold=0.15, attr_threshold=0.25, evidence=evidence)
    patterns = d["attributes"][0]["patterns"]
    assert _codes(patterns) == ["share_shift"]
    assert patterns[0]["label"] in ("store", "online")


def test_stat_dicts_are_not_treated_as_distributions():
    evidence = {"confidence": {"ref": {"mean": 0.85, "std": 0.07}, "cur": {"mean": 0.78, "std": 0.1}}}
    d = dx.build(verdict="OK", overall=0.0, top=[("confidence", 0.003)],
                 threshold=0.15, attr_threshold=0.25, evidence=evidence)
    summary = d["attributes"][0]["summary"]
    assert summary["kind"] == "stats"
    assert summary["values"]["mean"] == {"ref": 0.85, "cur": 0.78}
    assert "0.85 → 0.78" in dx.summary_text(summary, "en")


# ── numeric patterns (CSV comparator) ─────────────────────────────────


def _csv(path, header, rows):
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return str(path)


def test_csv_numeric_shift_produces_range_mean_and_spread_patterns(tmp_path):
    ref = _csv(tmp_path / "r.csv", "price", [str(100 + (i % 10)) for i in range(200)])
    cur = _csv(tmp_path / "c.csv", "price", [str(180 + (i % 40)) for i in range(200)])
    result = csv_diff([ref], [cur])
    s = result["summaries"]["price"]
    assert s["kind"] == "numeric" and s["outside_ref_range"] == 1.0
    d = dx.build(verdict="DRIFT", overall=1.0, top=[("price", 1.0)],
                 threshold=0.15, attr_threshold=0.25, result=result)
    assert _codes(d["attributes"][0]["patterns"]) == ["out_of_range", "mean_shift", "spread_change"]
    # one sentence per kind of check: range and mean collapse into a value-shift check
    texts = dx.check_texts(d, "en")
    assert sum("values of 'price' moved" in x for x in texts) == 1


def test_csv_missing_rate_change(tmp_path):
    # Two columns: a single-column CSV would skip blank lines entirely.
    ref = _csv(tmp_path / "r.csv", "x,id", ["1,a"] * 100)
    cur = _csv(tmp_path / "c.csv", "x,id", ["1,a"] * 70 + [",a"] * 30)
    result = csv_diff([ref], [cur])
    assert result["summaries"]["x"]["missing"] == {"ref": 0.0, "cur": 0.3}
    d = dx.build(verdict="DRIFT", overall=0.3, top=[("x", 0.3)],
                 threshold=0.15, attr_threshold=0.25, result=result)
    assert "missing_change" in _codes(d["attributes"][0]["patterns"])


# ── samples and checks ────────────────────────────────────────────────


def test_small_sample_warning_uses_the_smallest_attribute():
    evidence = {"a": {"ref": {"x": 100, "y": 100}, "cur": {"x": 90, "y": 110}},
                "b": {"ref": {"x": 10, "y": 5}, "cur": {"x": 4, "y": 12}}}
    d = dx.build(verdict="DRIFT", overall=0.3, top=[("b", 0.3), ("a", 0.01)],
                 threshold=0.15, attr_threshold=0.25, evidence=evidence)
    assert d["samples"] is None  # attributes disagree on N
    assert d["warnings"][0] == {"code": "small_sample", "ref": 15, "cur": 16}


def test_checks_only_for_attributes_that_drove_the_verdict():
    evidence = {"hot": {"ref": {"a": 80, "b": 20}, "cur": {"a": 20, "b": 80}},
                "calm": {"ref": {"a": 50, "b": 50}, "cur": {"a": 38, "b": 62}}}
    d = dx.build(verdict="DRIFT", overall=0.3, top=[("hot", 0.4), ("calm", 0.02)],
                 threshold=0.15, attr_threshold=0.25, evidence=evidence)
    assert {c["attr"] for c in d["checks"]} == {"hot"}
    calm = next(a for a in d["attributes"] if a["name"] == "calm")
    assert _codes(calm["patterns"]) == ["share_shift"]  # still described


def test_drift_without_patterns_falls_back_to_inspect():
    d = dx.build(verdict="DRIFT", overall=0.3, top=[("emb", 0.3)],
                 threshold=0.15, attr_threshold=0.25)
    assert d["checks"] == [{"code": "inspect", "attr": "emb"}]
    assert d["warnings"] == [{"code": "no_breakdown"}]


# ── end to end ────────────────────────────────────────────────────────


def test_diff_json_carries_diagnosis_and_specific_hints(tmp_path):
    ref = _csv(tmp_path / "r.csv", "channel", ["store"] * 75 + ["online"] * 25)
    cur = _csv(tmp_path / "c.csv", "channel", ["store"] * 20 + ["online"] * 80)
    res = runner.invoke(app, ["diff", ref, cur, "--json", "--lang", "en"])
    payload = json.loads(res.output.strip().splitlines()[-1])
    assert payload["verdict"] == "DRIFT"
    assert payload["diagnosis"]["checks"][0]["code"] == "share_shift"
    assert any("mix of 'channel' changed" in h for h in payload["hints"])


def test_note_sections_in_both_languages(tmp_path):
    ref = _csv(tmp_path / "r.csv", "price", [str(100 + (i % 10)) for i in range(100)])
    cur = _csv(tmp_path / "c.csv", "price", [str(180 + (i % 10)) for i in range(100)])
    for lang, words in (("en", ["Basis for the verdict", "Results by attribute", "Findings by attribute",
                                "Characteristics of the data", "Items to check",
                                "Limitations and remarks"]),
                        ("ko", ["판정 근거", "속성별 결과", "속성별 상세", "원본 데이터 특징",
                                "점검 항목", "한계 및 비고"])):
        note = tmp_path / f"note_{lang}.html"
        res = runner.invoke(app, ["diff", ref, cur, "--report", str(note), "--lang", lang])
        assert res.exit_code == 1
        body = note.read_text(encoding="utf-8")
        for w in words:
            assert w in body, (lang, w)


def test_note_renders_without_a_precomputed_diagnosis():
    html_out = build_consultation_html({
        "verdict": "DRIFT", "severity": "high", "overall_score": 0.3,
        "threshold": 0.15, "attr_threshold": 0.25, "top_attributes": [["a", 0.3]],
        "ref": "r", "cur": "c", "modality": "categorical", "engine": "plugin", "detector": "x",
        "raw": {"attribute_drifts": {"a": 0.3}}, "hints": [],
    }, {"a": {"ref": {"x": 80, "y": 20}, "cur": {"x": 20, "y": 80}}})
    assert "Items to check" in html_out


def test_report_header_issuer_and_document_number(tmp_path):
    ref = _csv(tmp_path / "r.csv", "c", ["a"] * 40 + ["b"] * 10)
    cur = _csv(tmp_path / "c.csv", "c", ["a"] * 10 + ["b"] * 40)
    note = tmp_path / "n.html"
    res = runner.invoke(app, ["diff", ref, cur, "--report", str(note), "--issuer", "Example Lab"])
    assert res.exit_code == 1
    body = note.read_text(encoding="utf-8")
    assert "Example Lab" in body
    import re as _re
    assert _re.search(r"DD-\d{8}-[0-9A-F]{6}", body)
    # charts come with a table view and hover tips; no unescaped data in tips
    assert "Show as table" in body and "data-tip=" in body


def test_report_escapes_attribute_names(tmp_path):
    html_out = build_consultation_html({
        "verdict": "DRIFT", "severity": "high", "overall_score": 0.3,
        "threshold": 0.15, "attr_threshold": 0.25, "top_attributes": [["<img src=x>", 0.3]],
        "ref": "r", "cur": "c", "modality": "categorical", "engine": "plugin", "detector": "x",
        "raw": {"attribute_drifts": {"<img src=x>": 0.3}}, "hints": [],
    }, {"<img src=x>": {"ref": {"<script>alert(1)</script>": 80, "y": 20},
                        "cur": {"<script>alert(1)</script>": 20, "y": 80}}})
    assert "<img src=x>" not in html_out
    assert "<script>alert(1)</script>" not in html_out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_out


def test_report_has_theme_switch_and_print_button_kept_out_of_print():
    payload = {
        "verdict": "OK", "severity": "none", "overall_score": 0.01,
        "threshold": 0.15, "attr_threshold": 0.25, "top_attributes": [["a", 0.01]],
        "ref": "r", "cur": "c", "modality": "categorical", "engine": "plugin", "detector": "x",
        "raw": {"attribute_drifts": {"a": 0.01}}, "hints": [],
    }
    ko = build_consultation_html(payload, lang="ko")
    for word in ("주간", "야간", "자동", "인쇄 / PDF"):
        assert word in ko
    assert "data-set-theme='dark'" in ko and "window.print()" in ko
    # night colours apply on screen only, so paper and PDF stay white
    assert "@media screen and (prefers-color-scheme: dark)" in ko
    assert ".bar { display: none !important; }" in ko

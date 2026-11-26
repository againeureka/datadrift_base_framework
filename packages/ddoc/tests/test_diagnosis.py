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
    assert any("spread of 'price' grew" in x for x in texts)


def test_spread_check_follows_the_direction():
    # a narrower spread is not mixed sources or noise: clipping, filling, a group left out
    narrower = {"code": "spread_change", "attr": "x", "ref": 2.0, "cur": 1.0}
    assert "narrowed" in dx.check_text(narrower, "en") and "mixed" not in dx.check_text(narrower, "en")
    assert "좁아졌습니다" in dx.check_text(narrower, "ko")
    assert "커졌습니다" in dx.check_text({**narrower, "ref": 1.0, "cur": 2.0}, "ko")


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


def test_few_distinct_integers_are_shown_per_value(tmp_path):
    ref = _csv(tmp_path / "r.csv", "rating", [str(1 + i % 5) for i in range(200)])
    cur = _csv(tmp_path / "c.csv", "rating", [str(3 + i % 3) for i in range(200)])
    result = csv_diff([ref], [cur])
    assert result["summaries"]["rating"]["discrete"] is True
    assert list(result["distributions"]["rating"]["ref"]) == ["1", "2", "3", "4", "5"]
    # same score as the shared-range histogram: each integer has its own bin
    import numpy as np
    from ddoc.scoring import score
    edges = np.linspace(1, 5, 21)
    rh, _ = np.histogram([1 + i % 5 for i in range(200)], bins=edges)
    ch, _ = np.histogram([3 + i % 3 for i in range(200)], bins=edges)
    assert result["attribute_drifts"]["rating"] == round(score(rh.tolist(), ch.tolist())[0], 4)
    payload = {"verdict": "DRIFT", "severity": "high", "overall_score": 0.4, "threshold": 0.15,
               "attr_threshold": 0.25, "top_attributes": [["rating", 0.4]], "ref": "r", "cur": "c",
               "modality": "tabular", "engine": "builtin", "detector": "csv", "raw": result, "hints": []}
    page = build_consultation_html(payload, result["distributions"])
    assert "data-tip='5\n" in page  # one hover target per value, labelled by the value


def test_wide_integer_ranges_keep_the_histogram(tmp_path):
    ref = _csv(tmp_path / "r.csv", "day", [str(i % 60) for i in range(240)])
    cur = _csv(tmp_path / "c.csv", "day", [str(i % 60) for i in range(240)])
    result = csv_diff([ref], [cur])
    assert "discrete" not in result["summaries"]["day"]


# ── what was compared / inspection table ──────────────────────────────


def _orders(path, month, extra=False, missing_every=0):
    head = "order_date,price,channel" + (",coupon" if extra else "")
    rows = []
    for i in range(120):
        ch = "" if missing_every and i % missing_every == 0 else ("store" if i % 3 else "online")
        rows.append(f"2026-{month:02d}-{1 + i % 28:02d},{100 + i % 7 + (40 if month > 6 else 0)},{ch}"
                    + (",Y" if extra else ""))
    return _csv(path, head, rows)


def test_profile_names_period_and_one_sided_columns(tmp_path):
    ref = _orders(tmp_path / "r.csv", 1)
    cur = _orders(tmp_path / "c.csv", 7, extra=True)
    result = csv_diff([ref], [cur])
    prof = result["profile"]
    assert prof["ref"]["period"] == {"column": "order_date", "start": "2026-01-01", "end": "2026-01-28"}
    assert prof["cur"]["period"]["start"] == "2026-07-01"
    assert prof["columns"] == {"compared": ["price", "channel"], "time": ["order_date"], "running": [],
                               "only_ref": [], "only_cur": ["coupon"]}
    # two periods always differ in their timestamps: not scored
    assert "order_date" not in result["attribute_drifts"]
    assert "sample" not in prof["ref"]  # raw rows only on request


def test_integer_timestamp_index_stays_numeric(tmp_path):
    ref = _csv(tmp_path / "r.csv", "timestamp,x", [f"{i},{i % 5}" for i in range(50)])
    cur = _csv(tmp_path / "c.csv", "timestamp,x", [f"{i},{i % 5}" for i in range(50)])
    result = csv_diff([ref], [cur])
    assert "timestamp" in result["attribute_drifts"] and result["profile"]["columns"]["time"] == []


def test_sample_rows_are_opt_in(tmp_path):
    ref = _orders(tmp_path / "r.csv", 1)
    cur = _orders(tmp_path / "c.csv", 7)
    off = tmp_path / "off.html"
    on = tmp_path / "on.html"
    runner.invoke(app, ["diff", ref, cur, "--report", str(off), "--lang", "ko"])
    res = runner.invoke(app, ["diff", ref, cur, "--report", str(on), "--lang", "ko",
                              "--sample-rows", "2", "--json"])
    payload = json.loads(res.output.strip().splitlines()[-1])
    assert len(payload["raw"]["profile"]["cur"]["sample"]["rows"]) == 2
    assert "원본 데이터입니다" not in off.read_text(encoding="utf-8")
    body = on.read_text(encoding="utf-8")
    assert "원본 데이터입니다" in body and "현재 (cur) 앞 2행" in body
    assert "검사 대상" in body and "2026-07-01 ~ 2026-07-28 (order_date 열)" in body


def test_inspection_rows_and_dropped_column_check(tmp_path):
    ref = _orders(tmp_path / "r.csv", 1, extra=True)
    cur = _orders(tmp_path / "c.csv", 7, missing_every=4)
    res = runner.invoke(app, ["diff", ref, cur, "--json"])
    d = json.loads(res.output.strip().splitlines()[-1])["diagnosis"]
    rows = {(r["system"], r["item"], r.get("attr"), r["status"]) for r in d["inspection"]}
    assert ("schema", "columns", None, "caution") in rows
    assert ("distribution", "attr_score", "price", "fail") in rows
    assert ("completeness", "missing_change", "channel", "caution") in rows
    assert ("numeric", "out_of_range", "price", "caution") in rows
    assert d["checks"][0] == {"code": "column_dropped", "columns": ["coupon"]}
    text = dx.inspection_text(next(r for r in d["inspection"] if r["item"] == "missing_change"), "ko")
    assert text["criterion"] == "5%p 미만" and text["result"] == "주의"


def test_inspection_without_profile_still_reports_scores():
    d = dx.build(verdict="OK", overall=0.02, top=[("a", 0.05), ("b", 0.01)],
                 threshold=0.15, attr_threshold=0.25,
                 evidence={"a": {"ref": {"x": 50, "y": 50}, "cur": {"x": 45, "y": 55}}})
    items = [(r["system"], r["item"], r["status"]) for r in d["inspection"]]
    assert ("distribution", "overall", "pass") in items
    assert not any(s == "schema" for s, _, _ in items)
    rest = next(r for r in d["inspection"] if r["item"] == "attr_score")
    assert dx.inspection_text(rest, "en")["target"] == "all (2)"



def test_report_with_more_attributes_over_the_threshold_than_rows(tmp_path):
    # Beyond eight attributes over the threshold the table folds the rest
    # into one "and N more" row; that row needs its criterion too, or the
    # whole report fails (seen on a real weather-station log).
    import numpy as np
    rng = np.random.default_rng(1)
    cols = [f"c{i}" for i in range(11)]
    for name, shift in (("r.csv", 0.0), ("c.csv", 3.0)):
        rows = [",".join(cols)] + [",".join(f"{x:.3f}" for x in rng.normal(shift, 1, len(cols)))
                                   for _ in range(300)]
        (tmp_path / name).write_text("\n".join(rows) + "\n")
    note = tmp_path / "n.html"
    res = runner.invoke(app, ["diff", str(tmp_path / "r.csv"), str(tmp_path / "c.csv"),
                              "--json", "--report", str(note)])
    payload = json.loads(res.output.strip().splitlines()[-1])
    assert "report_error" not in payload
    more = next(r for r in payload["diagnosis"]["inspection"] if r.get("more"))
    assert more["more"] == 3
    assert dx.inspection_text(more, "en")["criterion"] == dx.inspection_text(
        next(r for r in payload["diagnosis"]["inspection"]
             if r["item"] == "attr_score" and r.get("attr")), "en")["criterion"]
    assert note.exists() and "c10" in note.read_text(encoding="utf-8")

# ── data diagram ──────────────────────────────────────────────────────


def test_diagram_numbers_parts_that_need_attention_and_links_them(tmp_path):
    ref = _orders(tmp_path / "r.csv", 1)
    cur = _orders(tmp_path / "c.csv", 7, extra=True, missing_every=4)
    note = tmp_path / "n.html"
    runner.invoke(app, ["diff", ref, cur, "--report", str(note), "--lang", "ko"])
    body = note.read_text(encoding="utf-8")
    assert "데이터 도면" in body
    import re as _re
    from ddoc.cli.commands.diff_report import anchor_id
    # parts in the data's own order; the time column and the one-sided column are drawn as such
    order = _re.findall(r"class='part [^']*' (?:href='[^']*' )?data-col='([^']+)'", body)
    assert order == ["order_date", "price", "channel", "coupon"]
    assert "기간 열" in body and "현재에만 있음" in body
    # price and channel are numbered, linked, and their panels carry the anchor
    for name in ("price", "channel"):
        assert f"href='#{anchor_id(name)}'" in body and f"id='{anchor_id(name)}'" in body
    assert body.count("<li data-col=") == 2


def test_diagram_folds_passing_columns_when_there_are_many(tmp_path):
    cols = [f"c{i}" for i in range(60)]
    ref = _csv(tmp_path / "r.csv", ",".join(cols), [",".join(str((i + j) % 5) for j in range(60)) for i in range(60)])
    cur_rows = [",".join(str((i + j) % 5 if j else 9) for j in range(60)) for i in range(60)]
    cur = _csv(tmp_path / "c.csv", ",".join(cols), cur_rows)
    note = tmp_path / "n.html"
    runner.invoke(app, ["diff", ref, cur, "--report", str(note), "--lang", "en"])
    body = note.read_text(encoding="utf-8")
    assert "passing columns not drawn" in body
    assert "data-col='c0'" in body  # the drifted column is always drawn


# ── why a score is 0 ──────────────────────────────────────────────────


def _rows(path, values):
    path.write_text("x,flag\n" + "\n".join(f"{v},{f}" for v, f in values) + "\n", encoding="utf-8")
    return str(path)


def test_a_zero_says_whether_it_is_noise_or_nothing(tmp_path):
    import random

    rng = random.Random(2)
    # x: a small shift on 40 rows (within noise); flag: the same on both sides
    ref = _rows(tmp_path / "ref.csv", [(round(rng.gauss(0, 1), 3), "a") for _ in range(40)])
    cur = _rows(tmp_path / "cur.csv", [(round(rng.gauss(0.6, 1), 3), "a") for _ in range(40)])
    result = csv_diff([ref], [cur])
    d = dx.build(verdict="OK", overall=0.0, top=[], threshold=0.15, attr_threshold=0.25,
                 result=result, evidence=result["distributions"])
    x, flag = d["noise"]["x"], d["noise"]["flag"]
    assert x["state"] == "within_noise" and x["distance"] > 0.2 and x["p"] >= 0.001
    assert flag["state"] == "same" and flag["distance"] == 0
    # the zeros are ordered by how far they moved, and the reason is told
    assert [a["name"] for a in d["attributes"]] == ["x", "flag"]
    assert d["rationale"][1]["code"] == "noise" and d["rationale"][1]["attr"] == "x"
    assert "no more than sampling noise" in dx.rationale_text(d["rationale"][1], "en")
    assert dx.noise_text(x, "ko").startswith("잡음 범위 안 (거리 ")
    assert dx.noise_text(flag, "en") == "same distribution"
    assert dx.noise_text({**x, "state": "beyond"}, "en") is None


def test_a_zero_with_more_missing_values_is_not_called_the_same(tmp_path):
    ref = _rows(tmp_path / "ref.csv", [(i % 10, "a") for i in range(400)])
    cur = _rows(tmp_path / "cur.csv", [("" if i % 4 == 0 else i % 10, "a") for i in range(400)])
    result = csv_diff([ref], [cur])
    d = dx.build(verdict="DRIFT", overall=0.0, top=[], threshold=0.15, attr_threshold=0.25,
                 result=result, evidence=result["distributions"])
    note = d["noise"]["x"]
    assert result["attribute_drifts"]["x"] == 0 and note["state"] == "missing"
    assert dx.noise_text(note, "en") == "values alike; missing 0.0% → 25.0%"


def test_report_and_console_show_what_moved_within_noise(tmp_path):
    from ddoc.cli.commands.diff_report import build_consultation_markdown

    import random

    rng = random.Random(2)
    ref = _rows(tmp_path / "ref.csv", [(round(rng.gauss(0, 1), 3), "a") for _ in range(40)])
    cur = _rows(tmp_path / "cur.csv", [(round(rng.gauss(0.6, 1), 3), "a") for _ in range(40)])
    out = runner.invoke(app, ["diff", ref, cur, "--lang", "en"])
    line = next(x for x in out.stdout.splitlines() if "Within noise:" in x)
    assert out.exit_code == 0 and "x 0." in line and "flag" not in line     # the same is not noise
    # a distance below the threshold would not have mattered: not worth a console line
    quiet = runner.invoke(app, ["diff", ref, cur, "--lang", "en", "--threshold", "0.9"])
    assert "Within noise:" not in quiet.stdout
    payload = json.loads(runner.invoke(app, ["diff", ref, cur, "--json"]).stdout)
    page = build_consultation_html(payload, payload["raw"]["distributions"], lang="en")
    assert "within noise (distance " in page and "stroke-dasharray" in page
    assert "same distribution" in page                       # flag: nothing to tell apart
    md = build_consultation_markdown(payload, lang="en")
    assert "## Basis for the verdict" in md and "no more than sampling noise" in md


def test_blocks_are_described_in_words():
    blocks = {"ref": {"unit": "day", "blocks": 14}, "cur": {"unit": "stretch", "blocks": 6, "seconds": 7200}}
    assert dx.blocks_text(blocks, "en").endswith("14 days in the baseline, 6 stretches of 2 h in the current data.")
    assert "기준 14일, 현재 2 h씩 6구간" in dx.blocks_text(blocks, "ko")
    assert dx.blocks_text(None, "en") is None

"""ddoc.profile — the built-in CSV profile and its report."""
from __future__ import annotations

import json

from typer.testing import CliRunner

from ddoc import profile as pf
from ddoc.cli.commands.profile_report import build_profile_html, build_profile_markdown
from ddoc.cli.main import app

runner = CliRunner()


def _csv(path, header, rows):
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return str(path)


def _orders(tmp_path):
    rows = []
    for i in range(200):
        price = 900.0 if i % 40 == 0 else 100 + (i * 7) % 30
        channel = ["store", "Store", "online", "app"][i % 4]
        grade = "" if i % 4 == 0 else "AB"[i % 2]
        rows.append(f"{1000 + i},2026-07-{1 + i % 28:02d},{price},{channel},{1 + i % 4},{grade},x,"
                    f"{'1,234' if i % 9 else '1234'}")
    rows.append(rows[1])  # one duplicate
    return _csv(tmp_path / "orders.csv", "order_id,order_date,price,channel,qty,grade,flag,amount_text",
                [r.replace(",1,234", ',"1,234"') for r in rows])


def _codes(env):
    return {f["code"]: f.get("attrs") for f in env["findings"]}


def test_profile_reads_the_quality_patterns(tmp_path):
    env = pf.profile_csv([_orders(tmp_path)])
    assert env["profile"]["period"]["start"] == "2026-07-01"
    codes = _codes(env)
    assert codes["id_like"] == ["order_id"]
    assert codes["far_outliers"] == ["price"]
    assert codes["spelling_variants"] == ["channel"]
    assert codes["high_missing"] == ["grade"]
    assert codes["constant"] == ["flag"]
    assert codes["numbers_as_text"] == ["amount_text"]
    assert "duplicate_rows" not in codes  # 1 of 201 rows is under the 1 % mark
    assert env["columns"]["qty"]["discrete"] is True
    # far outliers do not stretch the bins: the bulk is binned, the rest counted apart
    assert env["columns"]["price"]["outside_bins"] > 0
    assert max(float(k.split("–")[1]) for k in env["columns"]["price"]["distribution"]) < 900


def test_small_and_duplicated_data(tmp_path):
    env = pf.profile_csv([_csv(tmp_path / "s.csv", "a,b", ["1,x"] * 10)])
    codes = _codes(env)
    assert "small_sample" in codes and "duplicate_rows" in codes


def test_findings_render_in_both_languages(tmp_path):
    env = pf.profile_csv([_orders(tmp_path)])
    for f in env["findings"]:
        for lang in ("en", "ko"):
            text = pf.finding_text(f, lang)
            assert text["title"] and text["signals"] and text["actions"]
    variants = next(f for f in env["findings"] if f["code"] == "spelling_variants")
    assert "Store ≈ store" in pf.finding_text(variants, "ko")["signals"][0]


def test_catalog_entries_agree_across_languages():
    import re
    assert set(pf.CATALOG) == set(pf.DATA_CODES) | set(pf.COLUMN_CODES)
    for code, entry in pf.CATALOG.items():
        en, ko = entry["en"], entry["ko"]
        assert set(en) == set(ko), code
        assert set(re.findall(r"{(\w+)}", en["signal"])) == set(re.findall(r"{(\w+)}", ko["signal"])), code


def test_analyze_eda_falls_back_to_the_builtin_profile(tmp_path):
    path = _orders(tmp_path)
    res = runner.invoke(app, ["analyze", "eda", "--data-path", path, "--json", "--quiet", "--sample-rows", "2"])
    env = json.loads(res.output.strip().splitlines()[-1])
    assert env["profiler"] == "builtin:csv_profile" and len(env["profile"]["sample"]["rows"]) == 2


def test_profile_report_html_markdown_and_render(tmp_path):
    env = pf.profile_csv([_orders(tmp_path)])
    page = build_profile_html(env, lang="ko", issuer="Lab")
    for word in ("데이터 프로필 보고서", "Lab", "데이터 도면", "품질 소견", "열별 특징", "열별 분포",
                 "식별자형 열", "그림에서 뺐습니다"):
        assert word in page, word
    md = build_profile_markdown(env, lang="en")
    assert md.startswith("# Data Profile Report") and "### Far outliers" in md and "- [ ] " in md
    # through `report render`, which must not hand the profile to the tabular plugin
    src = tmp_path / "eda.json"
    src.write_text(json.dumps(env))
    out = tmp_path / "p.html"
    res = runner.invoke(app, ["report", "render", "-i", str(src), "-o", str(out), "--json"])
    assert json.loads(res.output.strip().splitlines()[-1])["renderer"] == "builtin"
    assert "Data Profile Report" in out.read_text(encoding="utf-8")


def test_plugin_eda_envelope_gets_the_same_shell(tmp_path):
    env = {"status": "success", "modality": "timeseries", "series_analyzed": 3,
           "metrics_file": "analysis/metrics.json", "summary": {"num_series": 3, "timestamp": "20261120"}}
    page = build_profile_html(env, lang="en")
    assert "Data Profile Report" in page and "num_series" in page and "series_analyzed" in page
    assert "<script>" in page and "data-set-theme" in page  # toolbar and theme switch included


def test_profile_escapes_values(tmp_path):
    path = _csv(tmp_path / "x.csv", "name,v", [f"<script>alert({i})</script>,{i}" for i in range(60)])
    page = build_profile_html(pf.profile_csv([path]), lang="en")
    assert "<script>alert(" not in page and "&lt;script&gt;alert(" in page

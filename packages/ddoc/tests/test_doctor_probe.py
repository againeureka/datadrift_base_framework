"""ddoc doctor / ddoc probe — look before running."""
from __future__ import annotations

import json

from typer.testing import CliRunner

from ddoc.cli.commands import doctor as dr
from ddoc.cli.main import app

runner = CliRunner()


def _csv(path, header, rows):
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return str(path)


def test_doctor_reports_core_plugins_and_features():
    res = runner.invoke(app, ["doctor", "--json"])
    assert res.exit_code == 0, res.output
    body = json.loads(res.output.strip().splitlines()[-1])
    areas = {c["area"] for c in body["checks"]}
    assert body["ok"] is True and areas == {"core", "plugins", "features"}
    assert any(c["item"] == "pandas" and c["status"] == "ok" for c in body["checks"])


def test_doctor_flags_a_plugin_that_does_not_load(monkeypatch):
    monkeypatch.setattr(dr, "plugin_status",
                        lambda: [{"name": "ddoc_vision", "version": "1.0.0",
                                  "error": "ModuleNotFoundError: No module named 'clip'"}])
    rows = dr.environment("en")
    vision = next(r for r in rows if r["item"] == "ddoc_vision")
    assert vision["status"] == "fail" and "clip" in vision["detail"] and vision["fix"]
    res = runner.invoke(app, ["doctor", "--lang", "en"])
    assert res.exit_code == 0 and "1 plugin(s) do not load" in res.output  # the core still works


def test_probe_csv_pair_lists_what_would_be_compared(tmp_path):
    ref = _csv(tmp_path / "r.csv", "order_date,price,channel", [f"2026-01-{1 + i % 28:02d},{i},a" for i in range(40)])
    cur = _csv(tmp_path / "c.csv", "order_date,price,coupon", [f"2026-07-{1 + i % 28:02d},{i},Y" for i in range(40)])
    res = runner.invoke(app, ["probe", ref, cur, "--json"])
    assert res.exit_code == 0
    body = json.loads(res.output.strip().splitlines()[-1])
    assert body["csv"]["compared"] == ["price"] and body["csv"]["time"] == ["order_date"]
    assert body["csv"]["only_ref"] == ["channel"] and body["csv"]["only_cur"] == ["coupon"]


def test_probe_says_blind_and_why(tmp_path):
    ref = _csv(tmp_path / "r.csv", "a", ["1", "2"])
    img = tmp_path / "imgs"
    img.mkdir()
    (img / "x.png").write_bytes(b"\x89PNG")
    cases = [
        ([str(tmp_path / "missing.csv"), ref], "path not found"),
        ([ref, str(img)], "different kinds"),
        ([str(img), str(img)], "no installed plugin takes 'vision'"),
    ]
    for args, words in cases:
        res = runner.invoke(app, ["probe", *args, "--json", "--lang", "en"])
        body = json.loads(res.output.strip().splitlines()[-1])
        assert res.exit_code == 2 and body["blind"] and words in body["reason"], (args, body)


def test_probe_no_shared_columns_is_blind(tmp_path):
    ref = _csv(tmp_path / "r.csv", "a", ["1", "2"])
    cur = _csv(tmp_path / "c.csv", "b", ["1", "2"])
    res = runner.invoke(app, ["probe", ref, cur, "--json", "--lang", "en"])
    assert res.exit_code == 2 and "no column" in json.loads(res.output.strip().splitlines()[-1])["reason"]

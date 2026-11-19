"""Language selection for human-readable output (--lang / DDOC_LANG)."""
from __future__ import annotations

import json
import re
import string

from typer.testing import CliRunner

from ddoc.cli.main import app
from ddoc.i18n import MESSAGES, resolve_lang, t

runner = CliRunner()
HANGUL = re.compile(r"[가-힣]")


def _fields(template: str) -> set:
    return {f for _, f, _, _ in string.Formatter().parse(template) if f}


def test_catalogs_cover_the_same_keys_and_placeholders():
    en, ko = MESSAGES["en"], MESSAGES["ko"]
    assert set(en) == set(ko)
    for key in en:
        assert _fields(en[key]) == _fields(ko[key]), key


def test_resolve_lang(monkeypatch):
    monkeypatch.delenv("DDOC_LANG", raising=False)
    assert resolve_lang() == "en"
    assert resolve_lang("ko") == "ko"
    assert resolve_lang("ko_KR.UTF-8") == "ko"
    assert resolve_lang("fr") == "en"
    monkeypatch.setenv("DDOC_LANG", "ko")
    assert resolve_lang() == "ko"
    assert resolve_lang("en") == "en"


def test_unknown_key_falls_back_to_english_catalog():
    assert t("note.title", "xx") == MESSAGES["en"]["note.title"]


def _write_csv_pair(tmp_path):
    ref, cur = tmp_path / "ref.csv", tmp_path / "cur.csv"
    ref.write_text("color\n" + "red\n" * 40 + "blue\n" * 10)
    cur.write_text("color\n" + "red\n" * 5 + "blue\n" * 45)
    return ref, cur


def test_diff_defaults_to_english(tmp_path, monkeypatch):
    monkeypatch.delenv("DDOC_LANG", raising=False)
    ref, cur = _write_csv_pair(tmp_path)
    note = tmp_path / "note.html"
    res = runner.invoke(app, ["diff", str(ref), str(cur), "--report", str(note)])
    assert res.exit_code == 1
    assert not HANGUL.search(res.output)
    assert not HANGUL.search(note.read_text(encoding="utf-8"))


def test_diff_korean_report_and_console(tmp_path, monkeypatch):
    monkeypatch.delenv("DDOC_LANG", raising=False)
    ref, cur = _write_csv_pair(tmp_path)
    note = tmp_path / "note.html"
    res = runner.invoke(app, ["diff", str(ref), str(cur), "--report", str(note), "--lang", "ko"])
    assert res.exit_code == 1
    assert "심각도" in res.output
    body = note.read_text(encoding="utf-8")
    assert "<html lang='ko'>" in body
    assert MESSAGES["ko"]["note.title"] in body


def test_diff_json_keeps_machine_values_english(tmp_path, monkeypatch):
    monkeypatch.setenv("DDOC_LANG", "ko")
    ref, cur = _write_csv_pair(tmp_path)
    res = runner.invoke(app, ["diff", str(ref), str(cur), "--json"])
    payload = json.loads(res.output.strip().splitlines()[-1])
    assert payload["verdict"] == "DRIFT"
    assert payload["severity"] in ("high", "medium")
    assert any(HANGUL.search(h) for h in payload["hints"])


def test_report_render_korean_html_and_markdown(tmp_path):
    envelope = tmp_path / "drift.json"
    envelope.write_text(json.dumps({
        "status": "success", "modality": "tabular", "overall_score": 0.3,
        "attribute_drifts": {"price": 0.5},
    }))
    html_out, md_out = tmp_path / "r.html", tmp_path / "r.md"
    for out in (html_out, md_out):
        res = runner.invoke(app, ["report", "render", "-i", str(envelope), "-o", str(out), "--lang", "ko"])
        assert res.exit_code == 0, res.output
    assert MESSAGES["ko"]["report.drift_title"] in html_out.read_text(encoding="utf-8")
    assert MESSAGES["ko"]["report.attribute_drifts"] in md_out.read_text(encoding="utf-8")

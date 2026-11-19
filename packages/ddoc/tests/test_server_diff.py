"""``POST /diff`` — the verdict over HTTP, with the report inline."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client():
    from ddoc.server.app import create_app
    return TestClient(create_app(bind_info="testclient:0"))


def _csv(path, header, rows):
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return str(path)


def test_diff_returns_verdict_and_inline_report(client, tmp_path):
    ref = _csv(tmp_path / "r.csv", "channel", ["store"] * 75 + ["online"] * 25)
    cur = _csv(tmp_path / "c.csv", "channel", ["store"] * 20 + ["online"] * 80)
    r = client.post("/diff", json={"ref": ref, "cur": cur, "lang": "ko", "issuer": "Lab"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] == "DRIFT" and body["exit_code"] == 1  # an answer, not an error
    assert "데이터 드리프트 진단서" in body["report_html"] and "Lab" in body["report_html"]
    assert "report_path" not in body
    assert not any("report.html" in h for h in body["hints"])  # no server temp path leaks


def test_diff_ok_and_blind_are_answers_too(client, tmp_path):
    ref = _csv(tmp_path / "r.csv", "x", [str(i % 5) for i in range(100)])
    ok = client.post("/diff", json={"ref": ref, "cur": ref, "include_report": False})
    assert ok.status_code == 200 and ok.json()["verdict"] == "OK" and "report_html" not in ok.json()
    blind = client.post("/diff", json={"ref": ref, "cur": str(tmp_path / "missing.csv")})
    assert blind.status_code == 200 and blind.json()["verdict"] == "BLIND"
    assert blind.json()["exit_code"] == 2

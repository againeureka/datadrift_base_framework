"""The GUI's front page: choosing data without typing paths, example
cases with a known answer, and the diagnosis as display data."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from ddoc.examples import cases  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("DDOC_API_KEY", raising=False)
    monkeypatch.setenv("DDOC_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("DDOC_DATA_ROOT", str(tmp_path / "data"))
    (tmp_path / "data" / "sub").mkdir(parents=True)
    (tmp_path / "data" / "a.csv").write_text("x\n1\n2\n")
    (tmp_path / "data" / "sub" / "pic.png").write_bytes(b"\x89PNG")
    (tmp_path / "data" / ".hidden").write_text("x")
    (tmp_path / "secret.csv").write_text("x\n1\n")
    from ddoc.server.app import create_app
    return TestClient(create_app())


@pytest.mark.parametrize("case", [c for c in cases.CASES if c.folders is None], ids=lambda c: c.id)
def test_example_cases_get_the_verdict_they_promise(case, tmp_path):
    # the CSV cases; the image ones need the vision plugin (tests/test_dashcam.py)
    from ddoc.cli.commands.diff import compute_verdict, csv_diff

    ref, cur = cases.build(case.id, tmp_path)
    result = csv_diff([str(ref)], [str(cur)])
    assert compute_verdict(result, threshold=0.15, attr_threshold=0.25)[0] == case.verdict
    assert result["profile"]["ref"]["period"] and result["profile"]["cur"]["period"]   # dated, so periods show
    for lang in ("en", "ko"):
        assert all(case.text[lang][k] for k in ("title", "story", "look"))


def test_case_to_diagnosis_view(client):
    listed = client.get("/cases", params={"lang": "ko"}).json()
    assert [c["id"] for c in listed] == [c.id for c in cases.CASES] and listed[0]["title"]
    built = client.post("/cases/unit-change").json()
    assert client.post("/cases/nope").status_code == 404

    side = client.post("/describe", json={"path": built["ref"]}).json()
    assert side["kind"] == "csv" and side["profile"]["rows"] > 500 and side["profile"]["period"]
    assert {c["name"]: c["type"] for c in side["profile"]["columns"]}["date"] == "datetime"
    assert client.post("/describe", json={"path": built["ref"] + ".missing"}).json()["exists"] is False

    pair = client.post("/probe", json={"ref": built["ref"], "cur": built["cur"]}).json()
    assert pair["blind"] is False and "weight" in pair["csv"]["compared"] and pair["csv"]["time"] == ["date"]

    body = client.post("/diff", json={"ref": built["ref"], "cur": built["cur"], "lang": "ko"}).json()
    view = body["view"]
    assert view["verdict"] == "DRIFT" and view["token"] == "critical"
    assert "단위·스케일 변경형" in view["headline"]
    card = next(c for c in view["cards"] if c["code"] == "unit_change")
    assert card["match"] == "match" and card["attrs"] == ["weight"] and card["actions"] and card["signals"]
    top = view["attributes"][0]
    assert top["name"] == "weight" and top["status"] == "fail" and top["spark"].startswith("<svg") and top["summary"]
    assert view["sides"]["ref"]["period"]["start"] == "2026-04-01" and view["checks"]
    # dated rows: the noise test counted them by the day, and says so
    assert view["blocks"] and "기준 30일" in view["blocks"]


def test_view_tells_a_zero_within_noise_from_the_same(client):
    built = client.post("/cases/yearly-reset").json()
    view = client.post("/diff", json={"ref": built["ref"], "cur": built["cur"], "lang": "en"}).json()["view"]
    assert view["verdict"] == "OK"
    first = view["attributes"][0]          # the zeros, farthest first
    assert first["name"] == "dew_point_c" and first["score"] == 0 and first["distance"] > 0.3
    assert first["noise"].startswith("within noise (distance 0.31") and "counted by the day" in first["noise"]
    assert any("no more than sampling noise" in r for r in view["rationale"])


def test_uploads_keep_folders_and_stay_inside_the_session(client):
    up = client.post("/workspace/uploads").json()
    url = f"/workspace/uploads/{up['id']}"
    r = client.put(f"{url}/ref/week1/sales.csv", content=b"x,y\n1,2\n3,4\n")
    assert r.status_code == 200 and r.json()["size"] == 12 and r.json()["path"] == up["ref"]
    assert client.post("/describe", json={"path": up["ref"]}).json()["profile"]["rows"] == 2
    assert client.put(f"{url}/ref/.env", content=b"x").status_code == 400
    assert client.put(f"{url}/ref/a/%2e%2e/%2e%2e/%2e%2e/x.csv", content=b"x").status_code == 400
    assert client.put(f"{url}/other/x.csv", content=b"x").status_code == 404
    assert client.put("/workspace/uploads/up-000000000000/ref/x.csv", content=b"x").status_code == 404


def test_upload_size_limit(client, monkeypatch):
    monkeypatch.setenv("DDOC_UPLOAD_MAX_MB", "0.001")          # about 1 kB
    up = client.post("/workspace/uploads").json()
    r = client.put(f"/workspace/uploads/{up['id']}/cur/big.csv", content=b"x" * 5000)
    assert r.status_code == 413
    assert client.post("/describe", json={"path": up["cur"] + "/big.csv"}).json()["exists"] is False


def test_browsing_is_confined_to_the_data_roots(client, tmp_path):
    roots = client.get("/fs/list").json()
    assert roots["roots"] == [str((tmp_path / "data").resolve())]
    here = client.get("/fs/list", params={"path": roots["roots"][0]}).json()
    assert [e["name"] for e in here["entries"]] == ["sub", "a.csv"]      # folders first, hidden files left out
    assert here["counts"] == {"dir": 1, "csv": 1} and here["parent"] is None
    sub = client.get("/fs/list", params={"path": str(tmp_path / "data" / "sub")}).json()
    assert sub["parent"] == roots["roots"][0] and sub["entries"][0]["kind"] == "image"
    assert client.get("/fs/image", params={"path": sub["entries"][0]["path"]}).status_code == 200
    # outside the roots: not listed, not served — also through `..`
    assert client.get("/fs/list", params={"path": str(tmp_path)}).status_code == 403
    assert client.get("/fs/list", params={"path": str(tmp_path / "data" / "..")}).status_code == 403
    assert client.get("/fs/image", params={"path": str(tmp_path / "secret.csv")}).status_code == 403
    assert client.get("/fs/image", params={"path": str(tmp_path / "data" / "a.csv")}).status_code == 404


def test_old_sessions_are_swept(client, tmp_path):
    import os
    import time

    first = client.post("/workspace/uploads").json()
    old = time.time() - 2 * 24 * 3600
    os.utime(tmp_path / "workspace" / first["id"], (old, old))
    client.post("/workspace/uploads")
    assert not (tmp_path / "workspace" / first["id"]).exists()


def test_front_page_ships_the_workbench(client):
    page = client.get("/").text
    assert 'id="workbench"' in page and 'id="advanced"' in page and "/static/workbench.js" in page
    js = client.get("/static/workbench.js").text
    assert "ddoc serve" in js and "'/diff'" in js
    # the report is shown without same-origin access to the GUI page
    assert "sandbox: 'allow-scripts allow-modals allow-popups'" in js
    for key in ("intake.title", "res.patterns", "fs.use_folder"):       # both languages carry every key
        assert js.count(f"'{key}'") >= 2, key
    assert "ddoc serve" in client.get("/static/workbench.css").text


def test_report_render_passes_language(client):
    built = client.post("/cases/weekday-rhythm").json()
    body = client.post("/diff", json={"ref": built["ref"], "cur": built["cur"], "include_report": False}).json()
    envelope = {k: v for k, v in body.items() if k not in ("view", "exit_code")}
    r = client.post("/report/render", json={"envelope": envelope, "format": "md", "lang": "ko"})
    assert r.status_code == 200 and "데이터 드리프트 진단서" in r.text
    r = client.post("/report/render", json={"envelope": envelope, "format": "html", "lang": "ko", "issuer": "품질팀"})
    assert r.status_code == 200 and "품질팀" in r.text       # the issuer heads the HTML report

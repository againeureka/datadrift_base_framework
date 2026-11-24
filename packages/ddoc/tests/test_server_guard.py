"""A server without an API key answers only this machine.

Covers the guard against DNS rebinding (foreign Host) and cross-site
POSTs (foreign Origin), the key that lifts it, request validation that
keeps working for bodies that are not JSON, and values that would be
read as command-line options.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from ddoc.server import guard  # noqa: E402


def _client(monkeypatch, *, key=None, hosts=("localhost", "127.0.0.1", "::1")):
    from ddoc.server.app import create_app

    if key:
        monkeypatch.setenv("DDOC_API_KEY", key)
    else:
        monkeypatch.delenv("DDOC_API_KEY", raising=False)
    return TestClient(create_app(allowed_hosts=list(hosts) if hosts is not None else None),
                      base_url="http://127.0.0.1:8765", raise_server_exceptions=False)


def test_host_names(monkeypatch):
    monkeypatch.delenv("DDOC_ALLOWED_HOSTS", raising=False)
    assert guard.hostname("Example.com:8765") == "example.com"
    assert guard.hostname("[::1]:8765") == "::1"
    assert guard.hostname("::1") == "::1"
    assert guard.allowed_hosts("0.0.0.0") == ["localhost", "127.0.0.1", "::1"]
    assert guard.allowed_hosts("192.168.0.7")[-1] == "192.168.0.7"
    monkeypatch.setenv("DDOC_ALLOWED_HOSTS", "ddoc.lan, Box")
    assert guard.allowed_hosts("127.0.0.1")[-2:] == ["ddoc.lan", "box"]


def test_foreign_host_is_refused_without_a_key(monkeypatch):
    c = _client(monkeypatch)
    assert c.get("/scenarios").status_code == 200
    assert c.get("/scenarios", headers={"host": "localhost:8765"}).status_code == 200
    assert c.get("/scenarios", headers={"host": "[::1]:8765"}).status_code == 200
    r = c.get("/scenarios", headers={"host": "rebind.attacker.example:8765"})
    assert r.status_code == 403 and r.json()["error_code"] == "host_not_allowed"
    # monitors keep working from anywhere
    assert c.get("/healthz", headers={"host": "10.0.0.5:8765"}).status_code == 200


def test_cross_site_post_is_refused_without_a_key(monkeypatch):
    c = _client(monkeypatch)
    body = {"yaml": "name: t\nsteps: []\n"}
    r = c.post("/recipe/validate", json=body, headers={"origin": "https://attacker.example"})
    assert r.status_code == 403 and r.json()["error_code"] == "cross_origin_request"
    r = c.post("/recipe/validate", json=body, headers={"origin": "null"})
    assert r.status_code == 403
    # the GUI itself (same origin) and clients that send no Origin get through
    # (to the recipe check, which answers on its own: 400 for an empty recipe)
    for headers in ({"origin": "http://127.0.0.1:8765"}, {}):
        r = c.post("/recipe/validate", json=body, headers=headers)
        assert r.status_code != 403, r.text


def test_a_key_lifts_the_guard(monkeypatch):
    c = _client(monkeypatch, key="s3cret")
    headers = {"host": "ddoc.example.com", "origin": "https://ddoc.example.com", "x-api-key": "s3cret"}
    assert c.get("/scenarios", headers=headers).status_code == 200
    assert c.get("/scenarios", headers={**headers, "x-api-key": "s3creT"}).status_code == 401


def test_embedded_app_skips_only_the_host_check(monkeypatch):
    c = _client(monkeypatch, hosts=None)
    assert c.get("/scenarios", headers={"host": "anything:1"}).status_code == 200
    r = c.post("/recipe/validate", json={"yaml": "steps: []\n"}, headers={"origin": "https://x.example"})
    assert r.status_code == 403


def test_star_turns_the_host_check_off(monkeypatch):
    c = _client(monkeypatch, hosts=("localhost", "*"))
    assert c.get("/scenarios", headers={"host": "anything:1"}).status_code == 200


def test_body_that_is_not_json_is_a_422_not_a_crash(monkeypatch):
    c = _client(monkeypatch)
    body = json.dumps({"envelope": {"status": "success"}, "format": "md"}).encode()
    for headers in ({"content-type": "text/plain"}, {}):
        r = c.post("/report/render", content=body, headers=headers)
        assert r.status_code == 422, r.text
        assert r.json()["error_code"] == "invalid_request"


@pytest.mark.parametrize("path,body", [
    ("/diff", {"ref": "--report=/tmp/x.html", "cur": "b.csv"}),
    ("/fetch", {"source_uri": "--symlink", "dest": "out"}),
    ("/export/drift-report", {"input": "-x", "target": "file"}),
    ("/examples/generate", {"modality": "--help", "out": "o"}),
])
def test_values_read_as_options_are_refused(monkeypatch, path, body):
    r = _client(monkeypatch).post(path, json=body)
    assert r.status_code == 422, r.text


def test_recipe_positional_read_as_option_is_refused():
    from ddoc.core.recipe import RecipeError, _step_to_argv

    assert _step_to_argv("fetch", {"source_uri": "data/", "dest": "o"})[:2] == ["fetch", "data/"]
    with pytest.raises(RecipeError) as e:
        _step_to_argv("fetch", {"source_uri": "--symlink", "dest": "o"})
    assert e.value.code == "invalid_with"

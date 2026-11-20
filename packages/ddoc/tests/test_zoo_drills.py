"""Every drill and Drift Zoo story reads as it must (zoo/eval.py)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ZOO = Path(__file__).resolve().parent.parent / "zoo"


def test_every_case_reads_as_it_must():
    spec = importlib.util.spec_from_file_location("ddoc_zoo_eval", ZOO / "eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = mod.run()
    failures = {r["id"]: r["problems"] for r in rows if r["problems"]}
    assert failures == {}, failures
    # every archetype that can be drilled on CSV has a drill
    from ddoc import archetypes as ar
    drilled = {c for r in rows if r["kind"] == "drill" for c in r["expect"]}
    assert set(ar.DATA_CODES + ar.ATTR_CODES) - drilled == {"embedding_shift"}

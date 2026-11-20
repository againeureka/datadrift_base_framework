"""ddoc.embedding — the 2-D map of two embedding sets, and its report section."""
from __future__ import annotations

import json

import numpy as np

from ddoc.cli.commands.diff_report import build_consultation_html, for_paged_media
from ddoc.embedding import project


def _sets(offset, n=400, d=32, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.normal(size=(n, d)) + offset


def test_same_population_mixes_and_a_moved_one_separates():
    same = project(*_sets(0.0))
    moved = project(*_sets(1.0))
    assert same["mixing"] > 0.8 and same["centroid_shift"] < 0.3
    assert moved["mixing"] < 0.3 and moved["centroid_shift"] > 2
    json.dumps(moved)  # plain JSON for the envelope


def test_points_are_sampled_and_counts_kept():
    p = project(*_sets(0.5, n=1000), max_points=120)
    assert len(p["ref"]) == 120 and len(p["cur"]) == 120
    assert p["n"] == {"ref": 1000, "cur": 1000} and p["method"] == "pca"
    assert 0 < p["explained"][0] <= 1


def test_degenerate_input_returns_none():
    assert project(np.zeros((1, 4)), np.zeros((5, 4))) is None
    assert project(np.zeros((5, 4)), np.zeros((5, 3))) is None


def _payload(proj):
    return {"verdict": "DRIFT", "severity": "high", "overall_score": 0.4, "threshold": 0.15,
            "attr_threshold": 0.25, "top_attributes": [["brightness", 0.3]], "ref": "r", "cur": "c",
            "modality": "image", "engine": "plugin", "detector": "default",
            "raw": {"attribute_drifts": {"brightness": 0.3}, "embedding_projection": proj}, "hints": []}


def test_report_draws_the_map_and_pdf_drops_its_hover_targets():
    proj = project(*_sets(1.0, n=200))
    page = build_consultation_html(_payload(proj), lang="ko")
    assert "임베딩 지도" in page and "이웃 섞임" in page and "기준 중심" in page
    assert page.count("r='2.6'") == 400
    assert "<circle class='hit'" in page
    assert "<circle class='hit'" not in for_paged_media(page)


def test_no_map_without_a_projection():
    page = build_consultation_html(_payload(None), lang="en")
    assert "Embedding map" not in page


def test_identical_baseline_items_leave_the_shift_undefined():
    ref = np.tile(np.arange(8, dtype=float), (5, 1))
    cur = np.tile(np.arange(8, dtype=float)[::-1], (5, 1))
    p = project(ref, cur)
    assert p["centroid_shift"] is None and p["few_points"] is True
    page = build_consultation_html(_payload(p), lang="en")
    assert "sit on one spot" in page and "Fewer than 20 items" in page


def test_separated_embeddings_read_as_an_embedding_shift():
    from ddoc import diagnosis as dx
    from ddoc import archetypes as ar
    for offset, expected in ((1.0, "match"), (0.0, None)):
        proj = project(*_sets(offset, n=200))
        d = dx.build(verdict="DRIFT", overall=0.4, top=[("brightness", 0.3)], threshold=0.15,
                     attr_threshold=0.25, result={"attribute_drifts": {"brightness": 0.3},
                                                  "embedding_projection": proj})
        card = next((c for c in d["archetypes"] if c["code"] == "embedding_shift"), None)
        assert (card or {}).get("match") == expected
        if card:
            assert "이웃 섞임" in ar.card_text(card, "ko")["signals"][0]
    # a handful of items is not judged
    few = project(*_sets(1.0, n=10))
    d = dx.build(verdict="DRIFT", overall=0.4, top=[("a", 0.3)], threshold=0.15, attr_threshold=0.25,
                 result={"attribute_drifts": {"a": 0.3}, "embedding_projection": few})
    assert not any(c["code"] == "embedding_shift" for c in d["archetypes"])


def test_examples_name_typical_and_unlike_items():
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(60, 16))
    cur = rng.normal(size=(60, 16))
    cur[:3] += 6.0                                   # three current items far from everything
    names_r = [f"r{i:02d}.png" for i in range(60)]
    names_c = [f"c{i:02d}.png" for i in range(60)]
    p = project(ref, cur, ref_names=names_r, cur_names=names_c)
    ex = p["examples"]
    assert len(ex["ref_typical"]) == 5 and all(n.startswith("r") for n in ex["ref_typical"])
    assert set(ex["cur_unlike_ref"][:3]) == {"c00.png", "c01.png", "c02.png"}
    page = build_consultation_html(_payload(p), lang="ko")
    assert "기준과 가장 다른 현재 항목" in page and "c00.png" in page
    assert "examples" not in project(ref, cur)          # no names, no list

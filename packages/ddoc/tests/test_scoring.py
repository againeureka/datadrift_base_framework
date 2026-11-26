"""ddoc.scoring — one scale for every attribute: the JS distance, net of
sampling noise. The statistics are written without scipy; where scipy is
installed, they must agree with it."""
from __future__ import annotations

import math

import numpy as np
import pytest

from ddoc import scoring


def test_chi2_tail_and_g_test_agree_with_scipy():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(3)
    for x, df in [(0.5, 1), (3.84, 1), (10.0, 4), (30.0, 19), (120.0, 19), (2.0, 40)]:
        assert scoring.chi2_sf(x, df) == pytest.approx(stats.chi2.sf(x, df), rel=1e-10, abs=1e-14)
    for _ in range(50):
        k = int(rng.integers(2, 12))
        ref, cur = rng.integers(1, 80, k), rng.integers(1, 80, k)
        g, p, _, _ = stats.chi2_contingency(np.vstack([ref, cur]), correction=False,
                                            lambda_="log-likelihood")
        assert scoring.g_test_p(ref, cur) == pytest.approx(p, rel=1e-9, abs=1e-14)


def test_js_divergence_agrees_with_scipy():
    distance = pytest.importorskip("scipy.spatial.distance")
    rng = np.random.default_rng(4)
    for _ in range(50):
        ref, cur = rng.integers(0, 50, 10), rng.integers(0, 50, 10)
        if ref.sum() == 0 or cur.sum() == 0:
            continue
        assert scoring.js_divergence(ref, cur) == pytest.approx(
            distance.jensenshannon(ref, cur, base=2) ** 2, abs=1e-12)


def test_score_is_the_distance_only_beyond_noise():
    big_ref, big_cur = [400, 300, 200, 100], [100, 200, 300, 400]
    value, p = scoring.score(big_ref, big_cur)
    assert p < scoring.ALPHA and value == pytest.approx(math.sqrt(scoring.js_divergence(big_ref, big_cur)))
    # the same shapes on a handful of rows: chance could do it
    value, p = scoring.score([4, 3, 2, 1], [1, 2, 3, 4])
    assert p >= scoring.ALPHA and value == 0.0
    assert scoring.score([5, 5], [5, 5]) == (0.0, 1.0)
    assert scoring.score([3, 0], [0, 3])[0] == 0.0            # 3 against 3, fully apart: 1 in 20 by chance
    assert scoring.score([300, 0], [0, 300])[0] == pytest.approx(1.0)


def test_counts_score_refuses_what_is_not_counts():
    assert scoring.counts_score({"a": 30, "b": 10}, {"a": 10, "b": 30})[0] > 0
    assert scoring.counts_score({"mean": "x"}, {"mean": "y"}) is None


def test_embedding_separation_is_tested_on_neighbour_counts():
    one_population = {"mixing": 1.0, "neighbours": {"ref": 150, "total": 300, "expected": 0.5}}
    assert scoring.separation(one_population) == (0.0, 1.0)
    apart = {"mixing": 0.14, "neighbours": {"ref": 21, "total": 300, "expected": 0.5}}
    value, p = scoring.separation(apart)
    assert p < scoring.ALPHA and value == pytest.approx(0.86)
    slightly = {"mixing": 0.9, "neighbours": {"ref": 135, "total": 300, "expected": 0.5}}
    assert scoring.separation(slightly)[0] == 0.0             # within chance
    assert scoring.separation({"mixing": 0.1, "few_points": True,
                               "neighbours": {"ref": 1, "total": 50, "expected": 0.5}}) is None
    assert scoring.separation({"mixing": 0.5}) is None        # an older map without counts

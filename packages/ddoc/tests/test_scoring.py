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


def test_f_tail_agrees_with_scipy():
    stats = pytest.importorskip("scipy.stats")
    for x, d1, d2 in [(0.5, 1, 1), (3.0, 2.5, 7.3), (12.0, 6.2, 80.0), (40.0, 19, 4000.0),
                      (1.0, 30, 30), (2.0, 0.7, 5e5)]:
        assert scoring.f_sf(x, d1, d2) == pytest.approx(stats.f.sf(x, d1, d2), rel=1e-9, abs=1e-14)
    # beyond a million the F is read as a scaled χ²: off by about 1/d2
    assert scoring.f_sf(2.0, 0.7, 1e7) == pytest.approx(stats.f.sf(2.0, 0.7, 1e7), rel=1e-6)
    assert scoring.f_sf(0.0, 3, 9) == 1.0


def _days(rng, n_days, shift=0.0):
    """Ten-minute readings: each day its own weather, each reading close to the last."""
    out = []
    for _ in range(n_days):
        level, wobble = rng.normal(0, 2) + shift, 0.0
        for step in range(144):
            wobble = 0.95 * wobble + rng.normal(0, 0.3)
            out.append(level + 4 * math.sin(2 * math.pi * step / 144) + wobble)
    return np.array(out)


def _per_day(values, edges):
    cells = np.minimum(np.searchsorted(edges, values, side="right") - 1, len(edges) - 2)
    return scoring.block_counts(np.arange(len(values)) // 144, cells, len(edges) - 1)


def test_rows_close_in_time_are_counted_by_the_block():
    rng = np.random.default_rng(8)
    alarms_plain = alarms_blocked = 0
    for _ in range(40):
        ref, cur = _days(rng, 14), _days(rng, 14)
        edges = np.linspace(min(ref.min(), cur.min()), max(ref.max(), cur.max()), 21)
        R, C = _per_day(ref, edges), _per_day(cur, edges)
        alarms_plain += scoring.g_test_p(R.sum(0), C.sum(0)) < scoring.ALPHA
        p, effect = scoring.blocked_p(R, C)
        alarms_blocked += p < scoring.ALPHA
        assert p >= scoring.g_test_p(R.sum(0), C.sum(0))      # never more lenient
        assert effect > 1
    # the same weather on both sides: rows one by one call nearly every pair different
    assert alarms_plain >= 38 and alarms_blocked <= 1
    # a change bigger than the days' own swings still shows
    ref, cur = _days(rng, 14), _days(rng, 14, shift=8.0)
    edges = np.linspace(min(ref.min(), cur.min()), max(ref.max(), cur.max()), 21)
    value, p, effect = scoring.score_blocks(_per_day(ref, edges), _per_day(cur, edges))
    assert p < scoring.ALPHA and value > 0.5


def test_independent_rows_in_blocks_keep_the_plain_test():
    rng = np.random.default_rng(9)
    ref, cur = rng.integers(0, 5, 2000), rng.integers(0, 5, 2000)
    R = scoring.block_counts(np.arange(2000) // 100, ref, 5)
    C = scoring.block_counts(np.arange(2000) // 100, cur, 5)
    p, effect = scoring.blocked_p(R, C)
    assert effect == pytest.approx(1.0, abs=0.35)
    assert p == pytest.approx(scoring.g_test_p(R.sum(0), C.sum(0)), rel=0.6)
    # one block on a side: nothing to estimate the spread from
    assert scoring.blocked_p(R.sum(0, keepdims=True), C)[0] == scoring.g_test_p(R.sum(0), C.sum(0))


def test_time_blocks_are_days_or_equal_stretches():
    pd = pytest.importorskip("pandas")
    days = pd.date_range("2026-06-01", periods=14 * 144, freq="10min")
    hours = pd.date_range("2026-06-01", periods=12 * 6, freq="10min")       # twelve hours
    ref, cur, info = scoring.time_blocks(pd.Series(days), pd.Series(hours))
    assert info["ref"] == {"unit": "day", "blocks": 14}
    assert info["cur"] == {"unit": "stretch", "blocks": scoring.STRETCHES, "seconds": 7100}
    assert len(set(ref)) == 14 and len(set(cur)) == scoring.STRETCHES
    # text works too; a row without a time is a block of its own
    labels = scoring.time_blocks(["2026-06-01 00:00", None, "2026-06-03 00:00", "2026-06-05 00:00"],
                                 [str(t) for t in days[:500]])[0]
    assert labels[1] == -1 and len(set(labels)) == 4
    # a side that spans no time cannot be cut
    assert scoring.time_blocks(["2026-06-01"] * 5, [str(t) for t in days[:500]]) is None

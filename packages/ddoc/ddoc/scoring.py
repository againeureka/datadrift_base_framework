"""Attribute scores: how far two distributions sit apart, net of noise.

Every attribute is scored the same way, whatever produced it: a CSV
column, a property a plugin measured on each image, text or clip, the
counts of a categorical field. From the baseline and current counts over
shared bins (one per value for categories and few integers, otherwise
equal-width bins over the joint range):

* the **Jensen–Shannon distance** — the square root of the base-2
  divergence: 0 for the same distribution, 1 for no overlap;
* a **G-test of homogeneity** on the same bins. When sampling alone
  would explain the difference (p ≥ :data:`ALPHA`), the score is 0.

So the verdict thresholds (0.15 overall, 0.25 for one attribute) mean
the same thing for every kind of data, small samples do not raise
alarms on noise, and large ones are judged by the size of the change.
docs/sensitivity.md compares this rule with the ones it replaced.

The embedding map gets a score of the same kind: how far the current
items keep apart from the baseline's (1 − neighbour mixing), 0 when the
neighbour counts are what one population would give.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

ALPHA = 0.001   # significance for "more than sampling noise"; low enough that many columns stay quiet


def chi2_sf(x: float, df: int) -> float:
    """P(χ²_df ≥ x), the regularized upper incomplete gamma (no scipy)."""
    if df <= 0 or x <= 0:
        return 1.0
    a, z = df / 2.0, x / 2.0
    gln = math.lgamma(a)
    if z < a + 1.0:                                   # series for P, then 1 − P
        term = total = 1.0 / a
        ap = a
        for _ in range(1000):
            ap += 1.0
            term *= z / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-z + a * math.log(z) - gln))
    tiny = 1e-300                                     # continued fraction for Q (Lentz)
    b = z + 1.0 - a
    c, d = 1.0 / tiny, 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return min(1.0, math.exp(-z + a * math.log(z) - gln) * h)


def _pairs(ref: Sequence[float], cur: Sequence[float]):
    return [(float(r), float(c)) for r, c in zip(ref, cur) if float(r) + float(c) > 0]


def g_test_p(ref: Sequence[float], cur: Sequence[float]) -> float:
    """Homogeneity of two count vectors over the same bins (2×k G-test,
    empty bins dropped). 1.0 when there is nothing to compare."""
    pairs = _pairs(ref, cur)
    rt, ct = sum(r for r, _ in pairs), sum(c for _, c in pairs)
    if len(pairs) < 2 or rt == 0 or ct == 0:
        return 1.0
    total = rt + ct
    g = 0.0
    for r, c in pairs:
        col = r + c
        for obs, side in ((r, rt), (c, ct)):
            if obs > 0:
                g += 2.0 * obs * math.log(obs / (col * side / total))
    return chi2_sf(g, len(pairs) - 1)


def js_divergence(ref: Sequence[float], cur: Sequence[float]) -> float:
    """Jensen–Shannon divergence (base 2, 0–1) of two count vectors."""
    pairs = _pairs(ref, cur)
    rt, ct = sum(r for r, _ in pairs), sum(c for _, c in pairs)
    if rt == 0 or ct == 0:
        return 0.0
    out = 0.0
    for r, c in pairs:
        p, q = r / rt, c / ct
        m = (p + q) / 2
        if p > 0:
            out += 0.5 * p * math.log2(p / m)
        if q > 0:
            out += 0.5 * q * math.log2(q / m)
    return max(0.0, min(1.0, out))


def score(ref: Sequence[float], cur: Sequence[float]) -> Tuple[float, float]:
    """``(score, p)`` for two count vectors over the same bins: the JS
    distance, or 0 when the G-test finds no more than noise."""
    p = g_test_p(ref, cur)
    return (math.sqrt(js_divergence(ref, cur)) if p < ALPHA else 0.0), p


def counts_score(ref: Dict[str, Any], cur: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """``score`` for two ``{value: count}`` dicts; None if they are not counts."""
    keys = list(dict.fromkeys([*ref, *cur]))
    try:
        r = [float(ref.get(k) or 0) for k in keys]
        c = [float(cur.get(k) or 0) for k in keys]
    except (TypeError, ValueError):
        return None
    return score(r, c)


def separation(projection: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """``(score, p)`` for the embedding map: 1 − neighbour mixing, tested
    against the neighbour counts one population would give (1 df G-test);
    None when the map has no neighbour counts or too few points."""
    nb = projection.get("neighbours") if isinstance(projection, dict) else None
    if not isinstance(nb, dict) or projection.get("few_points"):
        return None
    total, observed, expected = nb.get("total") or 0, nb.get("ref") or 0, nb.get("expected") or 0.0
    if total <= 0 or not 0 < expected < 1:
        return None
    mixing = float(projection.get("mixing") or 0.0)
    want = total * expected
    if observed >= want:                              # at least as mixed as one population
        return 0.0, 1.0
    g = 0.0
    for obs, exp in ((observed, want), (total - observed, total - want)):
        if obs > 0:
            g += 2.0 * obs * math.log(obs / exp)
    p = chi2_sf(g, 1)
    return (max(0.0, 1.0 - mixing) if p < ALPHA else 0.0), p

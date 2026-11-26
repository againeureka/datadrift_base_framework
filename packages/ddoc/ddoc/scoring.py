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

**Rows in time order.** The G-test treats rows as independent. Readings
ten minutes apart are not: a humid morning is followed by a humid noon,
so two weeks of such rows hold far less information than their number,
and the plain test calls almost any two periods different. When the rows
carry a time, they are grouped into blocks (:func:`time_blocks`: calendar
days when each side covers at least :data:`MIN_DAYS` days, else
:data:`STRETCHES` equal stretches of each side) and the test asks whether
the two periods differ more than their blocks differ among themselves —
a second-order Rao–Scott correction of the G statistic, the standard one
for clustered samples, with an F reference on the blocks' degrees of
freedom (:func:`blocked_p`). It is never more lenient than the plain test.
Its assumption is that one block says little about the next;
docs/sensitivity.md measures how it holds up when days do resemble the
days before them.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

ALPHA = 0.001   # significance for "more than sampling noise"; low enough that many columns stay quiet
MIN_DAYS = 3    # calendar days needed on each side to group rows by day
STRETCHES = 6   # otherwise each side is cut into this many stretches of equal time


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


def _beta_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (tiny if abs(d) < tiny else d)
    h = d
    for m in range(1, 5000):
        for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                    -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + num * d
            d = 1.0 / (tiny if abs(d) < tiny else d)
            c = 1.0 + num / c
            c = tiny if abs(c) < tiny else c
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return h


def beta_inc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b) (no scipy)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_fraction(a, b, x) / a
    return 1.0 - front * _beta_fraction(b, a, 1.0 - x) / b


def f_sf(x: float, d1: float, d2: float) -> float:
    """P(F_{d1,d2} ≥ x); degrees of freedom need not be whole numbers."""
    if x <= 0:
        return 1.0
    if d2 > 1e6:                                      # the F is a scaled χ² by then
        return chi2_sf(x * d1, d1)
    return beta_inc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * x))


def _pairs(ref: Sequence[float], cur: Sequence[float]):
    return [(float(r), float(c)) for r, c in zip(ref, cur) if float(r) + float(c) > 0]


def _g_statistic(ref: Sequence[float], cur: Sequence[float]) -> Optional[Tuple[float, int]]:
    """The 2×k G statistic and its degrees of freedom (empty bins dropped);
    None when there is nothing to compare."""
    pairs = _pairs(ref, cur)
    rt, ct = sum(r for r, _ in pairs), sum(c for _, c in pairs)
    if len(pairs) < 2 or rt == 0 or ct == 0:
        return None
    total = rt + ct
    g = 0.0
    for r, c in pairs:
        col = r + c
        for obs, side in ((r, rt), (c, ct)):
            if obs > 0:
                g += 2.0 * obs * math.log(obs / (col * side / total))
    return g, len(pairs) - 1


def g_test_p(ref: Sequence[float], cur: Sequence[float]) -> float:
    """Homogeneity of two count vectors over the same bins (2×k G-test,
    empty bins dropped). 1.0 when there is nothing to compare."""
    stat = _g_statistic(ref, cur)
    return 1.0 if stat is None else chi2_sf(*stat)


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


def block_counts(labels: Sequence[Any], cells: Sequence[int], k: int):
    """Counts per block (rows) and bin (columns, ``k`` of them) from each
    row's block label and bin index."""
    import numpy as np

    _, block = np.unique(np.asarray(labels), return_inverse=True)
    out = np.zeros((int(block.max()) + 1 if len(block) else 0, k))
    np.add.at(out, (block.ravel(), np.asarray(cells, dtype=int)), 1.0)
    return out


def blocked_p(ref_blocks, cur_blocks) -> Tuple[float, float]:
    """``(p, design_effect)`` of the homogeneity test on counts kept per
    block (blocks × bins, one matrix per side), for rows that resemble
    their neighbours.

    The G statistic of the summed counts is corrected by the generalized
    design effects δ — how much more the bin shares vary between blocks
    than independent rows would make them vary — to second order
    (Satterthwaite: G·Σδ/Σδ² against χ² on (Σδ)²/Σδ² degrees of
    freedom, read as an F with the blocks' own degrees of freedom).
    The p is never below the plain test's; the design effect reported is
    the mean δ (at least 1): roughly, how many rows count as one."""
    import numpy as np

    R, C = np.asarray(ref_blocks, dtype=float), np.asarray(cur_blocks, dtype=float)
    r, c = R.sum(0), C.sum(0)
    stat = _g_statistic(r.tolist(), c.tolist())
    if stat is None:
        return 1.0, 1.0
    g, df = stat
    plain = chi2_sf(g, df)
    if len(R) < 2 or len(C) < 2:
        return plain, 1.0
    keep = (r + c) > 0
    R, C, r, c = R[:, keep], C[:, keep], r[keep], c[keep]
    nr, nc = r.sum(), c.sum()
    pooled = (r + c) / (nr + nc)
    parts = []
    for M, n in ((R, nr), (C, nc)):
        m = M.sum(1)
        parts.append((M - np.outer(m, M.sum(0) / n)) * math.sqrt(len(M) / (len(M) - 1)) / n)
    # δ are the eigenvalues of P⁻¹V, V = AᵀA the between-block covariance
    # of the share differences and P their covariance under independent
    # rows (multinomial; its inverse in closed form on k − 1 bins). Only
    # Σδ = tr(K) and Σδ² = ‖K‖² of K = A P⁻¹ Aᵀ (blocks × blocks) are needed.
    A = np.vstack(parts)[:, :-1]
    AP = (A / pooled[:-1] + A.sum(1, keepdims=True) / pooled[-1]) / (1.0 / nr + 1.0 / nc)
    K = AP @ A.T
    s1, s2 = float(np.trace(K)), float((K * K).sum())
    if s1 <= 0 or s2 <= 0:                            # every block alike: nothing to correct
        return plain, 1.0
    nu = s1 * s1 / s2
    p = f_sf(g * s1 / s2 / nu, nu, nu * (len(R) + len(C) - 2))
    return max(plain, p), max(1.0, s1 / df)


def score_blocks(ref_blocks, cur_blocks) -> Tuple[float, float, float]:
    """``(score, p, design_effect)``: :func:`score` for counts kept per
    block, the noise test being :func:`blocked_p`."""
    import numpy as np

    p, effect = blocked_p(ref_blocks, cur_blocks)
    r = np.asarray(ref_blocks, dtype=float).sum(0).tolist()
    c = np.asarray(cur_blocks, dtype=float).sum(0).tolist()
    return (math.sqrt(js_divergence(r, c)) if p < ALPHA else 0.0), p, effect


def _side_blocks(times) -> Tuple[Any, Dict[str, Any]]:
    import numpy as np
    import pandas as pd

    t = pd.to_datetime(pd.Series(list(times) if not isinstance(times, pd.Series) else times.to_numpy()),
                       errors="coerce")
    known = t.notna().to_numpy()
    labels = np.full(len(t), -1, dtype=np.int64)
    if not known.any():
        return labels, {"unit": "none", "blocks": 0}
    days = t[known].dt.floor("D")
    if days.nunique() >= MIN_DAYS:
        labels[known] = pd.factorize(days)[0]
        return labels, {"unit": "day", "blocks": int(days.nunique())}
    lo, hi = t[known].min(), t[known].max()
    span = (hi - lo).total_seconds()
    if span <= 0:
        return labels, {"unit": "none", "blocks": 1}
    where = ((t[known] - lo).dt.total_seconds() / span * STRETCHES).to_numpy()
    labels[known] = np.minimum(where.astype(np.int64), STRETCHES - 1)
    return labels, {"unit": "stretch", "blocks": int(len(np.unique(labels[known]))),
                    "seconds": round(span / STRETCHES)}


def time_blocks(ref_times, cur_times) -> Optional[Tuple[Any, Any, Dict[str, Any]]]:
    """Block labels for rows that carry a time, one array per side
    (anything ``pandas.to_datetime`` reads; rows without a time form one
    block of their own): calendar days when a side has rows on at least
    :data:`MIN_DAYS` days, else :data:`STRETCHES` stretches of equal time.
    Each side is cut on its own. None when a side spans no time."""
    ref_labels, ref_info = _side_blocks(ref_times)
    cur_labels, cur_info = _side_blocks(cur_times)
    if ref_info["blocks"] < 2 or cur_info["blocks"] < 2:
        return None
    return ref_labels, cur_labels, {"ref": ref_info, "cur": cur_info}


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

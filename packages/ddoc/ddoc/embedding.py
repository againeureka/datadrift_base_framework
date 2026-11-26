"""Embedding map — two sets of embeddings on one 2-D picture.

Plugins that compare embeddings (image, text) call :func:`project` and
put the result in their drift envelope as ``embedding_projection``; the
diagnostic report draws it as a scatter map. Only numpy is needed, so
the map works whatever model produced the vectors.

Besides the points it measures two things a reader can act on:

* ``centroid_shift`` — how far the current centre moved, in units of the
  baseline's spread on the map (0 = same centre); None when the baseline
  points sit on one spot.
* ``mixing`` — for each current point, the share of its nearest
  neighbours (in the full embedding space) that are baseline points,
  divided by the share expected if the two sets were one population.
  1 = evenly mixed, 0 = fully separated. The neighbour counts behind it
  ride along as ``neighbours`` so the separation can be tested against
  sampling noise (:func:`ddoc.scoring.separation`).

Given the item names, it also picks items to open and look at: the most
typical of each set (nearest its centre) and the current items least like
the baseline (farthest from their nearest baseline item).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

MAX_POINTS = 300      # points per side drawn on the map
FEW_POINTS = 20       # fewer per side and the map and its numbers are fragile
MAX_NEIGHBOUR = 400   # points per side used for the mixing measure
NEIGHBOURS = 5
EXAMPLES = 5          # items listed per group when names are given


def _sample(n: int, k: int, rng) -> "Any":
    import numpy as np

    return np.arange(n) if n <= k else np.sort(rng.choice(n, size=k, replace=False))


def project(ref, cur, *, ref_names: Optional[List[str]] = None, cur_names: Optional[List[str]] = None,
            max_points: int = MAX_POINTS, seed: int = 0) -> Optional[Dict[str, Any]]:
    """PCA of both sets together, plus centroid shift, mixing and — with
    names — items worth looking at.

    ``ref`` / ``cur`` are array-likes of shape (n, d). Returns None when
    either side has fewer than two vectors."""
    import numpy as np

    X, Y = np.asarray(ref, dtype=float), np.asarray(cur, dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or len(X) < 2 or len(Y) < 2 or X.shape[1] != Y.shape[1]:
        return None
    rng = np.random.default_rng(seed)

    both = np.vstack([X, Y])
    mean = both.mean(axis=0)
    _, s, vt = np.linalg.svd(both - mean, full_matrices=False)
    axes = vt[:2] if len(vt) >= 2 else np.vstack([vt, np.zeros_like(vt)])
    var = (s ** 2) / max(float((s ** 2).sum()), 1e-12)
    explained = [round(float(v), 4) for v in (list(var[:2]) + [0.0, 0.0])[:2]]
    px, py = (X - mean) @ axes.T, (Y - mean) @ axes.T

    cx, cy = px.mean(axis=0), py.mean(axis=0)
    spread = float(np.sqrt(((px - cx) ** 2).sum(axis=1).mean()))
    scale = float(np.sqrt(((np.vstack([px, py]) - np.vstack([px, py]).mean(axis=0)) ** 2).sum(axis=1).mean()))
    # Baseline points stacked on one spot (e.g. identical items): a move
    # "in units of its spread" means nothing, so it is left undefined.
    shift = None if spread <= 1e-6 * max(scale, 1e-12) else float(np.linalg.norm(cy - cx) / spread)

    # Mixing, measured in the full space on a sample of each side.
    xi, yi = _sample(len(X), MAX_NEIGHBOUR, rng), _sample(len(Y), MAX_NEIGHBOUR, rng)
    pool = np.vstack([X[xi], Y[yi]])
    is_ref = np.r_[np.ones(len(xi), bool), np.zeros(len(yi), bool)]
    sq = (pool ** 2).sum(axis=1)
    d2 = sq[len(xi):, None] + sq[None, :] - 2 * pool[len(xi):] @ pool.T
    d2[np.arange(len(yi)), len(xi) + np.arange(len(yi))] = np.inf  # not its own neighbour
    k = min(NEIGHBOURS, len(pool) - 1)
    nearest = np.argpartition(d2, k - 1, axis=1)[:, :k]
    observed = float(is_ref[nearest].mean())
    ref_neighbours = int(is_ref[nearest].sum())
    expected = len(xi) / (len(pool) - 1)
    mixing = min(1.0, observed / expected) if expected else 0.0

    examples = None
    if ref_names is not None and cur_names is not None and len(ref_names) == len(X) and len(cur_names) == len(Y):
        k = EXAMPLES
        near_ref = np.argsort(((X - X.mean(axis=0)) ** 2).sum(axis=1))[:k]
        near_cur = np.argsort(((Y - Y.mean(axis=0)) ** 2).sum(axis=1))[:k]
        # distance from each (sampled) current item to its nearest (sampled) baseline item
        far = np.sqrt(np.maximum(d2[:, :len(xi)].min(axis=1), 0))
        unlike = np.argsort(-far)[:k]
        examples = {"ref_typical": [str(ref_names[i]) for i in near_ref],
                    "cur_typical": [str(cur_names[i]) for i in near_cur],
                    "cur_unlike_ref": [str(cur_names[yi[i]]) for i in unlike]}

    sx, sy = _sample(len(X), max_points, rng), _sample(len(Y), max_points, rng)
    r4 = lambda a: [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in a]  # noqa: E731
    return {
        "method": "pca",
        "explained": explained,
        "ref": r4(px[sx]), "cur": r4(py[sy]),
        "n": {"ref": int(len(X)), "cur": int(len(Y))},
        "centroid": {"ref": r4([cx])[0], "cur": r4([cy])[0]},
        "centroid_shift": None if shift is None else round(shift, 4),
        "mixing": round(float(mixing), 4),
        "neighbours": {"ref": ref_neighbours, "total": int(nearest.size), "expected": round(float(expected), 6)},
        "few_points": bool(min(len(X), len(Y)) < FEW_POINTS),
        **({"examples": examples} if examples else {}),
    }

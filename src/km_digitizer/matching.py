from __future__ import annotations

from typing import List, Tuple
import itertools
import numpy as np

from .metrics import curve_iae


def match_curves_by_min_iae(
    true_curves: List[Tuple[np.ndarray, np.ndarray]],
    pred_curves: List[Tuple[np.ndarray, np.ndarray]],
) -> List[Tuple[int, int, float]]:
    """
    Match predicted curves to true curves by minimizing total IAE.
    Returns list of (true_idx, pred_idx, iae).
    """
    nt = len(true_curves)
    npred = len(pred_curves)
    if nt == 0 or npred == 0:
        return []

    cost = np.zeros((nt, npred), dtype=np.float64)
    for i, (tt, yt) in enumerate(true_curves):
        for j, (tp, yp) in enumerate(pred_curves):
            cost[i, j] = curve_iae(tt, yt, tp, yp)

    # Prefer scipy if available
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        r, c = linear_sum_assignment(cost)
        matches = [(int(i), int(j), float(cost[i, j])) for i, j in zip(r, c)]
        return matches
    except Exception:
        # brute force for small sizes (<=3)
        k = min(nt, npred)
        best = None
        best_val = float("inf")
        for perm in itertools.permutations(range(npred), k):
            val = sum(cost[i, perm[i]] for i in range(k))
            if val < best_val:
                best_val = val
                best = perm
        if best is None:
            return []
        return [(i, int(best[i]), float(cost[i, best[i]])) for i in range(k)]

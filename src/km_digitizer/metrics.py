from __future__ import annotations

from typing import Tuple
import numpy as np


def resample_on_grid(t: np.ndarray, y: np.ndarray, n: int = 1001) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate y(t) onto a uniform grid over [0,1] in t_norm.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    grid = np.linspace(0.0, 1.0, n, dtype=np.float64)

    # Ensure sorted by t
    order = np.argsort(t)
    t2 = t[order]
    y2 = y[order]

    # Clamp t to [0,1]
    t2 = np.clip(t2, 0.0, 1.0)

    # Make monotone non-decreasing t for interp
    # If duplicates exist, keep last
    uniq_t, uniq_idx = np.unique(t2, return_index=True)
    uniq_y = y2[uniq_idx]

    if uniq_t.size < 2:
        yy = np.full_like(grid, float(uniq_y[0] if uniq_y.size else np.nan))
        return grid, yy

    yy = np.interp(grid, uniq_t, uniq_y, left=uniq_y[0], right=uniq_y[-1])
    return grid, yy


def curve_ae(t_true: np.ndarray, y_true: np.ndarray, t_pred: np.ndarray, y_pred: np.ndarray, n: int = 1001) -> float:
    """
    Mean absolute error over a uniform grid in t_norm.
    """
    grid, yt = resample_on_grid(t_true, y_true, n=n)
    _, yp = resample_on_grid(t_pred, y_pred, n=n)
    return float(np.mean(np.abs(yt - yp)))


def curve_iae(t_true: np.ndarray, y_true: np.ndarray, t_pred: np.ndarray, y_pred: np.ndarray, n: int = 1001) -> float:
    """
    Integrated absolute error (area between curves) over a uniform grid in t_norm.
    """
    grid, yt = resample_on_grid(t_true, y_true, n=n)
    _, yp = resample_on_grid(t_pred, y_pred, n=n)
    return float(np.trapezoid(np.abs(yt - yp), grid))

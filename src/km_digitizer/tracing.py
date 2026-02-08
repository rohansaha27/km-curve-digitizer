from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def trace_curve_from_mask(mask255: np.ndarray, enforce_monotone: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Convert a curve mask (0/255) into sampled curve points (t_norm, s_norm).
    Strategy:
      - For each x column, compute y = mode of curve pixels
      - Interpolate missing columns
      - Convert to t_norm and survival s (1 - y/(h-1))
      - Optionally enforce non-increasing survival

    Returns None if too few valid columns.
    """
    if mask255.ndim != 2:
        raise ValueError("mask must be 2D")

    h, w = mask255.shape
    ys = np.full((w,), np.nan, dtype=np.float32)

    for x in range(w):
        col = mask255[:, x] > 0
        if not np.any(col):
            continue
        y_idx = np.where(col)[0].astype(np.int32)
        # mode of y positions (robust to censor '+' which adds outlier pixels)
        counts = np.bincount(y_idx, minlength=h)
        y_mode = int(np.argmax(counts))
        ys[x] = float(y_mode)

    valid = ~np.isnan(ys)
    if valid.mean() < 0.10:
        return None

    xs = np.arange(w, dtype=np.float32)
    ys_f = np.interp(xs, xs[valid], ys[valid]).astype(np.float32)

    # Convert to normalized curve
    t = xs / float(max(1, w - 1))
    s = 1.0 - (ys_f / float(max(1, h - 1)))

    # Clamp
    s = np.clip(s, 0.0, 1.0)

    if enforce_monotone:
        # enforce non-increasing survival over time: s[i] <= s[i-1]
        s_adj = s.copy()
        for i in range(1, s_adj.size):
            if s_adj[i] > s_adj[i - 1]:
                s_adj[i] = s_adj[i - 1]
        s = s_adj

    return t.astype(np.float32), s.astype(np.float32)

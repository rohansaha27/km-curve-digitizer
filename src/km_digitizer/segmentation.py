from __future__ import annotations

from typing import List, Tuple
import numpy as np
import cv2

from .config import DigitizeConfig


def _circular_hue_dist(h: np.ndarray, peak: int) -> np.ndarray:
    # hue range in OpenCV is [0,179]
    d1 = (h.astype(np.int16) - int(peak)) % 180
    d2 = (int(peak) - h.astype(np.int16)) % 180
    return np.minimum(d1, d2)


def _smooth_hist(hist: np.ndarray) -> np.ndarray:
    # simple circular smoothing
    k = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    k /= k.sum()
    hist_pad = np.concatenate([hist[-1:], hist, hist[:1]])
    sm = np.convolve(hist_pad, k, mode="same")[1:-1]
    return sm


def _find_hue_peaks(hues: np.ndarray, max_peaks: int, min_frac: float) -> List[int]:
    if hues.size == 0:
        return []
    hist = np.bincount(hues, minlength=180).astype(np.float32)
    total = hist.sum()
    if total <= 0:
        return []
    hist /= total
    sm = _smooth_hist(hist)

    # naive peak finding
    peaks: List[Tuple[int, float]] = []
    for i in range(180):
        prev = sm[(i - 1) % 180]
        cur = sm[i]
        nxt = sm[(i + 1) % 180]
        if cur >= prev and cur >= nxt and cur >= min_frac:
            peaks.append((i, float(cur)))

    peaks.sort(key=lambda x: x[1], reverse=True)

    chosen: List[int] = []
    for p, val in peaks:
        # avoid picking peaks too close to existing peaks
        if any(min((p - c) % 180, (c - p) % 180) < 2 * 10 for c in chosen):
            continue
        chosen.append(p)
        if len(chosen) >= max_peaks:
            break
    return chosen


def segment_curves_by_hue(plot_bgr: np.ndarray, cfg: DigitizeConfig) -> List[np.ndarray]:
    """
    Returns list of binary masks (uint8 0/255) for each curve cluster.
    """
    hsv = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    colored = s > cfg.curve_saturation_thresh
    masks: List[np.ndarray] = []

    # If requested, also allow dark pixels as candidates (helps black curves, but can pick up text)
    if cfg.include_dark_pixels:
        dark = v < cfg.dark_value_thresh
        cand = colored | dark
    else:
        cand = colored

    # Hue peaks computed only on colored pixels (more stable)
    hue_vals = h[colored].astype(np.int32)
    peaks = _find_hue_peaks(hue_vals, max_peaks=cfg.max_curves, min_frac=cfg.hue_peak_min_frac)

    if not peaks:
        # fallback: single mask from candidates
        m = (cand.astype(np.uint8) * 255)
        masks.append(_postprocess_mask(m, cfg))
        return [m for m in masks if m is not None and m.any()]

    for peak in peaks:
        dist = _circular_hue_dist(h, peak)
        m = (cand & (dist <= cfg.hue_band_radius)).astype(np.uint8) * 255
        m2 = _postprocess_mask(m, cfg)
        if m2 is not None and m2.any():
            masks.append(m2)

    return masks


def _postprocess_mask(mask255: np.ndarray, cfg: DigitizeConfig) -> np.ndarray:
    """
    Close small gaps and keep components above min area.
    Returns a 0/255 uint8 mask.
    """
    m = mask255.copy()
    if cfg.morph_close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_close_ksize, cfg.morph_close_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=cfg.morph_iters)
    if cfg.morph_dilate_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_dilate_ksize, cfg.morph_dilate_ksize))
        m = cv2.dilate(m, k, iterations=cfg.morph_iters)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, num):
        x, y, w, h, area = stats[i].tolist()
        if area >= cfg.min_component_area:
            out[labels == i] = 255
    return out

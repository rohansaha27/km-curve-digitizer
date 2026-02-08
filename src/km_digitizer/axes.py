from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2

from .config import DigitizeConfig
from .types import BBox


@dataclass
class AxisCandidates:
    y_axis: Optional[Tuple[int, int, int, int]] = None  # (x,y,w,h)
    x_axis: Optional[Tuple[int, int, int, int]] = None  # (x,y,w,h)


def _binary_dark_structures(gray: np.ndarray) -> np.ndarray:
    """
    Create a binary mask of dark lines/text on light background.
    Returns uint8 mask with 255 for foreground.
    """
    # Adaptive threshold on inverted polarity: dark -> white in mask
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    return bw


def _extract_long_lines(bw: np.ndarray, orientation: str, min_len_px: int) -> np.ndarray:
    """
    Morphologically extract long vertical/horizontal structures.
    """
    h, w = bw.shape[:2]
    if orientation == "vertical":
        k = max(10, min_len_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
    elif orientation == "horizontal":
        k = max(10, min_len_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    else:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened


def _connected_components(mask: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
    """
    Return list of components as (x,y,w,h,area), excluding background.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out: List[Tuple[int, int, int, int, int]] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i].tolist()
        out.append((x, y, w, h, area))
    return out


def _curve_candidate_mask(img_bgr: np.ndarray, cfg: DigitizeConfig) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    colored = s > cfg.curve_saturation_thresh
    if cfg.include_dark_pixels:
        dark = v < cfg.dark_value_thresh
        mask = (colored | dark).astype(np.uint8)
    else:
        mask = colored.astype(np.uint8)
    return mask


def detect_plot_bbox(img_bgr: np.ndarray, cfg: DigitizeConfig) -> Tuple[BBox, AxisCandidates]:
    """
    Detect plot panel bounding box (excluding risk table if present) using line morphology + curve proximity scoring.

    Returns:
      plot_bbox: BBox in original image coordinates
      axes: AxisCandidates (y_axis and x_axis bounding boxes)
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    bw = _binary_dark_structures(gray)

    min_vlen = int(h * cfg.axis_min_length_frac)
    min_hlen = int(w * cfg.axis_min_length_frac)

    vert = _extract_long_lines(bw, "vertical", min_vlen // 2)
    horiz = _extract_long_lines(bw, "horizontal", min_hlen // 2)

    vcomps = _connected_components(vert)
    hcomps = _connected_components(horiz)

    curve_mask = _curve_candidate_mask(img_bgr, cfg)

    # --- select y-axis by scoring "curve pixels to the right"
    y_axis_best = None
    best_score = -1.0
    for (x, y, ww, hh, area) in vcomps:
        if hh < min_vlen:
            continue
        if ww > cfg.axis_max_thickness_px:
            continue
        # score strip to right
        strip_x0 = min(w - 1, x + ww)
        strip_x1 = min(w, strip_x0 + cfg.axis_score_strip_px)
        strip_y0 = max(0, y)
        strip_y1 = min(h, y + hh)
        if strip_x1 <= strip_x0 or strip_y1 <= strip_y0:
            continue
        ssum = float(curve_mask[strip_y0:strip_y1, strip_x0:strip_x1].sum())
        denom = float((strip_y1 - strip_y0) * (strip_x1 - strip_x0) + 1)
        score = ssum / denom
        if score > best_score:
            best_score = score
            y_axis_best = (x, y, ww, hh)

    # Fallback if no axis found: assume left margin is y-axis
    if y_axis_best is None:
        y_axis_best = (max(0, int(0.08 * w)), 0, 2, h)

    yx, yy, yw, yh = y_axis_best
    y_axis_bottom = yy + yh

    # --- select x-axis: horizontal line crossing y-axis near bottom of y-axis
    x_axis_best = None
    best_y = -1
    for (x, y, ww, hh, area) in hcomps:
        if ww < min_hlen:
            continue
        if hh > cfg.axis_max_thickness_px:
            continue
        # must cross y-axis x coordinate
        if not (x <= yx <= x + ww):
            continue
        # must be below some fraction of y-axis
        if y < yy + int(0.3 * yh):
            continue
        if y > best_y and y <= y_axis_bottom + int(0.1 * h):
            best_y = y
            x_axis_best = (x, y, ww, hh)

    # Fallback: use bottom of curve pixels (common if x-axis is faint)
    if x_axis_best is None:
        # find bottom-most curve pixels (but ignore bottom 20% where risk table might be)
        search_ymax = int(0.85 * h)
        ys, xs = np.where(curve_mask[:search_ymax, :] > 0)
        if len(ys) > 0:
            bottom = int(np.percentile(ys, 99))
            x_axis_best = (0, bottom, w, 2)
        else:
            x_axis_best = (0, int(0.85 * h), w, 2)

    xx, xy, xw, xh = x_axis_best

    # --- define rough plot ROI: right of y-axis, above x-axis
    left = int(yx + yw + 2)
    bottom = int(xy)

    left = max(0, min(left, w - 1))
    bottom = max(1, min(bottom, h))

    # Use curve pixels within [0:bottom, left:w] to refine top/right
    roi_mask = curve_mask[:bottom, left:w]
    ys, xs = np.where(roi_mask > 0)
    if len(xs) > 0:
        x0 = left + int(np.min(xs))
        x1 = left + int(np.max(xs)) + 1
        y0 = int(np.min(ys))
        y1 = int(np.max(ys)) + 1
        # pad a bit
        pad_x = max(4, int(0.01 * w))
        pad_y = max(4, int(0.01 * h))
        plot_bbox = BBox(x0 - pad_x, y0 - pad_y, x1 + pad_x, bottom).clip(w, h)
    else:
        # fallback to simple rectangle
        plot_bbox = BBox(left, int(0.05 * h), w, bottom).clip(w, h)

    axes = AxisCandidates(y_axis=y_axis_best, x_axis=x_axis_best)
    return plot_bbox, axes

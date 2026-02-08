from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple
import os
import numpy as np
import cv2

from .config import DigitizeConfig
from .types import DigitizedCurve, BBox
from .axes import detect_plot_bbox
from .segmentation import segment_curves_by_hue
from .tracing import trace_curve_from_mask
from .ocr import calibrate_y_from_ticks


def digitize_km_curves(
    image_path: str,
    cfg: Optional[DigitizeConfig] = None,
) -> Tuple[List[DigitizedCurve], Dict[str, Any]]:
    """
    Digitize KM curves from a raster image.

    Returns (curves, debug_info) where:
      curves: list of DigitizedCurve with (t_norm, s)
      debug_info: dict with plot_bbox, axis bbox, optional calibration, and debug images (if cfg.debug)
    """
    cfg = cfg or DigitizeConfig()

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    H, W = img_bgr.shape[:2]

    plot_bbox, axes = detect_plot_bbox(img_bgr, cfg)
    plot_bbox = plot_bbox.clip(W, H)

    plot = img_bgr[plot_bbox.y0:plot_bbox.y1, plot_bbox.x0:plot_bbox.x1].copy()

    # Segment curve masks
    masks = segment_curves_by_hue(plot, cfg)

    # Optional OCR y-axis calibration
    y_cal = None
    if axes.y_axis is not None:
        y_cal = calibrate_y_from_ticks(img_bgr, plot_bbox, axes.y_axis, cfg)

    curves: List[DigitizedCurve] = []
    for i, m in enumerate(masks):
        traced = trace_curve_from_mask(m, enforce_monotone=cfg.enforce_monotone)
        if traced is None:
            continue
        t, s = traced

        # Apply calibration if available
        if y_cal is not None:
            a, b = y_cal
            # Here s is y_norm-based survival (0..1). Convert to calibrated units:
            s = a * s + b

        curves.append(DigitizedCurve(curve_id=i, t_norm=t, s=s))

    debug_info: Dict[str, Any] = {
        "image_path": image_path,
        "plot_bbox": plot_bbox,
        "axes": axes,
        "y_calibration": y_cal,
    }

    if cfg.debug:
        debug_info["debug_overlay_bgr"] = _make_debug_overlay(img_bgr, plot_bbox, curves)

    return curves, debug_info


def _make_debug_overlay(img_bgr: np.ndarray, plot_bbox: BBox, curves: List[DigitizedCurve]) -> np.ndarray:
    overlay = img_bgr.copy()
    # plot bbox
    cv2.rectangle(overlay, (plot_bbox.x0, plot_bbox.y0), (plot_bbox.x1, plot_bbox.y1), (0, 255, 255), 2)

    # draw curves
    for c in curves:
        # map (t_norm,s_norm) into plot pixel coords (assumes s in [0,1] for drawing)
        t = c.t_norm
        s = c.s
        # If calibrated, s might not be in [0,1]. Only draw if it looks like probabilities.
        if np.nanmin(s) < -0.05 or np.nanmax(s) > 1.05:
            continue

        xs = plot_bbox.x0 + (t * (plot_bbox.w - 1)).astype(np.int32)
        ys = plot_bbox.y0 + ((1.0 - s) * (plot_bbox.h - 1)).astype(np.int32)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

    return overlay

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import re
import numpy as np
import cv2

from .config import DigitizeConfig
from .types import BBox


@dataclass
class TickReading:
    value: float
    y_px: float
    conf: float


_FLOAT_RE = re.compile(r"(?<!\d)(\d+(\.\d+)?)(?!\d)")


def _try_import_tesseract():
    try:
        import pytesseract  # type: ignore
        return pytesseract
    except Exception:
        return None


def calibrate_y_from_ticks(
    img_bgr: np.ndarray,
    plot_bbox: BBox,
    y_axis_bbox: Tuple[int, int, int, int],
    cfg: DigitizeConfig,
) -> Optional[Tuple[float, float]]:
    """
    Attempt to calibrate survival axis S = a * y_norm + b using OCR tick labels on the left margin.

    Returns (a, b) mapping from y_norm in [0,1] (top=1) to S in real plot units.
    If OCR is unavailable or insufficient, returns None.
    """
    if not cfg.enable_ocr_y_calibration:
        return None

    pytesseract = _try_import_tesseract()
    if pytesseract is None:
        return None

    h, w = img_bgr.shape[:2]
    yx, yy, yw, yh = y_axis_bbox

    # Search region: left of y-axis and aligned vertically with plot bbox
    margin_w = int(cfg.ocr_left_margin_frac * w)
    x0 = max(0, yx - margin_w)
    x1 = max(0, yx + 2)  # include a tiny bit into axis
    y0 = max(0, plot_bbox.y0)
    y1 = min(h, plot_bbox.y1)

    if x1 <= x0 + 5 or y1 <= y0 + 5:
        return None

    roi = img_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # binarize to help OCR
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)

    # Use tesseract to get word-level boxes
    config = "--psm 6 -c tessedit_char_whitelist=0123456789."
    data = pytesseract.image_to_data(bw, output_type=pytesseract.Output.DICT, config=config)

    ticks: List[TickReading] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        conf = float(data.get("conf", [0] * n)[i])
        if conf < cfg.ocr_confidence_min:
            continue

        m = _FLOAT_RE.search(text)
        if not m:
            continue
        val = float(m.group(1))

        # y position of the text box center (relative to roi -> absolute)
        top = float(data["top"][i])
        height = float(data["height"][i])
        y_center = (y0 + top + 0.5 * height)

        ticks.append(TickReading(value=val, y_px=y_center, conf=conf))

    # Need >=2 ticks
    if len(ticks) < 2:
        return None

    # Convert y_px to y_norm (top=1, bottom=0) based on plot bbox
    ys = np.array([t.y_px for t in ticks], dtype=np.float32)
    y_norm = 1.0 - ((ys - float(plot_bbox.y0)) / float(max(1, plot_bbox.h)))

    vals = np.array([t.value for t in ticks], dtype=np.float32)

    # Robust-ish: drop extreme values by quantiles
    qlo, qhi = np.quantile(vals, [0.05, 0.95])
    keep = (vals >= qlo) & (vals <= qhi)
    if keep.sum() < 2:
        keep = np.ones_like(vals, dtype=bool)

    y_norm = y_norm[keep]
    vals = vals[keep]

    # Fit vals ≈ a*y_norm + b
    A = np.stack([y_norm, np.ones_like(y_norm)], axis=1)
    try:
        sol, *_ = np.linalg.lstsq(A, vals, rcond=None)
        a, b = float(sol[0]), float(sol[1])
    except Exception:
        return None

    # sanity: survival axis typically increases with y_norm (higher on plot = higher survival)
    if a <= 0:
        return None

    return a, b

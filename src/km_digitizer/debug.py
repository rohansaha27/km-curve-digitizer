from __future__ import annotations

from typing import List
import numpy as np
import cv2

from .types import BBox, DigitizedCurve


def save_debug_overlay(path: str, overlay_bgr: np.ndarray) -> None:
    cv2.imwrite(path, overlay_bgr)


def curves_to_csv_rows(curves: List[DigitizedCurve]) -> List[dict]:
    rows = []
    for c in curves:
        for t, s in zip(c.t_norm.tolist(), c.s.tolist()):
            rows.append({"curve_id": c.curve_id, "t_norm": float(t), "survival": float(s)})
    return rows

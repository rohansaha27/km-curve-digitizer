from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in pixel coords (x0,y0) inclusive, (x1,y1) exclusive."""
    x0: int
    y0: int
    x1: int
    y1: int

    def clip(self, w: int, h: int) -> "BBox":
        x0 = max(0, min(self.x0, w))
        x1 = max(0, min(self.x1, w))
        y0 = max(0, min(self.y0, h))
        y1 = max(0, min(self.y1, h))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return BBox(x0, y0, x1, y1)

    @property
    def w(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def h(self) -> int:
        return max(0, self.y1 - self.y0)


@dataclass
class DigitizedCurve:
    curve_id: int
    t_norm: np.ndarray  # shape [N], in [0,1]
    s: np.ndarray       # shape [N], survival prob in [0,1] (or calibrated range)
    color_hint: Optional[str] = None

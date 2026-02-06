"""
Shared utilities for KM curve digitization.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def pixel_to_data(
    px: float, py: float,
    plot_bbox: Dict[str, float],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> Tuple[float, float]:
    """Convert pixel coordinates to data coordinates.

    Args:
        px, py: Pixel coordinates (origin at top-left).
        plot_bbox: Dict with keys 'x_min', 'x_max', 'y_min', 'y_max' in pixel space.
            y_min is the TOP of the plot area (smaller pixel value),
            y_max is the BOTTOM of the plot area (larger pixel value).
        x_range: (data_x_min, data_x_max).
        y_range: (data_y_min, data_y_max).

    Returns:
        (data_x, data_y) in data coordinates.
    """
    # Normalize pixel position within the plot bounding box
    x_frac = (px - plot_bbox['x_min']) / (plot_bbox['x_max'] - plot_bbox['x_min'])
    # y is inverted: top of image is y_min pixel but y_max data
    y_frac = 1.0 - (py - plot_bbox['y_min']) / (plot_bbox['y_max'] - plot_bbox['y_min'])

    data_x = x_range[0] + x_frac * (x_range[1] - x_range[0])
    data_y = y_range[0] + y_frac * (y_range[1] - y_range[0])

    return data_x, data_y


def data_to_pixel(
    data_x: float, data_y: float,
    plot_bbox: Dict[str, float],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> Tuple[float, float]:
    """Convert data coordinates to pixel coordinates.

    Returns:
        (px, py) in pixel coordinates (origin at top-left).
    """
    x_frac = (data_x - x_range[0]) / (x_range[1] - x_range[0])
    y_frac = (data_y - y_range[0]) / (y_range[1] - y_range[0])

    px = plot_bbox['x_min'] + x_frac * (plot_bbox['x_max'] - plot_bbox['x_min'])
    py = plot_bbox['y_min'] + (1.0 - y_frac) * (plot_bbox['y_max'] - plot_bbox['y_min'])

    return px, py


def enforce_monotonicity(values: np.ndarray) -> np.ndarray:
    """Enforce non-increasing monotonicity on a survival curve.

    Uses a running minimum approach - at each point, the value should be
    at most the minimum of all previous values.
    """
    result = np.copy(values)
    running_min = result[0]
    for i in range(1, len(result)):
        if result[i] > running_min:
            result[i] = running_min
        else:
            running_min = result[i]
    return result


def interpolate_step_function(
    times: np.ndarray,
    survival: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Interpolate a KM step function at given query times.

    Uses left-continuous interpolation (the KM convention):
    at a step time t_i, the value is the NEW (lower) value.
    Between steps, the value is constant.
    """
    result = np.zeros_like(query_times, dtype=float)
    for i, t in enumerate(query_times):
        if t < times[0]:
            result[i] = 1.0  # Before first event, survival = 1
        elif t >= times[-1]:
            result[i] = survival[-1]
        else:
            # Find the last time point <= t
            idx = np.searchsorted(times, t, side='right') - 1
            result[i] = survival[idx]
    return result


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex string."""
    return f'#{r:02x}{g:02x}{b:02x}'


def load_ground_truth(json_path: str) -> Dict:
    """Load ground truth data from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_ground_truth(data: Dict, json_path: str):
    """Save ground truth data to JSON."""
    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    serializable = json.loads(json.dumps(data, default=convert))
    with open(json_path, 'w') as f:
        json.dump(serializable, f, indent=2)


def color_distance_lab(color1_bgr: np.ndarray, color2_bgr: np.ndarray) -> float:
    """Compute perceptual color distance using CIELAB.

    Args:
        color1_bgr, color2_bgr: Colors in BGR format (OpenCV convention).

    Returns:
        Delta-E distance (lower = more similar).
    """
    import cv2
    c1 = np.uint8([[color1_bgr]])
    c2 = np.uint8([[color2_bgr]])
    lab1 = cv2.cvtColor(c1, cv2.COLOR_BGR2LAB).astype(float)[0, 0]
    lab2 = cv2.cvtColor(c2, cv2.COLOR_BGR2LAB).astype(float)[0, 0]
    return np.sqrt(np.sum((lab1 - lab2) ** 2))

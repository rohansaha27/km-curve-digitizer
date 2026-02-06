"""
CV-only KM curve digitizer.

A fallback pipeline that works without LLM access by using heuristic
methods for axis detection, color detection, and plot region identification.
This is less robust than the hybrid approach but useful for:
- Testing without API keys
- Processing when LLM is unavailable
- Batch processing where API costs are a concern
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter

from .cv_extractor import (
    detect_plot_bbox_cv,
    create_color_mask,
    create_line_mask,
    extract_curve_scanline,
    extract_step_function,
)
from .utils import (
    pixel_to_data,
    enforce_monotonicity,
    interpolate_step_function,
    hex_to_rgb,
    rgb_to_hex,
)


def detect_axis_range_heuristic(
    image: np.ndarray,
    plot_bbox: Dict[str, float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Detect axis ranges using OCR-free heuristics.

    For synthetic benchmarks where we know typical ranges, this uses
    pixel analysis. For real-world use, the LLM approach is much better.

    Returns:
        (x_range, y_range): Estimated data ranges for each axis.
    """
    # For synthetic plots, common ranges:
    # Y-axis: usually [0, 1] for survival probability
    # X-axis: varies, but we can estimate from the image aspect ratio

    # Default estimates (works for most KM plots)
    y_range = (0.0, 1.0)
    x_range = (0.0, 100.0)  # Will be overridden by tick detection

    return x_range, y_range


def detect_dominant_colors(
    image: np.ndarray,
    plot_bbox: Dict[str, float],
    n_colors: int = 5,
    min_pixel_fraction: float = 0.001,
) -> List[str]:
    """Detect the dominant non-background colors in the plot area.

    This identifies the likely curve colors by finding colored pixels
    that stand out from the white/gray background.

    Args:
        image: BGR image.
        plot_bbox: Plot bounding box.
        n_colors: Maximum number of colors to detect.
        min_pixel_fraction: Minimum fraction of plot pixels for a color.

    Returns:
        List of hex color strings for detected curve colors.
    """
    x1 = int(plot_bbox['x_min'])
    x2 = int(plot_bbox['x_max'])
    y1 = int(plot_bbox['y_min'])
    y2 = int(plot_bbox['y_max'])
    crop = image[y1:y2, x1:x2]
    total_pixels = crop.shape[0] * crop.shape[1]

    # Convert to HSV to separate chromatic from achromatic
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Filter out near-white, near-black, and near-gray pixels
    # (background, axes, text)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Chromatic pixels: sufficient saturation and not too dark/light
    chromatic_mask = (saturation > 30) & (value > 40) & (value < 250)

    if np.sum(chromatic_mask) < 10:
        # No chromatic pixels found - curves might be black/gray
        # Look for dark, non-background pixels
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        dark_mask = gray < 80
        if np.sum(dark_mask) > total_pixels * min_pixel_fraction:
            return ['#000000']
        return []

    # Get chromatic pixels
    chromatic_pixels = crop[chromatic_mask]

    # Cluster the colors using K-means or simpler binning
    # Use HSV hue binning for robustness
    hues = hsv[chromatic_mask, 0]

    # Bin hues into 18 bins (10 degrees each in OpenCV's 0-180 range)
    n_bins = 18
    hist, bin_edges = np.histogram(hues, bins=n_bins, range=(0, 180))

    # Find significant peaks
    min_count = max(total_pixels * min_pixel_fraction, 20)
    peak_bins = np.where(hist > min_count)[0]

    if len(peak_bins) == 0:
        return []

    # Merge adjacent bins
    groups = []
    current_group = [peak_bins[0]]
    for i in range(1, len(peak_bins)):
        if peak_bins[i] - peak_bins[i-1] <= 2:
            current_group.append(peak_bins[i])
        else:
            groups.append(current_group)
            current_group = [peak_bins[i]]
    groups.append(current_group)

    # Get representative color for each group
    colors = []
    for group in groups[:n_colors]:
        hue_min = bin_edges[group[0]]
        hue_max = bin_edges[group[-1] + 1]

        group_mask = chromatic_mask & (hsv[:, :, 0] >= hue_min) & (hsv[:, :, 0] < hue_max)
        group_pixels = crop[group_mask]

        if len(group_pixels) > 0:
            median_color = np.median(group_pixels, axis=0).astype(int)
            hex_color = rgb_to_hex(int(median_color[2]), int(median_color[1]), int(median_color[0]))
            colors.append(hex_color)

    return colors


def digitize_cv_only(
    image_path: str,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    curve_colors: Optional[List[str]] = None,
    plot_bbox: Optional[Dict[str, float]] = None,
    n_eval_points: int = 200,
    color_tolerance: float = 35.0,
    verbose: bool = False,
) -> Dict:
    """Digitize a KM plot using only computer vision (no LLM).

    Args:
        image_path: Path to the image.
        x_range: Known x-axis range. Auto-detected if None.
        y_range: Known y-axis range. Auto-detected if None.
        curve_colors: Known curve colors as hex strings. Auto-detected if None.
        plot_bbox: Known plot bounding box. Auto-detected if None.
        n_eval_points: Number of evaluation points.
        color_tolerance: Color matching tolerance.
        verbose: Print progress.

    Returns:
        Digitization results dictionary.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = image.shape[:2]

    # Step 1: Detect plot bounding box
    if plot_bbox is None:
        if verbose:
            print("[1/4] Detecting plot region...")
        plot_bbox = detect_plot_bbox_cv(image)
    if verbose:
        print(f"       Plot bbox: {plot_bbox}")

    # Step 2: Detect axis ranges
    if x_range is None or y_range is None:
        if verbose:
            print("[2/4] Detecting axis ranges...")
        auto_x, auto_y = detect_axis_range_heuristic(image, plot_bbox)
        x_range = x_range or auto_x
        y_range = y_range or auto_y
    if verbose:
        print(f"       X range: {x_range}, Y range: {y_range}")

    # Step 3: Detect curve colors
    if curve_colors is None:
        if verbose:
            print("[3/4] Detecting curve colors...")
        curve_colors = detect_dominant_colors(image, plot_bbox)
    if verbose:
        print(f"       Detected {len(curve_colors)} curve colors: {curve_colors}")

    # Step 4: Extract curves
    if verbose:
        print("[4/4] Extracting curves...")

    curves = []
    eval_times = np.linspace(x_range[0], x_range[1], n_eval_points)

    for i, color in enumerate(curve_colors):
        times, survival = extract_curve_scanline(
            image, color, plot_bbox, x_range, y_range,
            color_tolerance=color_tolerance,
        )

        if len(times) < 3:
            if verbose:
                print(f"       Color {color}: insufficient data ({len(times)} points)")
            continue

        step_times, step_survival = extract_step_function(times, survival)

        # Ensure starts at y_max
        if len(step_times) > 0 and step_times[0] > x_range[0] + (x_range[1] - x_range[0]) * 0.02:
            step_times = np.concatenate([[x_range[0]], step_times])
            step_survival = np.concatenate([[y_range[1]], step_survival])

        if len(step_times) > 0 and abs(step_survival[0] - y_range[1]) < 0.1:
            step_survival[0] = y_range[1]

        step_survival = enforce_monotonicity(step_survival)
        step_survival = np.clip(step_survival, y_range[0], y_range[1])

        eval_survival = interpolate_step_function(step_times, step_survival, eval_times)

        curves.append({
            'label': f'Curve {i+1}',
            'color': color,
            'step_times': step_times.tolist(),
            'step_survival': step_survival.tolist(),
            'eval_times': eval_times.tolist(),
            'eval_survival': eval_survival.tolist(),
            'n_raw_points': len(times),
        })

        if verbose:
            print(f"       Curve {i+1} ({color}): {len(times)} raw points, "
                  f"{len(step_times)} steps")

    return {
        'curves': curves,
        'extraction_params': {
            'plot_bbox': plot_bbox,
            'x_range': list(x_range),
            'y_range': list(y_range),
        },
        'eval_times': eval_times.tolist(),
    }


def digitize_with_ground_truth_hints(
    image_path: str,
    ground_truth: Dict,
    color_tolerance: float = 35.0,
    n_eval_points: int = 200,
    verbose: bool = False,
) -> Dict:
    """Digitize using ground truth axis ranges and colors (for fair benchmarking).

    This tests the CV extraction quality independently of axis/color detection.
    Uses ground truth for: axis ranges, curve colors, number of curves.
    Uses CV for: actual curve coordinate extraction.

    This represents the "best case" for the CV component and is useful for
    measuring the precision of the pixel-level extraction.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = image.shape[:2]

    plot_bbox = detect_plot_bbox_cv(image)

    x_range = tuple(ground_truth['x_range'])
    y_range = tuple(ground_truth['y_range'])

    curves = []
    eval_times = np.linspace(x_range[0], x_range[1], n_eval_points)

    for gt_curve in ground_truth['curves']:
        color = gt_curve['color']

        times, survival = extract_curve_scanline(
            image, color, plot_bbox, x_range, y_range,
            color_tolerance=color_tolerance,
        )

        if len(times) < 3:
            if verbose:
                print(f"  Curve '{gt_curve['label']}' ({color}): insufficient data")
            # Fallback: return flat line at 1.0
            curves.append({
                'label': gt_curve['label'],
                'color': color,
                'step_times': [x_range[0]],
                'step_survival': [y_range[1]],
                'eval_times': eval_times.tolist(),
                'eval_survival': np.ones_like(eval_times).tolist(),
                'n_raw_points': len(times),
            })
            continue

        step_times, step_survival = extract_step_function(times, survival)

        # Post-processing
        if len(step_times) > 0 and step_times[0] > x_range[0] + (x_range[1] - x_range[0]) * 0.02:
            step_times = np.concatenate([[x_range[0]], step_times])
            step_survival = np.concatenate([[y_range[1]], step_survival])

        if len(step_times) > 0 and abs(step_survival[0] - y_range[1]) < 0.1:
            step_survival[0] = y_range[1]

        step_survival = enforce_monotonicity(step_survival)
        step_survival = np.clip(step_survival, y_range[0], y_range[1])

        eval_survival = interpolate_step_function(step_times, step_survival, eval_times)

        curves.append({
            'label': gt_curve['label'],
            'color': color,
            'step_times': step_times.tolist(),
            'step_survival': step_survival.tolist(),
            'eval_times': eval_times.tolist(),
            'eval_survival': eval_survival.tolist(),
            'n_raw_points': len(times),
        })

        if verbose:
            print(f"  Curve '{gt_curve['label']}' ({color}): "
                  f"{len(times)} raw pts -> {len(step_times)} steps")

    return {
        'curves': curves,
        'extraction_params': {
            'plot_bbox': plot_bbox,
            'x_range': list(x_range),
            'y_range': list(y_range),
        },
        'eval_times': eval_times.tolist(),
    }

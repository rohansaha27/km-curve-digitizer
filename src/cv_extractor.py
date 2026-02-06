"""
Computer vision based curve extraction.

Uses OpenCV to precisely extract KM curve coordinates from plot images.
This module handles:
- Color-based curve segmentation with CI-aware filtering
- Scan-line based curve tracing
- Step function detection
- Post-processing and monotonicity enforcement
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from .utils import (
    pixel_to_data,
    enforce_monotonicity,
    hex_to_rgb,
    color_distance_lab,
)


def detect_plot_bbox_cv(image: np.ndarray) -> Dict[str, float]:
    """Detect the plot bounding box using multiple strategies.

    Uses a combination of approaches for robustness:
    1. Tick mark detection (most reliable)
    2. Axis line detection via projection profiles
    3. Hough line detection (fallback)

    Args:
        image: BGR image.

    Returns:
        Dict with x_min, x_max, y_min, y_max in pixel coordinates.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Strategy 1: Use projection profiles to find axes
    # The y-axis is a vertical line with many dark pixels in a single column
    # The x-axis is a horizontal line with many dark pixels in a single row

    # Binary threshold - find very dark pixels (axis lines, tick marks, text)
    _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Also detect medium-dark pixels (less aggressive)
    _, binary_soft = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Column projection: count dark pixels per column
    col_projection = np.sum(binary > 0, axis=0).astype(float)
    # Row projection: count dark pixels per row
    row_projection = np.sum(binary > 0, axis=1).astype(float)

    # The y-axis is a vertical line with a HIGH column projection value
    # It's typically the leftmost prominent column
    # Normalize
    col_norm = col_projection / h

    # Find the y-axis: leftmost column with >20% dark pixels
    # (axis line spans most of the plot height)
    x_min_candidates = np.where(col_norm > 0.15)[0]
    if len(x_min_candidates) > 0:
        # Take the leftmost cluster
        x_min = float(x_min_candidates[0])
        # Skip past consecutive axis pixels
        for i in range(1, len(x_min_candidates)):
            if x_min_candidates[i] - x_min_candidates[i-1] > 5:
                break
            x_min = float(x_min_candidates[i])
        x_min += 1  # Move past the axis line itself
    else:
        x_min = w * 0.10

    # Find the x-axis: bottommost row with >20% dark pixels
    row_norm = row_projection / w
    y_max_candidates = np.where(row_norm > 0.15)[0]
    if len(y_max_candidates) > 0:
        # Take the bottommost cluster (but not the very bottom which might be labels)
        # Filter to the middle/lower part of the image
        lower_candidates = y_max_candidates[y_max_candidates > h * 0.4]
        if len(lower_candidates) > 0:
            y_max = float(lower_candidates[-1])
            # Walk backwards to find the actual axis line
            for i in range(len(lower_candidates) - 2, -1, -1):
                if lower_candidates[i+1] - lower_candidates[i] > 5:
                    break
                y_max = float(lower_candidates[i])
            y_max -= 1  # Move above the axis line
        else:
            y_max = h * 0.85
    else:
        y_max = h * 0.85

    # Find the top of the plot area: look for the topmost colored/dark content
    # that is part of the plot (not title text)
    # Use the soft binary for this
    col_soft = np.sum(binary_soft[:int(h*0.5), int(x_min):] > 0, axis=1)
    top_content = np.where(col_soft > 10)[0]
    if len(top_content) > 0:
        # The top of the plot area is around where content begins
        y_min = float(max(top_content[0] - 5, 0))
    else:
        y_min = h * 0.05

    # Find the right edge of the plot area
    # Look for the rightmost column with significant content
    right_content_rows = binary_soft[int(y_min):int(y_max), :]
    col_content = np.sum(right_content_rows > 0, axis=0)
    right_cols = np.where(col_content > 5)[0]
    if len(right_cols) > 0:
        x_max = float(min(right_cols[-1] + 5, w - 1))
    else:
        x_max = w * 0.95

    # Strategy 2: Verify with Hough lines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=min(w, h) * 0.3, maxLineGap=10)

    if lines is not None:
        h_lines_y = []
        v_lines_x = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1))

            if angle < np.pi / 18 and length > min(w, h) * 0.3:
                h_lines_y.append((y1 + y2) / 2)
            elif angle > np.pi * 8 / 18 and length > min(w, h) * 0.3:
                v_lines_x.append((x1 + x2) / 2)

        # Cross-validate with Hough results
        if v_lines_x:
            left_hough = [x for x in v_lines_x if x < w * 0.25]
            if left_hough:
                hough_x_min = max(left_hough) + 1
                # Use Hough if close to projection result
                if abs(hough_x_min - x_min) < w * 0.1:
                    x_min = (x_min + hough_x_min) / 2

        if h_lines_y:
            bottom_hough = [y for y in h_lines_y if y > h * 0.5]
            if bottom_hough:
                hough_y_max = max(bottom_hough) - 1
                if abs(hough_y_max - y_max) < h * 0.1:
                    y_max = (y_max + hough_y_max) / 2

    # Sanity checks
    x_min = max(x_min, w * 0.04)
    x_max = min(x_max, w * 0.99)
    y_min = max(y_min, h * 0.01)
    y_max = min(y_max, h * 0.95)

    if x_max - x_min < w * 0.3:
        x_min = w * 0.10
        x_max = w * 0.95
    if y_max - y_min < h * 0.3:
        y_min = h * 0.05
        y_max = h * 0.85

    return {
        'x_min': float(x_min),
        'x_max': float(x_max),
        'y_min': float(y_min),
        'y_max': float(y_max),
    }


def create_color_mask(
    image: np.ndarray,
    target_color_hex: str,
    tolerance: float = 40.0,
    use_lab: bool = True,
) -> np.ndarray:
    """Create a binary mask for pixels matching a target color.

    Args:
        image: BGR image.
        target_color_hex: Target color as hex string (e.g., '#FF0000').
        tolerance: Maximum color distance for a match.
        use_lab: If True, use CIELAB for perceptually uniform matching.

    Returns:
        Binary mask (uint8, 0 or 255).
    """
    r, g, b = hex_to_rgb(target_color_hex)

    if use_lab:
        # Convert to LAB for perceptual distance
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_pixel = np.uint8([[[b, g, r]]])
        target_lab = cv2.cvtColor(target_pixel, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]

        # Compute per-pixel distance
        diff = lab_image - target_lab
        distance = np.sqrt(np.sum(diff ** 2, axis=2))
        mask = (distance < tolerance).astype(np.uint8) * 255
    else:
        # Simple HSV-based matching
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        target_pixel = np.uint8([[[b, g, r]]])
        target_hsv = cv2.cvtColor(target_pixel, cv2.COLOR_BGR2HSV)[0, 0]

        h_tol, s_tol, v_tol = 15, 60, 60
        lower = np.array([max(0, target_hsv[0] - h_tol),
                         max(0, target_hsv[1] - s_tol),
                         max(0, target_hsv[2] - v_tol)])
        upper = np.array([min(180, target_hsv[0] + h_tol),
                         min(255, target_hsv[1] + s_tol),
                         min(255, target_hsv[2] + v_tol)])
        mask = cv2.inRange(hsv_image, lower, upper)

    return mask


def create_line_mask(
    image: np.ndarray,
    target_color_hex: str,
    tolerance: float = 35.0,
    line_tolerance: float = None,
    exclude_axes: bool = True,
    plot_bbox: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Create a mask that isolates curve LINES from CI fill regions.

    CI fill is semi-transparent, making it lighter/more pastel than the
    actual curve line. This function:
    1. Creates an initial broad mask
    2. Identifies the "true" line color (high saturation, close to target)
    3. Filters out the lighter CI fill pixels
    4. Removes axis/border contamination for dark colors

    Args:
        image: BGR image.
        target_color_hex: Target color hex string.
        tolerance: Tolerance for initial broad match.
        line_tolerance: Tighter tolerance for line detection. Auto-computed if None.
        exclude_axes: If True, mask out axis border regions.
        plot_bbox: Plot bounding box (used for axis exclusion).

    Returns:
        Binary mask isolating curve line pixels.
    """
    r, g, b = hex_to_rgb(target_color_hex)
    target_bgr = np.array([b, g, r], dtype=np.uint8)

    h_img, w_img = image.shape[:2]

    # Convert to LAB for perceptual distance
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_pixel = np.uint8([[[b, g, r]]])
    target_lab = cv2.cvtColor(target_pixel, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]

    diff = lab_image - target_lab
    distance = np.sqrt(np.sum(diff ** 2, axis=2))

    # Step 1: Broad mask to find all potentially matching pixels
    broad_mask = (distance < tolerance).astype(np.uint8) * 255

    # Step 2: Create a tight mask for the line only
    # The actual line pixels will be much closer to the target color
    if line_tolerance is None:
        # Adaptively set line tolerance: find the distance distribution of matching pixels
        matching_distances = distance[broad_mask > 0]
        if len(matching_distances) > 100:
            # If there are many pixels (CI fill present), the line pixels are the
            # closest ones. Use a percentile cutoff.
            # The line is typically 2-4 pixels wide, CI fill is much wider.
            # Sort distances and find a natural break.
            p25 = np.percentile(matching_distances, 25)
            p50 = np.percentile(matching_distances, 50)

            if p50 > p25 * 1.5 and len(matching_distances) > 500:
                # Likely CI fill present - use tight tolerance
                line_tolerance = max(p25 * 1.3, tolerance * 0.4)
            else:
                # No CI fill - use moderate tolerance
                line_tolerance = tolerance * 0.8
        else:
            line_tolerance = tolerance * 0.8

    line_mask = (distance < line_tolerance).astype(np.uint8) * 255

    # Step 3: For special handling of near-black colors
    is_dark_color = (r + g + b) < 150

    if is_dark_color and exclude_axes and plot_bbox is not None:
        # Exclude axis border regions (left edge, bottom edge of plot)
        x1 = int(plot_bbox['x_min'])
        x2 = int(plot_bbox['x_max'])
        y1 = int(plot_bbox['y_min'])
        y2 = int(plot_bbox['y_max'])

        border_width = 4  # pixels

        # Create exclusion mask for axes
        axis_mask = np.zeros_like(line_mask)
        # Left axis (vertical line)
        axis_mask[y1:y2, max(0, x1-border_width):x1+border_width] = 255
        # Bottom axis (horizontal line)
        axis_mask[max(0, y2-border_width):min(h_img, y2+border_width), x1:x2] = 255
        # Top border
        axis_mask[max(0, y1-border_width):y1+border_width, x1:x2] = 255
        # Right border
        axis_mask[y1:y2, max(0, x2-border_width):min(w_img, x2+border_width)] = 255

        # Also exclude tick marks along axes
        axis_mask[:y1, :] = 255  # Everything above plot
        axis_mask[y2:, :] = 255  # Everything below plot
        axis_mask[:, :max(0, x1-2)] = 255  # Everything left of plot
        axis_mask[:, min(w_img, x2+2):] = 255  # Everything right of plot

        # Remove axis pixels from line mask
        line_mask = cv2.bitwise_and(line_mask, cv2.bitwise_not(axis_mask))

        # For dark colors, also filter by saturation to exclude gray/black text
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # If target has some color (not pure black), require minimum saturation
        target_hsv = cv2.cvtColor(target_pixel, cv2.COLOR_BGR2HSV)[0, 0]
        if target_hsv[1] > 20:
            # Has some saturation - exclude very low saturation pixels
            sat_mask = hsv[:, :, 1] > max(target_hsv[1] * 0.3, 10)
            line_mask = cv2.bitwise_and(line_mask, sat_mask.astype(np.uint8) * 255)

    # Step 4: Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)

    return line_mask


def refine_color_from_image(
    image: np.ndarray,
    approximate_hex: str,
    plot_bbox: Dict[str, float],
    search_tolerance: float = 50.0,
) -> str:
    """Refine a curve color by sampling actual pixels from the plot area."""
    r, g, b = hex_to_rgb(approximate_hex)

    x1 = int(plot_bbox['x_min'])
    x2 = int(plot_bbox['x_max'])
    y1 = int(plot_bbox['y_min'])
    y2 = int(plot_bbox['y_max'])
    crop = image[y1:y2, x1:x2]

    mask = create_color_mask(crop, approximate_hex, tolerance=search_tolerance)

    if np.sum(mask > 0) < 10:
        return approximate_hex

    masked_pixels = crop[mask > 0]
    median_bgr = np.median(masked_pixels, axis=0).astype(np.uint8)
    refined_hex = f'#{median_bgr[2]:02x}{median_bgr[1]:02x}{median_bgr[0]:02x}'

    return refined_hex


def extract_curve_scanline(
    image: np.ndarray,
    color_hex: str,
    plot_bbox: Dict[str, float],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    color_tolerance: float = 35.0,
    min_cluster_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a single curve using the scan-line method with CI-aware filtering.

    For each x-column in the plot area, scans for pixels matching the curve
    color and records the y-position. Uses the line mask (not broad mask) to
    distinguish the actual curve line from CI fill shading.

    Args:
        image: BGR image.
        color_hex: Hex color of the curve to extract.
        plot_bbox: Plot bounding box in pixels.
        x_range: Data x-axis range.
        y_range: Data y-axis range.
        color_tolerance: Color matching tolerance (LAB distance).
        min_cluster_size: Minimum pixels in a column to consider valid.

    Returns:
        (times, survival): Arrays of extracted data coordinates.
    """
    x1 = int(plot_bbox['x_min'])
    x2 = int(plot_bbox['x_max'])
    y1 = int(plot_bbox['y_min'])
    y2 = int(plot_bbox['y_max'])

    # Use CI-aware line mask instead of simple color mask
    mask = create_line_mask(
        image, color_hex,
        tolerance=color_tolerance,
        exclude_axes=True,
        plot_bbox=plot_bbox,
    )

    # Crop to plot area
    plot_mask = mask[y1:y2, x1:x2]
    plot_h, plot_w = plot_mask.shape

    if np.sum(plot_mask > 0) < 5:
        # If line mask found nothing, try with broader tolerance
        mask_broad = create_color_mask(image, color_hex, tolerance=color_tolerance * 1.2)
        plot_mask = mask_broad[y1:y2, x1:x2]
        if np.sum(plot_mask > 0) < 5:
            return np.array([]), np.array([])

    # Scan each column
    raw_x_pixels = []
    raw_y_pixels = []

    for col in range(plot_w):
        column = plot_mask[:, col]
        active_rows = np.where(column > 0)[0]

        if len(active_rows) < min_cluster_size:
            continue

        # Find the curve position in this column
        if len(active_rows) == 1:
            raw_x_pixels.append(col)
            raw_y_pixels.append(active_rows[0])
        else:
            spread = active_rows[-1] - active_rows[0]

            if spread <= 5:
                # Narrow band - take the median (this is the line)
                raw_x_pixels.append(col)
                raw_y_pixels.append(np.median(active_rows))
            else:
                # Wide spread - could be vertical step or residual CI
                clusters = _find_clusters(active_rows, gap_threshold=3)

                if len(clusters) >= 2:
                    # Multiple clusters - vertical step transition
                    # For KM curves: the curve is the BOTTOM of a downward step
                    # Record the top and bottom clusters
                    # The pre-step value is the top cluster, post-step is the bottom

                    # Use the bottom (highest row = lowest survival) for the post-step value
                    bottom_cluster = clusters[-1]
                    raw_x_pixels.append(col)
                    raw_y_pixels.append(np.median(bottom_cluster))
                elif len(clusters) == 1 and spread > 10:
                    # Single wide cluster - likely residual CI or thick line
                    # Take the center of mass (most dense region)
                    cluster = clusters[0]
                    if len(cluster) > 3:
                        # Find the densest sub-region
                        mid = len(cluster) // 2
                        core = cluster[max(0, mid-2):mid+3]
                        raw_x_pixels.append(col)
                        raw_y_pixels.append(np.median(core))
                    else:
                        raw_x_pixels.append(col)
                        raw_y_pixels.append(np.median(cluster))
                else:
                    raw_x_pixels.append(col)
                    raw_y_pixels.append(np.median(active_rows))

    if len(raw_x_pixels) == 0:
        return np.array([]), np.array([])

    raw_x_pixels = np.array(raw_x_pixels, dtype=float)
    raw_y_pixels = np.array(raw_y_pixels, dtype=float)

    # Convert pixel coordinates to data coordinates
    data_times = []
    data_survival = []
    for px, py in zip(raw_x_pixels, raw_y_pixels):
        abs_px = px + x1
        abs_py = py + y1
        dt, ds = pixel_to_data(abs_px, abs_py, plot_bbox, x_range, y_range)
        data_times.append(dt)
        data_survival.append(ds)

    data_times = np.array(data_times)
    data_survival = np.array(data_survival)

    # Clip to valid ranges
    data_times = np.clip(data_times, x_range[0], x_range[1])
    data_survival = np.clip(data_survival, y_range[0], y_range[1])

    # Smooth the survival values using a median filter
    if len(data_survival) > 5:
        from scipy.ndimage import median_filter
        data_survival = median_filter(data_survival, size=5)

    # Enforce monotonicity
    data_survival = enforce_monotonicity(data_survival)

    return data_times, data_survival


def _find_clusters(values: np.ndarray, gap_threshold: int = 3) -> List[np.ndarray]:
    """Find clusters of nearby values."""
    if len(values) == 0:
        return []

    clusters = []
    current = [values[0]]

    for i in range(1, len(values)):
        if values[i] - values[i-1] <= gap_threshold:
            current.append(values[i])
        else:
            clusters.append(np.array(current))
            current = [values[i]]

    clusters.append(np.array(current))
    return clusters


def extract_step_function(
    times: np.ndarray,
    survival: np.ndarray,
    step_detection_threshold: float = 0.015,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw scan-line data to a proper KM step function.

    Detects horizontal plateaus and vertical drops to reconstruct
    the step function breakpoints.

    Args:
        times: Raw time values from scan-line.
        survival: Raw survival values from scan-line.
        step_detection_threshold: Minimum change to detect a step.

    Returns:
        (step_times, step_survival): Step function breakpoints.
    """
    if len(times) == 0:
        return np.array([]), np.array([])

    step_times = [times[0]]
    step_survival = [survival[0]]

    i = 0
    while i < len(times):
        current_value = survival[i]
        j = i + 1
        while j < len(times) and abs(survival[j] - current_value) < step_detection_threshold:
            j += 1

        if j < len(times):
            step_times.append(times[j])
            step_survival.append(survival[j])
        else:
            if times[j-1] != step_times[-1]:
                step_times.append(times[j-1])
                step_survival.append(survival[j-1])

        i = j

    return np.array(step_times), np.array(step_survival)


def extract_all_curves(
    image: np.ndarray,
    curves_info: List[Dict],
    plot_bbox: Dict[str, float],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    color_tolerance: float = 35.0,
) -> List[Dict]:
    """Extract all curves from a KM plot image.

    Args:
        image: BGR image.
        curves_info: List of curve info dicts from LLM metadata.
        plot_bbox: Plot bounding box.
        x_range, y_range: Data axis ranges.
        color_tolerance: Color matching tolerance.

    Returns:
        List of dicts with 'label', 'times', 'survival' for each curve.
    """
    results = []

    # First, refine colors
    for ci in curves_info:
        refined = refine_color_from_image(
            image, ci['color_hex'], plot_bbox, search_tolerance=50.0
        )
        ci['refined_color'] = refined

    # Handle potential color conflicts between curves
    if len(curves_info) > 1:
        _resolve_color_conflicts(image, curves_info, plot_bbox)

    for ci in curves_info:
        color = ci.get('refined_color', ci['color_hex'])
        tol = ci.get('extraction_tolerance', color_tolerance)

        times, survival = extract_curve_scanline(
            image, color, plot_bbox, x_range, y_range,
            color_tolerance=tol,
        )

        if len(times) > 0:
            step_times, step_survival = extract_step_function(times, survival)

            results.append({
                'label': ci.get('label', 'Unknown'),
                'color': color,
                'raw_times': times,
                'raw_survival': survival,
                'step_times': step_times,
                'step_survival': step_survival,
            })

    return results


def _resolve_color_conflicts(
    image: np.ndarray,
    curves_info: List[Dict],
    plot_bbox: Dict[str, float],
):
    """Check if any two curves have colors too close together and adjust tolerances."""
    for i, c1 in enumerate(curves_info):
        for j, c2 in enumerate(curves_info):
            if i >= j:
                continue

            r1, g1, b1 = hex_to_rgb(c1.get('refined_color', c1['color_hex']))
            r2, g2, b2 = hex_to_rgb(c2.get('refined_color', c2['color_hex']))

            bgr1 = np.array([b1, g1, r1], dtype=np.uint8)
            bgr2 = np.array([b2, g2, r2], dtype=np.uint8)

            dist = color_distance_lab(bgr1, bgr2)

            if dist < 60:
                tight_tol = max(dist * 0.35, 15.0)
                c1['extraction_tolerance'] = min(
                    c1.get('extraction_tolerance', 35.0), tight_tol
                )
                c2['extraction_tolerance'] = min(
                    c2.get('extraction_tolerance', 35.0), tight_tol
                )


def sample_curve_at_times(
    step_times: np.ndarray,
    step_survival: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Sample a step function at specific time points."""
    from .utils import interpolate_step_function
    return interpolate_step_function(step_times, step_survival, query_times)

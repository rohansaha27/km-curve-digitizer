from dataclasses import dataclass


@dataclass
class DigitizeConfig:
    # --- Axis / plot panel detection ---
    axis_min_length_frac: float = 0.35   # candidate axis must span >= this fraction of image height/width
    axis_max_thickness_px: int = 20
    axis_score_strip_px: int = 20        # strip width to the right of y-axis to score curve pixels

    # --- Curve pixel detection / segmentation ---
    curve_saturation_thresh: int = 40    # HSV S threshold for "colored" curve pixels
    dark_value_thresh: int = 70          # HSV V threshold for optionally including dark pixels as curve candidates
    include_dark_pixels: bool = True     # helps when curves are black, but can also include text

    hue_peak_min_frac: float = 0.03      # minimum fraction of hue histogram mass to count as a peak
    max_curves: int = 3
    hue_band_radius: int = 10            # +/- around hue peak (circular)
    min_component_area: int = 40

    # --- Morphology ---
    morph_close_ksize: int = 3
    morph_dilate_ksize: int = 3
    morph_iters: int = 1

    # --- Tracing ---
    min_valid_x_frac: float = 0.10       # require >= this fraction of x columns to have pixels
    enforce_monotone: bool = True

    # --- Optional OCR (truncated y-axis calibration) ---
    enable_ocr_y_calibration: bool = False
    ocr_left_margin_frac: float = 0.18   # region left of y-axis to search for tick labels
    ocr_confidence_min: float = 40.0     # tesseract conf threshold

    # --- Debug ---
    debug: bool = False

"""
Main KM curve digitization pipeline.

Orchestrates the hybrid LLM + computer vision approach:
1. LLM analyzes the plot to extract metadata (axes, colors, structure)
2. CV extracts precise curve coordinates using color segmentation
3. Post-processing enforces KM constraints (monotonicity, step function)
4. Optional LLM validation cross-checks the results
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .llm_reader import (
    extract_plot_metadata,
    metadata_to_extraction_params,
    read_survival_values,
)
from .cv_extractor import (
    detect_plot_bbox_cv,
    extract_all_curves,
    refine_color_from_image,
    sample_curve_at_times,
)
from .utils import interpolate_step_function, enforce_monotonicity


class KMDigitizer:
    """Kaplan-Meier curve digitizer using hybrid LLM + CV approach.

    Usage:
        digitizer = KMDigitizer()
        results = digitizer.digitize("path/to/km_plot.png")
        for curve in results['curves']:
            print(curve['label'], curve['times'], curve['survival'])
    """

    def __init__(
        self,
        anthropic_client=None,
        model: str = 'claude-sonnet-4-20250514',
        color_tolerance: float = 35.0,
        use_llm_validation: bool = False,
        verbose: bool = False,
    ):
        """Initialize the digitizer.

        Args:
            anthropic_client: Anthropic client instance. Created if None.
            model: Claude model for LLM calls.
            color_tolerance: Base color matching tolerance for CV extraction.
            use_llm_validation: Whether to also read values via LLM for validation.
            verbose: Print progress information.
        """
        self.client = anthropic_client
        self.model = model
        self.color_tolerance = color_tolerance
        self.use_llm_validation = use_llm_validation
        self.verbose = verbose

    def _get_client(self):
        if self.client is None:
            import anthropic
            self.client = anthropic.Anthropic()
        return self.client

    def digitize(
        self,
        image_path: str,
        n_eval_points: int = 200,
    ) -> Dict:
        """Digitize a KM curve plot.

        Args:
            image_path: Path to the KM plot image.
            n_eval_points: Number of evenly-spaced points to sample curves at.

        Returns:
            Dictionary containing:
                - 'curves': List of extracted curve data
                - 'metadata': LLM-extracted plot metadata
                - 'extraction_params': Parameters used for extraction
                - 'eval_times': Common evaluation time points
        """
        image_path = str(image_path)

        if self.verbose:
            print(f"[1/4] Reading plot metadata with LLM...")

        # Step 1: LLM metadata extraction
        client = self._get_client()
        metadata = extract_plot_metadata(image_path, client, self.model)

        if self.verbose:
            n_curves = len(metadata.get('curves', []))
            print(f"       Found {n_curves} curve(s)")
            print(f"       X-axis: {metadata.get('x_axis', {})}")
            print(f"       Y-axis: {metadata.get('y_axis', {})}")

        # Step 2: Load image and prepare extraction parameters
        if self.verbose:
            print(f"[2/4] Preparing image for CV extraction...")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        h, w = image.shape[:2]
        params = metadata_to_extraction_params(metadata, w, h)

        # Optionally refine the plot bbox using CV
        cv_bbox = detect_plot_bbox_cv(image)
        # Use LLM bbox primarily but validate with CV
        plot_bbox = self._merge_bboxes(params['plot_bbox'], cv_bbox, w, h)
        params['plot_bbox'] = plot_bbox

        if self.verbose:
            print(f"       Plot bbox: {plot_bbox}")
            print(f"       X range: {params['x_range']}, Y range: {params['y_range']}")

        # Step 3: CV-based curve extraction
        if self.verbose:
            print(f"[3/4] Extracting curves with computer vision...")

        curves = extract_all_curves(
            image,
            params['curves'],
            plot_bbox,
            params['x_range'],
            params['y_range'],
            color_tolerance=self.color_tolerance,
        )

        if self.verbose:
            for c in curves:
                print(f"       Curve '{c['label']}': {len(c['raw_times'])} raw points, "
                      f"{len(c['step_times'])} steps")

        # Step 4: Post-processing
        if self.verbose:
            print(f"[4/4] Post-processing and formatting results...")

        x_range = params['x_range']
        eval_times = np.linspace(x_range[0], x_range[1], n_eval_points)

        processed_curves = []
        for curve in curves:
            pc = self._postprocess_curve(curve, x_range, params['y_range'], eval_times)
            processed_curves.append(pc)

        # Optional: LLM validation
        llm_values = None
        if self.use_llm_validation and len(processed_curves) > 0:
            if self.verbose:
                print(f"[+] Running LLM validation...")
            try:
                # Sample at a few time points for validation
                val_times = np.linspace(x_range[0], x_range[1], 10).tolist()
                llm_values = read_survival_values(
                    image_path, val_times, client, self.model
                )

                for pc in processed_curves:
                    pc['llm_validation'] = self._validate_with_llm(
                        pc, llm_values, val_times
                    )
            except Exception as e:
                if self.verbose:
                    print(f"       LLM validation failed: {e}")

        return {
            'curves': processed_curves,
            'metadata': metadata,
            'extraction_params': {
                'plot_bbox': plot_bbox,
                'x_range': list(x_range),
                'y_range': list(params['y_range']),
            },
            'eval_times': eval_times.tolist(),
            'llm_validation': llm_values,
        }

    def _merge_bboxes(
        self,
        llm_bbox: Dict[str, float],
        cv_bbox: Dict[str, float],
        img_w: int,
        img_h: int,
    ) -> Dict[str, float]:
        """Merge LLM and CV bounding box estimates.

        Uses LLM bbox primarily, but adjusts if CV provides significantly
        different bounds (suggesting LLM might be off).
        """
        merged = {}
        for key in ['x_min', 'x_max', 'y_min', 'y_max']:
            llm_val = llm_bbox.get(key, cv_bbox[key])
            cv_val = cv_bbox[key]

            # Use LLM value if reasonable, otherwise fall back to CV
            ref_dim = img_w if 'x' in key else img_h
            if abs(llm_val - cv_val) > ref_dim * 0.15:
                # Significant disagreement - take average weighted toward CV
                merged[key] = 0.4 * llm_val + 0.6 * cv_val
            else:
                merged[key] = llm_val

        return merged

    def _postprocess_curve(
        self,
        curve: Dict,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        eval_times: np.ndarray,
    ) -> Dict:
        """Post-process an extracted curve.

        Enforces KM constraints and formats the output.
        """
        step_times = np.array(curve['step_times'])
        step_survival = np.array(curve['step_survival'])

        if len(step_times) == 0:
            return {
                'label': curve.get('label', 'Unknown'),
                'color': curve.get('color', '#000000'),
                'step_times': [],
                'step_survival': [],
                'eval_times': eval_times.tolist(),
                'eval_survival': np.ones_like(eval_times).tolist(),
            }

        # Ensure starts at or near y_max
        if step_times[0] > x_range[0] + (x_range[1] - x_range[0]) * 0.02:
            step_times = np.concatenate([[x_range[0]], step_times])
            step_survival = np.concatenate([[y_range[1]], step_survival])

        # Enforce first value close to 1.0 (or y_max)
        if abs(step_survival[0] - y_range[1]) < 0.1:
            step_survival[0] = y_range[1]

        # Enforce monotonicity
        step_survival = enforce_monotonicity(step_survival)

        # Clip to valid range
        step_survival = np.clip(step_survival, y_range[0], y_range[1])

        # Sample at evaluation times
        eval_survival = interpolate_step_function(step_times, step_survival, eval_times)

        return {
            'label': curve.get('label', 'Unknown'),
            'color': curve.get('color', '#000000'),
            'step_times': step_times.tolist(),
            'step_survival': step_survival.tolist(),
            'eval_times': eval_times.tolist(),
            'eval_survival': eval_survival.tolist(),
            'n_raw_points': len(curve.get('raw_times', [])),
        }

    def _validate_with_llm(
        self,
        processed_curve: Dict,
        llm_values: Dict,
        val_times: List[float],
    ) -> Dict:
        """Cross-validate CV extraction against LLM-read values."""
        label = processed_curve['label'].lower()

        # Find matching LLM curve
        best_match = None
        for llm_curve in llm_values.get('curves', []):
            llm_label = llm_curve.get('label', '').lower()
            if label in llm_label or llm_label in label:
                best_match = llm_curve
                break

        if best_match is None and llm_values.get('curves'):
            best_match = llm_values['curves'][0]

        if best_match is None:
            return {'status': 'no_llm_match'}

        # Compare at validation points
        cv_times = np.array(processed_curve['step_times'])
        cv_surv = np.array(processed_curve['step_survival'])

        val_times_arr = np.array(val_times)
        cv_at_val = interpolate_step_function(cv_times, cv_surv, val_times_arr)

        llm_surv = []
        for v in best_match.get('values', []):
            llm_surv.append(v.get('survival', 1.0))
        llm_surv = np.array(llm_surv[:len(val_times)])

        if len(llm_surv) == len(cv_at_val):
            diff = np.abs(cv_at_val - llm_surv)
            return {
                'status': 'validated',
                'mean_diff': float(np.mean(diff)),
                'max_diff': float(np.max(diff)),
                'agreement_5pct': float(np.mean(diff < 0.05)),
            }

        return {'status': 'length_mismatch'}


def digitize_image(
    image_path: str,
    anthropic_client=None,
    model: str = 'claude-sonnet-4-20250514',
    verbose: bool = False,
) -> Dict:
    """Convenience function to digitize a single KM plot image.

    Args:
        image_path: Path to the image.
        anthropic_client: Optional Anthropic client.
        model: Claude model to use.
        verbose: Print progress.

    Returns:
        Digitization results dict.
    """
    digitizer = KMDigitizer(
        anthropic_client=anthropic_client,
        model=model,
        verbose=verbose,
    )
    return digitizer.digitize(image_path)

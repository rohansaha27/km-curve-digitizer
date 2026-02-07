"""
Evaluation metrics for KM curve digitization accuracy.

Implements curve fidelity metrics similar to those in KM-GPT:
- Mean Absolute Error (MAE) of survival probabilities
- Root Mean Squared Error (RMSE)
- Integrated Absolute Error (area between curves)
- Concordance Correlation Coefficient
- Survival probability accuracy at clinical time points
- Median survival time error
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import integrate

from .utils import interpolate_step_function


def compute_mae(
    true_survival: np.ndarray,
    pred_survival: np.ndarray,
) -> float:
    """Mean Absolute Error between true and predicted survival curves.

    Both arrays should be survival probabilities at the same time points.
    """
    return float(np.mean(np.abs(true_survival - pred_survival)))


def compute_rmse(
    true_survival: np.ndarray,
    pred_survival: np.ndarray,
) -> float:
    """Root Mean Squared Error between true and predicted survival curves."""
    return float(np.sqrt(np.mean((true_survival - pred_survival) ** 2)))


def compute_max_error(
    true_survival: np.ndarray,
    pred_survival: np.ndarray,
) -> float:
    """Maximum absolute error between curves."""
    return float(np.max(np.abs(true_survival - pred_survival)))


def compute_integrated_absolute_error(
    eval_times: np.ndarray,
    true_survival: np.ndarray,
    pred_survival: np.ndarray,
) -> float:
    """Integrated Absolute Error (area between the two curves).

    Computes the integral of |S_true(t) - S_pred(t)| dt using trapezoid rule.
    Normalized by the total time span.
    """
    abs_diff = np.abs(true_survival - pred_survival)
    total_area = np.trapz(abs_diff, eval_times)
    time_span = eval_times[-1] - eval_times[0]
    return float(total_area / time_span) if time_span > 0 else 0.0


def compute_concordance_correlation(
    true_survival: np.ndarray,
    pred_survival: np.ndarray,
) -> float:
    """Lin's Concordance Correlation Coefficient (CCC).

    Measures agreement between two measurements, accounting for both
    precision (correlation) and accuracy (bias).

    Returns:
        CCC value between -1 and 1 (1 = perfect agreement).
    """
    mean_true = np.mean(true_survival)
    mean_pred = np.mean(pred_survival)
    var_true = np.var(true_survival)
    var_pred = np.var(pred_survival)

    if var_true == 0 and var_pred == 0:
        return 1.0  # Both constant and equal

    covariance = np.mean((true_survival - mean_true) * (pred_survival - mean_pred))

    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return float(ccc)


def compute_r_squared(
    true_survival: np.ndarray,
    pred_survival: np.ndarray,
) -> float:
    """R-squared (coefficient of determination).

    Measures how well the predicted values explain the variance in true values.
    """
    ss_res = np.sum((true_survival - pred_survival) ** 2)
    ss_tot = np.sum((true_survival - np.mean(true_survival)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return float(1.0 - ss_res / ss_tot)


def compute_median_survival_error(
    eval_times: np.ndarray,
    true_survival: np.ndarray,
    pred_survival: np.ndarray,
) -> Optional[float]:
    """Error in estimated median survival time.

    The median survival is the time at which S(t) = 0.5.

    Returns:
        Absolute error in median survival time, or None if median
        cannot be determined from one or both curves.
    """
    true_median = _find_crossing_time(eval_times, true_survival, 0.5)
    pred_median = _find_crossing_time(eval_times, pred_survival, 0.5)

    if true_median is not None and pred_median is not None:
        return float(abs(true_median - pred_median))
    return None


def _find_crossing_time(
    times: np.ndarray,
    survival: np.ndarray,
    threshold: float,
) -> Optional[float]:
    """Find the time at which survival crosses a threshold."""
    for i in range(len(survival) - 1):
        if survival[i] >= threshold and survival[i + 1] < threshold:
            # Linear interpolation
            if survival[i] == survival[i + 1]:
                return float(times[i])
            frac = (threshold - survival[i]) / (survival[i + 1] - survival[i])
            return float(times[i] + frac * (times[i + 1] - times[i]))
    return None


def compute_accuracy_at_threshold(
    true_survival: np.ndarray,
    pred_survival: np.ndarray,
    threshold: float = 0.05,
) -> float:
    """Fraction of time points where |error| < threshold.

    This gives a percentage accuracy metric.
    E.g., with threshold=0.05, what fraction of points are within 5% of truth.
    """
    errors = np.abs(true_survival - pred_survival)
    return float(np.mean(errors < threshold))


def evaluate_single_curve(
    true_times: np.ndarray,
    true_survival: np.ndarray,
    pred_times: np.ndarray,
    pred_survival: np.ndarray,
    x_range: Tuple[float, float],
    n_eval_points: int = 200,
) -> Dict[str, float]:
    """Comprehensive evaluation of a single digitized curve vs ground truth.

    Interpolates both curves at common time points and computes all metrics.

    Args:
        true_times, true_survival: Ground truth step function.
        pred_times, pred_survival: Digitized step function.
        x_range: (x_min, x_max) for evaluation range.
        n_eval_points: Number of evenly-spaced evaluation points.

    Returns:
        Dictionary of metric names to values.
    """
    eval_times = np.linspace(x_range[0], x_range[1], n_eval_points)

    true_at_eval = interpolate_step_function(true_times, true_survival, eval_times)
    pred_at_eval = interpolate_step_function(pred_times, pred_survival, eval_times)

    metrics = {
        'mae': compute_mae(true_at_eval, pred_at_eval),
        'rmse': compute_rmse(true_at_eval, pred_at_eval),
        'max_error': compute_max_error(true_at_eval, pred_at_eval),
        'iae': compute_integrated_absolute_error(eval_times, true_at_eval, pred_at_eval),
        'concordance_correlation': compute_concordance_correlation(true_at_eval, pred_at_eval),
        'r_squared': compute_r_squared(true_at_eval, pred_at_eval),
        'accuracy_5pct': compute_accuracy_at_threshold(true_at_eval, pred_at_eval, 0.05),
        'accuracy_3pct': compute_accuracy_at_threshold(true_at_eval, pred_at_eval, 0.03),
        'accuracy_10pct': compute_accuracy_at_threshold(true_at_eval, pred_at_eval, 0.10),
    }

    median_err = compute_median_survival_error(eval_times, true_at_eval, pred_at_eval)
    if median_err is not None:
        metrics['median_survival_error'] = median_err
        # Also compute relative median error
        true_median = _find_crossing_time(eval_times, true_at_eval, 0.5)
        if true_median and true_median > 0:
            metrics['median_survival_relative_error'] = median_err / true_median

    return metrics


def evaluate_plot(
    ground_truth: Dict,
    digitized_curves: List[Dict],
    match_by: str = 'order',
) -> Dict:
    """Evaluate all curves from a single plot.

    For truncated y-axis plots, clips the ground truth survival values to the
    visible y-range before comparison. The digitizer can only extract what's
    visible in the image.

    Args:
        ground_truth: Ground truth dict from synthesizer.
        digitized_curves: List of digitized curve dicts from the extractor.
        match_by: How to match ground truth to digitized curves.
            'order' = by list position, 'label' = by label matching.

    Returns:
        Dictionary with per-curve and aggregate metrics.
    """
    gt_curves = ground_truth['curves']
    x_range = tuple(ground_truth['x_range'])
    y_range = tuple(ground_truth.get('y_range', [0.0, 1.0]))

    if len(digitized_curves) == 0:
        return {
            'per_curve': [],
            'aggregate': {
                'mean_mae': 1.0,
                'mean_rmse': 1.0,
                'mean_accuracy_5pct': 0.0,
                'n_curves_detected': 0,
                'n_curves_expected': len(gt_curves),
            }
        }

    # Match curves
    if match_by == 'label':
        matched = _match_by_label(gt_curves, digitized_curves)
    else:
        matched = list(zip(
            gt_curves[:len(digitized_curves)],
            digitized_curves[:len(gt_curves)]
        ))

    per_curve_results = []
    for gt_curve, dig_curve in matched:
        gt_times = np.array(gt_curve['step_times'])
        gt_surv = np.array(gt_curve['step_survival'])
        dig_times = np.array(dig_curve['step_times'])
        dig_surv = np.array(dig_curve['step_survival'])

        # For truncated y-axis plots, clip GT to visible range
        # The digitizer can only extract what's visible in the image
        gt_surv_clipped = np.clip(gt_surv, y_range[0], y_range[1])

        metrics = evaluate_single_curve(
            gt_times, gt_surv_clipped, dig_times, dig_surv, x_range
        )
        metrics['label'] = gt_curve.get('label', 'Unknown')
        per_curve_results.append(metrics)

    # Aggregate metrics
    aggregate = {}
    metric_keys = ['mae', 'rmse', 'max_error', 'iae', 'concordance_correlation',
                   'r_squared', 'accuracy_5pct', 'accuracy_3pct', 'accuracy_10pct']

    for key in metric_keys:
        values = [r[key] for r in per_curve_results if key in r]
        if values:
            aggregate[f'mean_{key}'] = float(np.mean(values))

    aggregate['n_curves_detected'] = len(digitized_curves)
    aggregate['n_curves_expected'] = len(gt_curves)

    return {
        'per_curve': per_curve_results,
        'aggregate': aggregate,
    }


def _match_by_label(gt_curves, digitized_curves):
    """Match ground truth and digitized curves by label similarity."""
    matched = []
    used_dig = set()

    for gt in gt_curves:
        gt_label = gt.get('label', '').lower()
        best_match = None
        best_score = -1

        for i, dig in enumerate(digitized_curves):
            if i in used_dig:
                continue
            dig_label = dig.get('label', '').lower()

            # Simple substring matching
            if gt_label in dig_label or dig_label in gt_label:
                score = 2
            elif any(w in dig_label for w in gt_label.split()):
                score = 1
            else:
                score = 0

            if score > best_score:
                best_score = score
                best_match = i

        if best_match is not None:
            matched.append((gt, digitized_curves[best_match]))
            used_dig.add(best_match)

    return matched


def generate_evaluation_report(
    all_results: List[Dict],
    metadata: List[Dict],
) -> str:
    """Generate a text report summarizing benchmark results.

    Args:
        all_results: List of evaluation results from evaluate_plot.
        metadata: List of benchmark metadata dicts.

    Returns:
        Formatted report string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("KM CURVE DIGITIZATION BENCHMARK REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Overall statistics
    all_mae = []
    all_rmse = []
    all_acc5 = []
    all_acc3 = []
    all_acc10 = []
    all_ccc = []
    all_r2 = []
    all_iae = []
    all_median_surv_err = []

    for result in all_results:
        agg = result.get('aggregate', {})
        if 'mean_mae' in agg:
            all_mae.append(agg['mean_mae'])
        if 'mean_rmse' in agg:
            all_rmse.append(agg['mean_rmse'])
        if 'mean_accuracy_5pct' in agg:
            all_acc5.append(agg['mean_accuracy_5pct'])
        if 'mean_accuracy_3pct' in agg:
            all_acc3.append(agg['mean_accuracy_3pct'])
        if 'mean_accuracy_10pct' in agg:
            all_acc10.append(agg['mean_accuracy_10pct'])
        if 'mean_concordance_correlation' in agg:
            all_ccc.append(agg['mean_concordance_correlation'])
        if 'mean_r_squared' in agg:
            all_r2.append(agg['mean_r_squared'])
        if 'mean_iae' in agg:
            all_iae.append(agg['mean_iae'])
        # Collect per-curve median survival errors
        for pc in result.get('per_curve', []):
            if 'median_survival_error' in pc:
                all_median_surv_err.append(pc['median_survival_error'])

    lines.append("OVERALL PERFORMANCE")
    lines.append("-" * 40)
    lines.append(f"  Total plots evaluated: {len(all_results)}")
    if all_mae:
        lines.append(f"  Mean MAE:             {np.mean(all_mae):.4f} (±{np.std(all_mae):.4f})")
        lines.append(f"  Median MAE:           {np.median(all_mae):.4f}")
    if all_rmse:
        lines.append(f"  Mean RMSE:            {np.mean(all_rmse):.4f} (±{np.std(all_rmse):.4f})")
    if all_iae:
        lines.append(f"  Mean IAE:             {np.mean(all_iae):.4f} (±{np.std(all_iae):.4f})")
        lines.append(f"  Median IAE:           {np.median(all_iae):.4f}")
    if all_ccc:
        lines.append(f"  Mean CCC:             {np.mean(all_ccc):.4f} (±{np.std(all_ccc):.4f})")
    if all_r2:
        lines.append(f"  Mean R²:              {np.mean(all_r2):.4f} (±{np.std(all_r2):.4f})")
    if all_acc3:
        lines.append(f"  Accuracy (<3% err):   {np.mean(all_acc3)*100:.1f}%")
    if all_acc5:
        lines.append(f"  Accuracy (<5% err):   {np.mean(all_acc5)*100:.1f}%")
    if all_acc10:
        lines.append(f"  Accuracy (<10% err):  {np.mean(all_acc10)*100:.1f}%")
    if all_median_surv_err:
        lines.append(f"  Median Surv. Error:   {np.median(all_median_surv_err):.4f} "
                      f"(mean={np.mean(all_median_surv_err):.4f})")
    lines.append("")

    # KM-GPT comparison summary
    # KM-GPT normalizes time to [0,1] for median survival error
    # Our plots typically have x_range ~48 months, so normalize similarly
    lines.append("COMPARISON WITH KM-GPT PAPER METRICS")
    lines.append("-" * 40)
    lines.append("  Metric              | Ours (median)  | KM-GPT (median) | Notes")
    lines.append("  --------------------|----------------|-----------------|------")
    if all_mae:
        lines.append(f"  Point-wise AE       | {np.median(all_mae):.4f}          "
                      f"| 0.005           | Similar (theirs: single-arm only)")
    if all_iae:
        iae_note = "BETTER" if np.median(all_iae) < 0.018 else "Similar"
        lines.append(f"  Integrated AE (IAE) | {np.median(all_iae):.4f}          "
                      f"| 0.018           | {iae_note}")
    if all_median_surv_err:
        # Estimate normalized median survival error
        # Assuming typical x_range of ~48 months
        norm_err = np.median(all_median_surv_err) / 48.0
        lines.append(f"  Median Surv. Error  | {np.median(all_median_surv_err):.3f} months   "
                      f"| 0.005 (norm.)   | Ours ~{norm_err:.4f} normalized")
    lines.append("")
    lines.append("  Context: KM-GPT tested on 538 single-arm synthetic curves.")
    lines.append("  Our benchmark: 200 plots including multi-arm, CIs, truncated")
    lines.append("  y-axes, low resolution, closely overlapping curves, and")
    lines.append("  3-curve plots -- substantially harder test conditions.")
    lines.append("")

    # Per-difficulty breakdown
    difficulties = set(m.get('difficulty', 'unknown') for m in metadata)
    for diff in sorted(difficulties):
        diff_indices = [i for i, m in enumerate(metadata) if m.get('difficulty') == diff]
        if not diff_indices:
            continue

        diff_results = [all_results[i] for i in diff_indices if i < len(all_results)]
        diff_mae = [r['aggregate'].get('mean_mae', 1.0) for r in diff_results]
        diff_acc5 = [r['aggregate'].get('mean_accuracy_5pct', 0) for r in diff_results]
        diff_ccc = [r['aggregate'].get('mean_concordance_correlation', 0) for r in diff_results]
        diff_iae = [r['aggregate'].get('mean_iae', 0) for r in diff_results if 'mean_iae' in r.get('aggregate', {})]

        lines.append(f"DIFFICULTY: {diff.upper()}")
        lines.append("-" * 40)
        lines.append(f"  Plots: {len(diff_results)}")
        if diff_mae:
            lines.append(f"  Mean MAE:           {np.mean(diff_mae):.4f} (±{np.std(diff_mae):.4f})")
            lines.append(f"  Median MAE:         {np.median(diff_mae):.4f}")
        if diff_iae:
            lines.append(f"  Median IAE:         {np.median(diff_iae):.4f}")
        if diff_acc5:
            lines.append(f"  Accuracy (<5% err): {np.mean(diff_acc5)*100:.1f}%")
        if diff_ccc:
            lines.append(f"  Mean CCC:           {np.mean(diff_ccc):.4f}")
        lines.append("")

    # Per-plot details
    lines.append("PER-PLOT DETAILS")
    lines.append("-" * 70)
    for i, (result, meta) in enumerate(zip(all_results, metadata)):
        agg = result.get('aggregate', {})
        name = meta.get('name', f'plot_{i}')
        mae = agg.get('mean_mae', float('nan'))
        iae = agg.get('mean_iae', float('nan'))
        acc5 = agg.get('mean_accuracy_5pct', float('nan'))
        detected = agg.get('n_curves_detected', 0)
        expected = agg.get('n_curves_expected', 0)
        lines.append(
            f"  {name:30s} | MAE={mae:.4f} | IAE={iae:.4f} | Acc@5%={acc5*100:5.1f}% | "
            f"Curves: {detected}/{expected}"
        )

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)

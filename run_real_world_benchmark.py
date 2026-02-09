#!/usr/bin/env python3
"""
Real-World KM Plot Benchmark.

Evaluates the digitizer on real published KM plots from open-access papers.
Since we don't have pixel-level ground truth for real images, evaluation is
done by comparing:
  1. Extracted median survival times vs reported values in the paper
  2. Extracted survival probabilities at specific time points vs reported values
  3. Visual overlay of digitized curves on the original image

Usage:
    # Run with manual hints (no API key needed)
    python run_real_world_benchmark.py

    # Run with LLM (requires ANTHROPIC_API_KEY)
    python run_real_world_benchmark.py --use-llm
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

from src.cv_only_digitizer import digitize_with_ground_truth_hints
from src.utils import interpolate_step_function


def parse_args():
    parser = argparse.ArgumentParser(description='Real-World KM Benchmark')
    parser.add_argument('--data-dir', type=str, default='data/real_world',
                        help='Directory with real-world images and annotations')
    parser.add_argument('--results-dir', type=str, default='data/results_real_world',
                        help='Output directory for results')
    parser.add_argument('--use-llm', action='store_true',
                        help='Use LLM-based pipeline instead of manual hints')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--color-tolerance', type=float, default=40.0,
                        help='Color matching tolerance')
    return parser.parse_args()


def _find_median_survival(eval_times, eval_survival, threshold=0.5):
    """Find time where survival crosses threshold."""
    for i in range(len(eval_survival) - 1):
        if eval_survival[i] >= threshold and eval_survival[i + 1] < threshold:
            if eval_survival[i] == eval_survival[i + 1]:
                return float(eval_times[i])
            frac = (threshold - eval_survival[i]) / (eval_survival[i + 1] - eval_survival[i])
            return float(eval_times[i] + frac * (eval_times[i + 1] - eval_times[i]))
    return None


def evaluate_real_world_plot(annotation, dig_results, results_dir):
    """Evaluate a single real-world plot against reported clinical values."""
    plot_id = annotation['id']
    results = {
        'id': plot_id,
        'n_curves_expected': len(annotation['curves']),
        'n_curves_detected': len(dig_results.get('curves', [])),
        'curve_results': [],
        'overall_status': 'success',
    }

    if not dig_results.get('curves'):
        results['overall_status'] = 'no_curves_detected'
        return results

    x_range = annotation['x_range']
    y_scale = 100.0 if annotation.get('y_scale_percent') else 1.0

    # Evaluate each curve
    for i, (annot_curve, dig_curve) in enumerate(zip(
        annotation['curves'][:len(dig_results['curves'])],
        dig_results['curves'][:len(annotation['curves'])]
    )):
        curve_result = {
            'label': annot_curve['label'],
            'errors': {},
        }

        # Get digitized survival function
        step_times = np.array(dig_curve['step_times'])
        step_surv = np.array(dig_curve['step_survival'])

        # Evaluate at fine grid
        n_eval = 500
        eval_times = np.linspace(x_range[0], x_range[1], n_eval)
        eval_surv = interpolate_step_function(step_times, step_surv, eval_times)

        # 1. Check median survival
        reported_median = annot_curve.get('reported_median_survival')
        if reported_median is not None:
            extracted_median = _find_median_survival(eval_times, eval_surv, 0.5 / y_scale * y_scale)
            # For percentage scale, the survival values from CV are in [0,1],
            # but y_range should handle this
            extracted_median_raw = _find_median_survival(eval_times, eval_surv, 0.5)
            curve_result['reported_median'] = reported_median
            curve_result['extracted_median'] = extracted_median_raw
            if extracted_median_raw is not None:
                abs_err = abs(extracted_median_raw - reported_median)
                rel_err = abs_err / reported_median if reported_median > 0 else None
                curve_result['errors']['median_abs_error'] = abs_err
                curve_result['errors']['median_rel_error'] = rel_err
            else:
                curve_result['errors']['median_abs_error'] = None
                curve_result['errors']['median_rel_error'] = None

        # 2. Check survival at specific time points
        reported_at = annot_curve.get('reported_survival_at', {})
        timepoint_errors = []
        for time_str, reported_val in reported_at.items():
            t = float(time_str)
            if t < x_range[0] or t > x_range[1]:
                continue
            # Interpolate
            idx = np.searchsorted(eval_times, t)
            if idx >= len(eval_surv):
                idx = len(eval_surv) - 1
            extracted_val = eval_surv[idx]

            abs_err = abs(extracted_val - reported_val)
            timepoint_errors.append({
                'time': t,
                'reported': reported_val,
                'extracted': float(extracted_val),
                'abs_error': float(abs_err),
            })

        curve_result['timepoint_checks'] = timepoint_errors
        if timepoint_errors:
            curve_result['errors']['mean_timepoint_abs_error'] = float(
                np.mean([e['abs_error'] for e in timepoint_errors])
            )

        results['curve_results'].append(curve_result)

    # Generate overlay visualization
    _save_overlay(annotation, dig_results, results_dir / f'{plot_id}_overlay.png')

    return results


def _save_overlay(annotation, dig_results, output_path):
    """Overlay digitized curves on the original image."""
    img_path = Path(annotation['_image_path'])
    img = cv2.imread(str(img_path))
    if img is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: original image
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Published Figure')
    axes[0].axis('off')

    # Right: digitized curves
    ax = axes[1]
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']

    for i, curve in enumerate(dig_results.get('curves', [])):
        times = curve.get('eval_times', curve.get('step_times', []))
        surv = curve.get('eval_survival', curve.get('step_survival', []))
        label = curve.get('label', f'Curve {i+1}')
        color = colors[i % len(colors)]
        ax.step(times, surv, where='post', color=color, linewidth=2, label=f'Digitized: {label}')

    # Add reported ground truth points
    for i, annot_curve in enumerate(annotation['curves']):
        color = colors[i % len(colors)]
        reported_at = annot_curve.get('reported_survival_at', {})
        for time_str, val in reported_at.items():
            t = float(time_str)
            ax.plot(t, val, 'o', color=color, markersize=10, markeredgecolor='black',
                    markeredgewidth=1.5, zorder=5)

        # Mark median
        median = annot_curve.get('reported_median_survival')
        if median is not None:
            ax.axvline(x=median, color=color, linestyle=':', alpha=0.5)
            ax.plot(median, 0.5, 's', color=color, markersize=10, markeredgecolor='black',
                    markeredgewidth=1.5, zorder=5)

    ax.set_xlabel(annotation.get('x_label', 'Time'))
    ax.set_ylabel(annotation.get('y_label', 'Survival Probability'))
    ax.set_title(f'Digitized: {annotation["id"]}')
    ax.legend(fontsize=8)
    ax.set_xlim(annotation['x_range'])
    ax.set_ylim(annotation['y_range'])
    ax.grid(True, alpha=0.2)

    # Add legend note for ground truth dots
    ax.annotate('● = Reported values from paper\n■ = Reported median survival',
                xy=(0.02, 0.02), xycoords='axes fraction', fontsize=7,
                va='bottom', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_report(all_results, annotations):
    """Generate a text report for real-world evaluation."""
    lines = []
    lines.append("=" * 70)
    lines.append("REAL-WORLD KM PLOT DIGITIZATION BENCHMARK")
    lines.append("=" * 70)
    lines.append("")

    n_total = len(all_results)
    n_success = sum(1 for r in all_results if r['overall_status'] == 'success')
    lines.append(f"  Total plots: {n_total}")
    lines.append(f"  Successful: {n_success}/{n_total}")
    lines.append("")

    # Aggregate median survival errors
    all_median_abs = []
    all_median_rel = []
    all_timepoint_abs = []

    for result in all_results:
        for cr in result.get('curve_results', []):
            errs = cr.get('errors', {})
            if errs.get('median_abs_error') is not None:
                all_median_abs.append(errs['median_abs_error'])
            if errs.get('median_rel_error') is not None:
                all_median_rel.append(errs['median_rel_error'])
            if errs.get('mean_timepoint_abs_error') is not None:
                all_timepoint_abs.append(errs['mean_timepoint_abs_error'])

    lines.append("AGGREGATE RESULTS")
    lines.append("-" * 40)
    if all_median_abs:
        lines.append(f"  Median Survival Absolute Error:")
        lines.append(f"    Mean: {np.mean(all_median_abs):.2f} months")
        lines.append(f"    Median: {np.median(all_median_abs):.2f} months")
        lines.append(f"    Max: {np.max(all_median_abs):.2f} months")
    if all_median_rel:
        lines.append(f"  Median Survival Relative Error:")
        lines.append(f"    Mean: {np.mean(all_median_rel)*100:.1f}%")
        lines.append(f"    Median: {np.median(all_median_rel)*100:.1f}%")
    if all_timepoint_abs:
        lines.append(f"  Timepoint Survival Absolute Error:")
        lines.append(f"    Mean: {np.mean(all_timepoint_abs):.4f}")
        lines.append(f"    Median: {np.median(all_timepoint_abs):.4f}")
    lines.append("")

    # Compare with KM-GPT paper
    lines.append("COMPARISON WITH KM-GPT PAPER")
    lines.append("-" * 40)
    lines.append("  KM-GPT reports for real-world clinical plots:")
    lines.append("    - 100% accuracy on axis/risk table extraction")
    lines.append("    - Visual overlay agreement with original curves")
    lines.append("    - Median OS error within reported 95% CIs")
    lines.append("")
    lines.append("  Our results on real-world plots:")
    if all_median_abs:
        lines.append(f"    - Mean median survival error: {np.mean(all_median_abs):.2f} months")
    if all_median_rel:
        lines.append(f"    - Mean relative error: {np.mean(all_median_rel)*100:.1f}%")
    if all_timepoint_abs:
        lines.append(f"    - Mean timepoint error: {np.mean(all_timepoint_abs):.4f}")
    lines.append("")

    # Per-plot details
    lines.append("PER-PLOT DETAILS")
    lines.append("-" * 70)
    for result in all_results:
        lines.append(f"\n  Plot: {result['id']}")
        lines.append(f"    Curves detected: {result['n_curves_detected']}/{result['n_curves_expected']}")
        lines.append(f"    Status: {result['overall_status']}")

        for cr in result.get('curve_results', []):
            lines.append(f"\n    Curve: {cr['label']}")
            if cr.get('reported_median') is not None:
                lines.append(f"      Reported median:  {cr['reported_median']} months")
                lines.append(f"      Extracted median: {cr.get('extracted_median', 'N/A')}"
                             f" months")
                errs = cr.get('errors', {})
                if errs.get('median_abs_error') is not None:
                    lines.append(f"      Abs error:       {errs['median_abs_error']:.2f} months")
                    if errs.get('median_rel_error') is not None:
                        lines.append(f"      Rel error:       {errs['median_rel_error']*100:.1f}%")

            for tp in cr.get('timepoint_checks', []):
                lines.append(f"      S({tp['time']:.0f}m): "
                             f"reported={tp['reported']:.2f}, "
                             f"extracted={tp['extracted']:.3f}, "
                             f"error={tp['abs_error']:.3f}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    annot_path = data_dir / 'annotations.json'
    with open(annot_path, 'r') as f:
        annotations = json.load(f)

    sources = annotations['sources']
    print(f"\nLoaded {len(sources)} real-world plot annotations")

    all_results = []

    for annot in sources:
        plot_id = annot['id']
        image_file = data_dir / annot['image']

        if not image_file.exists():
            print(f"\n  SKIP: {plot_id} - image not found: {annot['image']}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {plot_id}")
        print(f"  Paper: {annot['paper']}")
        print(f"  DOI: {annot['doi']}")
        print(f"  Difficulty: {annot['difficulty']}")
        print(f"  Curves: {len(annot['curves'])}")

        try:
            # Build a ground truth dict compatible with digitize_with_ground_truth_hints
            gt_for_cv = {
                'x_range': annot['x_range'],
                'y_range': annot['y_range'],
                'curves': [
                    {
                        'color': c['color'],
                        'label': c['label'],
                    }
                    for c in annot['curves']
                ],
            }

            # Run digitization
            dig_results = digitize_with_ground_truth_hints(
                str(image_file),
                gt_for_cv,
                color_tolerance=args.color_tolerance,
                verbose=args.verbose,
            )

            print(f"  Detected {len(dig_results.get('curves', []))} curves")

            # Store image path for overlay
            annot['_image_path'] = str(image_file)

            # Evaluate
            result = evaluate_real_world_plot(annot, dig_results, results_dir)
            all_results.append(result)

            # Print summary
            for cr in result.get('curve_results', []):
                label = cr['label']
                errs = cr.get('errors', {})
                median_err = errs.get('median_abs_error')
                tp_err = errs.get('mean_timepoint_abs_error')
                print(f"    {label}: median_err={median_err}, tp_err={tp_err}")

        except Exception as e:
            print(f"  ERROR: {e}")
            if args.verbose:
                traceback.print_exc()
            all_results.append({
                'id': plot_id,
                'n_curves_expected': len(annot['curves']),
                'n_curves_detected': 0,
                'curve_results': [],
                'overall_status': f'error: {str(e)}',
            })

    # Generate report
    report = generate_report(all_results, annotations)
    print("\n" + report)

    # Save report
    report_path = results_dir / 'real_world_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)

    # Save JSON results
    json_path = results_dir / 'real_world_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {results_dir}/")
    print(f"Report: {report_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
KM Curve Digitizer Benchmark Runner.

Generates synthetic KM plots, runs the digitizer on each, and evaluates performance.

Usage:
    # Default benchmark (40 plots, no API key needed for CV-only)
    python run_benchmark.py --cv-only

    # Large benchmark (200 plots) for rigorous evaluation
    python run_benchmark.py --cv-only --size large

    # XL benchmark (500 plots) for publication-grade evaluation
    python run_benchmark.py --cv-only --size xl

    # Full hybrid benchmark (requires ANTHROPIC_API_KEY)
    python run_benchmark.py

    # Generate synthetic data only (no API key needed)
    python run_benchmark.py --generate-only --size large

    # Run on a single image
    python run_benchmark.py --single path/to/image.png

    # Control benchmark size
    python run_benchmark.py --max-plots 10
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.synthesizer import generate_benchmark_suite
from src.digitizer import KMDigitizer
from src.cv_only_digitizer import digitize_with_ground_truth_hints, digitize_cv_only
from src.metrics import evaluate_plot, generate_evaluation_report
from src.utils import load_ground_truth, save_ground_truth


def parse_args():
    parser = argparse.ArgumentParser(description='KM Curve Digitizer Benchmark')
    parser.add_argument('--output-dir', type=str, default='data/synthetic',
                        help='Directory for synthetic benchmark data')
    parser.add_argument('--results-dir', type=str, default='data/results',
                        help='Directory for results')
    parser.add_argument('--generate-only', action='store_true',
                        help='Only generate synthetic data, skip digitization')
    parser.add_argument('--skip-generate', action='store_true',
                        help='Skip generation, use existing data')
    parser.add_argument('--single', type=str, default=None,
                        help='Digitize a single image')
    parser.add_argument('--max-plots', type=int, default=None,
                        help='Maximum number of plots to process')
    parser.add_argument('--model', type=str, default='claude-sonnet-4-20250514',
                        help='Claude model to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--color-tolerance', type=float, default=35.0,
                        help='Color matching tolerance for CV extraction')
    parser.add_argument('--cv-only', action='store_true',
                        help='Use CV-only mode (no LLM, uses ground truth hints for axis/color)')
    parser.add_argument('--cv-auto', action='store_true',
                        help='Use fully automatic CV-only mode (no LLM, no ground truth hints)')
    parser.add_argument('--size', type=str, default='default',
                        choices=['small', 'default', 'large', 'xl'],
                        help='Benchmark size: small(20), default(40), large(200), xl(500)')
    return parser.parse_args()


def run_single_image(image_path: str, model: str, verbose: bool = True):
    """Digitize a single image and display results."""
    print(f"\nDigitizing: {image_path}")
    print("=" * 60)

    digitizer = KMDigitizer(model=model, verbose=verbose)
    results = digitizer.digitize(image_path)

    print(f"\nResults:")
    print(f"  Curves detected: {len(results['curves'])}")
    for curve in results['curves']:
        print(f"\n  Curve: {curve['label']}")
        print(f"    Color: {curve['color']}")
        print(f"    Raw points: {curve.get('n_raw_points', 'N/A')}")
        print(f"    Step transitions: {len(curve['step_times'])}")

        # Print some sample values
        times = np.array(curve['eval_times'])
        surv = np.array(curve['eval_survival'])
        sample_idx = np.linspace(0, len(times) - 1, 10, dtype=int)
        print(f"    Sample values:")
        for idx in sample_idx:
            print(f"      t={times[idx]:.1f}: S(t)={surv[idx]:.3f}")

    return results


BENCHMARK_SIZES = {
    'small':   {'n_easy': 5,   'n_medium': 8,   'n_hard': 7},
    'default': {'n_easy': 8,   'n_medium': 13,  'n_hard': 19},
    'large':   {'n_easy': 40,  'n_medium': 80,  'n_hard': 80},
    'xl':      {'n_easy': 100, 'n_medium': 200, 'n_hard': 200},
}


def generate_benchmark(output_dir: str, seed: int, size: str = 'default'):
    """Generate the synthetic benchmark dataset."""
    sizes = BENCHMARK_SIZES[size]
    total = sum(sizes.values())
    print("\n" + "=" * 60)
    print(f"GENERATING SYNTHETIC BENCHMARK (size={size}, {total} plots)")
    print("=" * 60)

    metadata = generate_benchmark_suite(output_dir, seed=seed, **sizes)
    print(f"\nGenerated {len(metadata)} synthetic plots in {output_dir}")
    return metadata


def run_benchmark(
    output_dir: str,
    results_dir: str,
    model: str,
    max_plots: int = None,
    color_tolerance: float = 35.0,
    verbose: bool = False,
    cv_only: bool = False,
    cv_auto: bool = False,
):
    """Run the full benchmark: digitize all plots and evaluate.

    Args:
        cv_only: Use CV extraction with ground truth axis/color hints.
        cv_auto: Use fully automatic CV extraction (no hints, no LLM).
    """
    mode_name = "CV-ONLY (GT hints)" if cv_only else ("CV-AUTO" if cv_auto else "HYBRID LLM+CV")
    print("\n" + "=" * 60)
    print(f"RUNNING BENCHMARK — Mode: {mode_name}")
    print("=" * 60)

    # Load metadata
    meta_path = os.path.join(output_dir, 'benchmark_index.json')
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    if max_plots:
        metadata = metadata[:max_plots]

    print(f"Processing {len(metadata)} plots...")

    # Initialize digitizer (only for LLM mode)
    digitizer = None
    if not cv_only and not cv_auto:
        digitizer = KMDigitizer(
            model=model,
            color_tolerance=color_tolerance,
            verbose=verbose,
        )

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_timings = []
    errors = []

    for i, meta in enumerate(metadata):
        name = meta['name']
        image_path = meta['image_path']
        gt_path = meta['ground_truth_path']

        print(f"\n[{i+1}/{len(metadata)}] Processing: {name} ({meta['difficulty']}, "
              f"{meta['n_curves']} curves)")

        try:
            # Load ground truth (always needed for evaluation)
            gt = load_ground_truth(gt_path)

            start_time = time.time()

            if cv_only:
                dig_results = digitize_with_ground_truth_hints(
                    image_path, gt,
                    color_tolerance=color_tolerance,
                    verbose=verbose,
                )
            elif cv_auto:
                # Use ground truth ranges but auto-detect colors
                dig_results = digitize_cv_only(
                    image_path,
                    x_range=tuple(gt['x_range']),
                    y_range=tuple(gt['y_range']),
                    color_tolerance=color_tolerance,
                    verbose=verbose,
                )
            else:
                dig_results = digitizer.digitize(image_path)

            elapsed = time.time() - start_time
            all_timings.append(elapsed)

            print(f"  Digitized in {elapsed:.1f}s, found {len(dig_results['curves'])} curves")

            # Build digitized curves in the format expected by evaluate_plot
            dig_curves = []
            for dc in dig_results['curves']:
                dig_curves.append({
                    'label': dc['label'],
                    'step_times': dc['step_times'],
                    'step_survival': dc['step_survival'],
                })

            # Evaluate
            eval_result = evaluate_plot(gt, dig_curves)
            eval_result['name'] = name
            eval_result['timing'] = elapsed

            agg = eval_result['aggregate']
            mae = agg.get('mean_mae', float('nan'))
            acc5 = agg.get('mean_accuracy_5pct', float('nan'))
            ccc = agg.get('mean_concordance_correlation', float('nan'))
            print(f"  MAE={mae:.4f}, Accuracy@5%={acc5*100:.1f}%, CCC={ccc:.4f}")

            all_results.append(eval_result)

            # Save individual result
            result_path = results_dir / f'{name}_result.json'
            save_ground_truth(eval_result, str(result_path))

            # Generate visualization for this plot
            _save_comparison_plot(
                gt, dig_results, name,
                str(results_dir / f'{name}_comparison.png')
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            if verbose:
                traceback.print_exc()
            errors.append({'name': name, 'error': str(e)})
            # Add a failure result
            all_results.append({
                'name': name,
                'aggregate': {
                    'mean_mae': 1.0,
                    'mean_rmse': 1.0,
                    'mean_accuracy_5pct': 0.0,
                    'n_curves_detected': 0,
                    'n_curves_expected': meta['n_curves'],
                },
                'per_curve': [],
                'error': str(e),
            })

    # Generate report
    report = generate_evaluation_report(all_results, metadata)
    print("\n" + report)

    # Save report
    report_path = results_dir / 'benchmark_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)

    # Save summary results
    summary = {
        'n_plots': len(metadata),
        'n_errors': len(errors),
        'mean_timing': float(np.mean(all_timings)) if all_timings else 0,
        'errors': errors,
        'results': all_results,
    }
    save_ground_truth(summary, str(results_dir / 'benchmark_summary.json'))

    # Generate aggregate visualization
    _save_aggregate_plots(all_results, metadata, str(results_dir))

    print(f"\nResults saved to {results_dir}/")
    print(f"Report: {report_path}")
    if errors:
        print(f"Errors: {len(errors)} plots failed")

    return all_results


def _save_comparison_plot(gt, dig_results, name, output_path):
    """Save a visual comparison of ground truth vs digitized curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors_gt = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    colors_dig = ['#aec7e8', '#ff9896', '#98df8a', '#ffbb78', '#c5b0d5']

    for i, gt_curve in enumerate(gt['curves']):
        gt_times = gt_curve['eval_times']
        gt_surv = gt_curve['eval_survival']
        color = colors_gt[i % len(colors_gt)]
        ax.plot(gt_times, gt_surv, '-', color=color, linewidth=2,
                label=f"GT: {gt_curve['label']}", alpha=0.8)

    for i, dig_curve in enumerate(dig_results['curves']):
        times = dig_curve['eval_times']
        surv = dig_curve['eval_survival']
        color = colors_dig[i % len(colors_dig)]
        ax.plot(times, surv, '--', color=color, linewidth=2,
                label=f"Dig: {dig_curve['label']}", alpha=0.8)

    ax.set_xlabel('Time')
    ax.set_ylabel('Survival Probability')
    ax.set_title(f'Ground Truth vs Digitized: {name}')
    ax.legend(fontsize=8)
    ax.set_ylim(gt['y_range'])
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def _save_aggregate_plots(all_results, metadata, output_dir):
    """Save aggregate performance visualization."""
    output_dir = Path(output_dir)

    # MAE by difficulty
    difficulties = sorted(set(m['difficulty'] for m in metadata))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: MAE distribution by difficulty
    ax = axes[0]
    for diff in difficulties:
        indices = [i for i, m in enumerate(metadata) if m['difficulty'] == diff]
        maes = [all_results[i]['aggregate'].get('mean_mae', 1.0) for i in indices
                if i < len(all_results)]
        if maes:
            ax.boxplot([maes], positions=[difficulties.index(diff)],
                      tick_labels=[diff], widths=0.6)
    ax.set_ylabel('MAE')
    ax.set_title('MAE by Difficulty')
    ax.grid(True, alpha=0.3)

    # Plot 2: Accuracy distribution
    ax = axes[1]
    for diff in difficulties:
        indices = [i for i, m in enumerate(metadata) if m['difficulty'] == diff]
        acc5 = [all_results[i]['aggregate'].get('mean_accuracy_5pct', 0) * 100
                for i in indices if i < len(all_results)]
        if acc5:
            ax.boxplot([acc5], positions=[difficulties.index(diff)],
                      tick_labels=[diff], widths=0.6)
    ax.set_ylabel('Accuracy @ 5% (%)')
    ax.set_title('Accuracy by Difficulty')
    ax.grid(True, alpha=0.3)

    # Plot 3: CCC distribution
    ax = axes[2]
    for diff in difficulties:
        indices = [i for i, m in enumerate(metadata) if m['difficulty'] == diff]
        ccc = [all_results[i]['aggregate'].get('mean_concordance_correlation', 0)
               for i in indices if i < len(all_results)]
        if ccc:
            ax.boxplot([ccc], positions=[difficulties.index(diff)],
                      tick_labels=[diff], widths=0.6)
    ax.set_ylabel('Concordance Correlation')
    ax.set_title('CCC by Difficulty')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(output_dir / 'aggregate_performance.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot: Per-plot MAE scatter
    fig, ax = plt.subplots(figsize=(12, 5))
    for diff in difficulties:
        indices = [i for i, m in enumerate(metadata) if m['difficulty'] == diff]
        maes = [(i, all_results[i]['aggregate'].get('mean_mae', 1.0))
                for i in indices if i < len(all_results)]
        if maes:
            xs, ys = zip(*maes)
            ax.scatter(xs, ys, label=diff, s=30, alpha=0.7)

    ax.set_xlabel('Plot Index')
    ax.set_ylabel('MAE')
    ax.set_title('Per-Plot MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='5% threshold')

    plt.tight_layout()
    fig.savefig(str(output_dir / 'per_plot_mae.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()

    # Single image mode
    if args.single:
        run_single_image(args.single, args.model, args.verbose)
        return

    # Benchmark mode
    if not args.skip_generate:
        generate_benchmark(args.output_dir, args.seed, size=args.size)

    if args.generate_only:
        print("\nBenchmark data generated. Run without --generate-only to evaluate.")
        return

    # Check for API key (not needed for CV-only modes)
    if not args.cv_only and not args.cv_auto and not os.environ.get('ANTHROPIC_API_KEY'):
        print("\nNo ANTHROPIC_API_KEY found. Falling back to --cv-only mode.")
        print("Set it with: export ANTHROPIC_API_KEY=your_key_here")
        print("Or use --cv-only to run with ground truth hints (no LLM needed).")
        args.cv_only = True

    run_benchmark(
        args.output_dir,
        args.results_dir,
        args.model,
        max_plots=args.max_plots,
        color_tolerance=args.color_tolerance,
        verbose=args.verbose,
        cv_only=args.cv_only,
        cv_auto=args.cv_auto,
    )


if __name__ == '__main__':
    main()

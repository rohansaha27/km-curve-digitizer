"""
Synthetic Kaplan-Meier curve generator.

Generates KM plots with known ground truth for benchmarking the digitizer.
Supports various difficulty levels including:
- Single and multiple overlapping curves
- Truncated y-axis
- Low resolution images
- Censoring tick marks
- Confidence intervals
- Various color schemes and line styles
- Number-at-risk tables
"""

import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from lifelines import KaplanMeierFitter

from .utils import save_ground_truth


@dataclass
class CurveSpec:
    """Specification for a single KM curve."""
    label: str
    median_survival: float  # Median survival time
    n_subjects: int = 100
    censoring_rate: float = 0.2  # Fraction of subjects censored
    color: str = '#0000FF'
    linewidth: float = 2.0
    hazard_shape: float = 1.0  # Weibull shape (1=exponential, <1=decreasing, >1=increasing hazard)


@dataclass
class PlotSpec:
    """Specification for a KM plot."""
    curves: List[CurveSpec]
    x_label: str = 'Time (months)'
    y_label: str = 'Survival Probability'
    x_max: Optional[float] = None  # Auto if None
    y_min: float = 0.0
    y_max: float = 1.0
    show_ci: bool = False
    show_censoring: bool = True
    show_grid: bool = False
    show_at_risk: bool = False
    show_legend: bool = True
    figsize: Tuple[float, float] = (8, 6)
    dpi: int = 150
    title: str = ''
    difficulty: str = 'medium'  # easy, medium, hard, extreme
    # Optional image degradation (applied after saving) for extreme tier
    degrade_jpeg_quality: Optional[int] = None  # e.g. 25 for heavy compression
    degrade_blur_sigma: Optional[float] = None  # Gaussian blur sigma in pixels
    degrade_noise_std: Optional[float] = None   # Additive Gaussian noise std


def generate_survival_data(
    spec: CurveSpec,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate survival time-to-event data from a Weibull distribution.

    Returns:
        times: Array of observed times.
        events: Array of event indicators (1=event, 0=censored).
    """
    # Weibull distribution parameterized by median and shape
    # Median of Weibull: lambda * (ln 2)^(1/k)
    # So lambda = median / (ln 2)^(1/k)
    k = spec.hazard_shape
    lam = spec.median_survival / (np.log(2) ** (1.0 / k))

    # Generate event times from Weibull
    event_times = lam * (-np.log(rng.uniform(size=spec.n_subjects))) ** (1.0 / k)

    # Generate censoring times (uniform)
    max_follow_up = spec.median_survival * 3.0
    censor_times = rng.uniform(0, max_follow_up, size=spec.n_subjects)

    # Apply censoring
    n_to_censor = int(spec.n_subjects * spec.censoring_rate)
    censored_indices = rng.choice(spec.n_subjects, size=n_to_censor, replace=False)

    times = event_times.copy()
    events = np.ones(spec.n_subjects, dtype=int)
    for idx in censored_indices:
        if censor_times[idx] < event_times[idx]:
            times[idx] = censor_times[idx]
            events[idx] = 0

    return times, events


def fit_km_curve(
    times: np.ndarray,
    events: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Fit a Kaplan-Meier curve and return step function data.

    Returns:
        km_times: Time points (including 0).
        km_survival: Survival probabilities at each time.
        ci_lower: Lower CI bound (if computed).
        ci_upper: Upper CI bound (if computed).
    """
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed=events)

    # Extract the step function
    sf = kmf.survival_function_
    km_times = np.concatenate([[0.0], sf.index.values])
    km_survival = np.concatenate([[1.0], sf.values.flatten()])

    # Confidence intervals
    ci = kmf.confidence_interval_survival_function_
    ci_lower = np.concatenate([[1.0], ci.iloc[:, 0].values])
    ci_upper = np.concatenate([[1.0], ci.iloc[:, 1].values])

    return km_times, km_survival, ci_lower, ci_upper


def generate_plot(
    plot_spec: PlotSpec,
    output_path: str,
    ground_truth_path: str,
    seed: int = 42,
) -> Dict:
    """Generate a synthetic KM plot and save ground truth.

    Args:
        plot_spec: Specification for the plot.
        output_path: Path to save the PNG image.
        ground_truth_path: Path to save the JSON ground truth.
        seed: Random seed for reproducibility.

    Returns:
        Ground truth dictionary.
    """
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=plot_spec.figsize)

    ground_truth = {
        'curves': [],
        'x_label': plot_spec.x_label,
        'y_label': plot_spec.y_label,
        'y_range': [plot_spec.y_min, plot_spec.y_max],
        'difficulty': plot_spec.difficulty,
        'show_ci': plot_spec.show_ci,
        'show_censoring': plot_spec.show_censoring,
    }

    all_max_times = []
    curve_data_list = []

    for curve_spec in plot_spec.curves:
        times, events = generate_survival_data(curve_spec, rng)
        km_times, km_survival, ci_lower, ci_upper = fit_km_curve(times, events)
        all_max_times.append(km_times[-1])

        curve_data_list.append({
            'spec': curve_spec,
            'times': times,
            'events': events,
            'km_times': km_times,
            'km_survival': km_survival,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        })

    x_max = plot_spec.x_max or max(all_max_times) * 1.05

    for cd in curve_data_list:
        curve_spec = cd['spec']
        km_times = cd['km_times']
        km_survival = cd['km_survival']
        ci_lower = cd['ci_lower']
        ci_upper = cd['ci_upper']
        times = cd['times']
        events = cd['events']

        # Plot the KM step function
        ax.step(km_times, km_survival, where='post', color=curve_spec.color,
                linewidth=curve_spec.linewidth, label=curve_spec.label)

        # Plot confidence intervals
        if plot_spec.show_ci:
            ax.fill_between(km_times, ci_lower, ci_upper,
                          step='post', alpha=0.15, color=curve_spec.color)

        # Plot censoring marks
        if plot_spec.show_censoring:
            censored_mask = events == 0
            if np.any(censored_mask):
                cens_times = times[censored_mask]
                # For each censored time, find the survival probability
                from .utils import interpolate_step_function
                cens_surv = interpolate_step_function(km_times, km_survival, cens_times)
                ax.plot(cens_times, cens_surv, '|', color=curve_spec.color,
                       markersize=8, markeredgewidth=1.5)

        # Store ground truth - sample the step function at regular intervals
        # Also store the exact step function breakpoints
        n_eval_points = 200
        eval_times = np.linspace(0, x_max, n_eval_points)
        from .utils import interpolate_step_function
        eval_survival = interpolate_step_function(km_times, km_survival, eval_times)

        ground_truth['curves'].append({
            'label': curve_spec.label,
            'color': curve_spec.color,
            'step_times': km_times.tolist(),
            'step_survival': km_survival.tolist(),
            'eval_times': eval_times.tolist(),
            'eval_survival': eval_survival.tolist(),
            'n_subjects': curve_spec.n_subjects,
            'median_survival': curve_spec.median_survival,
        })

    # Axis formatting
    ax.set_xlim(0, x_max)
    ax.set_ylim(plot_spec.y_min, plot_spec.y_max)
    ax.set_xlabel(plot_spec.x_label, fontsize=12)
    ax.set_ylabel(plot_spec.y_label, fontsize=12)

    if plot_spec.title:
        ax.set_title(plot_spec.title, fontsize=14)

    if plot_spec.show_legend and len(plot_spec.curves) > 1:
        ax.legend(loc='best', fontsize=10)

    if plot_spec.show_grid:
        ax.grid(True, alpha=0.3)

    ground_truth['x_range'] = [0, float(x_max)]

    # Add number at risk table
    if plot_spec.show_at_risk:
        _add_at_risk_table(ax, curve_data_list, x_max, plot_spec)

    plt.tight_layout()
    fig.savefig(output_path, dpi=plot_spec.dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    # Save ground truth
    save_ground_truth(ground_truth, ground_truth_path)

    return ground_truth


def _add_at_risk_table(ax, curve_data_list, x_max, plot_spec):
    """Add number-at-risk table below the plot."""
    n_ticks = 6
    tick_times = np.linspace(0, x_max, n_ticks)

    for i, cd in enumerate(curve_data_list):
        times = cd['times']
        events = cd['events']
        label = cd['spec'].label
        color = cd['spec'].color

        at_risk = []
        for t in tick_times:
            n = np.sum(times >= t)
            at_risk.append(n)

        y_pos = -0.15 - i * 0.06
        for j, (t, n) in enumerate(zip(tick_times, at_risk)):
            ax.text(t, y_pos, str(n), transform=ax.get_xaxis_transform(),
                   ha='center', va='top', fontsize=8, color=color)

        ax.text(-0.02, y_pos, label[:15], transform=ax.get_xaxis_transform(),
               ha='right', va='top', fontsize=8, color=color, fontweight='bold')


def apply_image_degradation(
    img_path: str,
    jpeg_quality: Optional[int] = None,
    blur_sigma: Optional[float] = None,
    noise_std: Optional[float] = None,
) -> None:
    """Apply optional degradation to a saved image (in place). Used for extreme tier."""
    if jpeg_quality is None and blur_sigma is None and noise_std is None:
        return
    img = cv2.imread(img_path)
    if img is None:
        return
    if blur_sigma is not None and blur_sigma > 0:
        k = max(3, int(blur_sigma * 2) | 1)  # odd kernel
        img = cv2.GaussianBlur(img, (k, k), blur_sigma)
    if noise_std is not None and noise_std > 0:
        noise = np.random.default_rng().normal(0, noise_std, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if jpeg_quality is not None and jpeg_quality < 100:
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return
    cv2.imwrite(img_path, img)


def generate_benchmark_suite(
    output_dir: str,
    seed: int = 42,
    n_easy: int = 8,
    n_medium: int = 13,
    n_hard: int = 19,
    n_extreme: int = 5,
    only_difficulty: Optional[str] = None,
) -> List[Dict]:
    """Generate a comprehensive benchmark suite of synthetic KM plots.

    Creates plots across multiple difficulty levels with various edge cases.
    Sizes are configurable -- use larger values for more rigorous evaluation.

    Default (45 plots): n_easy=8, n_medium=13, n_hard=19, n_extreme=5
    Large   (220 plots): n_easy=40, n_medium=80, n_hard=80, n_extreme=20
    XL      (550 plots): n_easy=100, n_medium=200, n_hard=200, n_extreme=50

    If only_difficulty is 'easy', 'medium', 'hard', or 'extreme', only that
    tier is generated (using the corresponding n_* count).

    Returns:
        List of metadata dicts for each generated plot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if only_difficulty:
        only_difficulty = only_difficulty.lower()
        if only_difficulty == 'easy':
            n_medium, n_hard, n_extreme = 0, 0, 0
        elif only_difficulty == 'medium':
            n_easy, n_hard, n_extreme = 0, 0, 0
        elif only_difficulty == 'hard':
            n_easy, n_medium, n_extreme = 0, 0, 0
        elif only_difficulty == 'extreme':
            n_easy, n_medium, n_hard = 0, 0, 0
        else:
            only_difficulty = None

    # Pairs of similar colors (hard for CV to separate)
    SIMILAR_COLOR_PAIRS = [
        ('#1f77b4', '#3d8fd6'),   # Blue vs lighter blue
        ('#000000', '#2a2a2a'),   # Black vs dark gray
        ('#d62728', '#e85555'),   # Red vs light red
        ('#2ca02c', '#4cb84c'),   # Green vs light green
        ('#9467bd', '#b088d4'),   # Purple vs light purple
        ('#404040', '#606060'),   # Dark gray vs medium gray
    ]

    ALL_COLORS = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
        '#8c564b', '#e377c2', '#17becf', '#000000', '#7f7f7f',
        '#bcbd22', '#aec7e8', '#ff9896', '#98df8a', '#c49c94',
    ]
    TWO_CURVE_COLORS = [
        ('#1f77b4', '#d62728'),  # Blue vs Red
        ('#2ca02c', '#ff7f0e'),  # Green vs Orange
        ('#9467bd', '#e377c2'),  # Purple vs Pink
        ('#000000', '#d62728'),  # Black vs Red
        ('#1f77b4', '#2ca02c'),  # Blue vs Green
        ('#17becf', '#bcbd22'),  # Cyan vs Olive
        ('#8c564b', '#7f7f7f'),  # Brown vs Gray
        ('#1f77b4', '#ff7f0e'),  # Blue vs Orange
    ]
    THREE_CURVE_COLORS = [
        ('#d62728', '#ff7f0e', '#2ca02c'),  # Red / Orange / Green
        ('#1f77b4', '#9467bd', '#2ca02c'),  # Blue / Purple / Green
        ('#000000', '#d62728', '#1f77b4'),  # Black / Red / Blue
        ('#e377c2', '#17becf', '#bcbd22'),  # Pink / Cyan / Olive
    ]

    benchmark_specs = []
    idx = 0

    # ===== EASY: Single curve, clean, full y-axis =====
    for i in range(n_easy):
        rng = np.random.default_rng(seed + i)
        median = rng.uniform(4, 48)
        shape = rng.choice([0.7, 0.8, 1.0, 1.2, 1.5, 2.0])
        color = ALL_COLORS[i % len(ALL_COLORS)]

        spec = PlotSpec(
            curves=[CurveSpec(
                label='Group A',
                median_survival=median,
                n_subjects=rng.integers(30, 300),
                censoring_rate=rng.uniform(0.05, 0.4),
                color=color,
                hazard_shape=shape,
            )],
            show_censoring=bool(rng.choice([True, False])),
            show_grid=bool(rng.choice([True, False])),
            dpi=rng.choice([100, 150, 200]),
            difficulty='easy',
            title='Survival Analysis' if rng.random() > 0.5 else '',
        )
        benchmark_specs.append((f'easy_{idx:03d}', spec, seed + i))
        idx += 1

    # ===== MEDIUM: Two curves, various styles =====
    # Subdivide medium into categories
    n_med_basic = max(1, n_medium * 4 // 10)   # Basic two-curve
    n_med_ci = max(1, n_medium * 3 // 10)      # With CIs
    n_med_atrisk = max(1, n_medium - n_med_basic - n_med_ci)  # With at-risk tables

    for i in range(n_med_basic):
        rng = np.random.default_rng(seed + 100 + i)
        colors = TWO_CURVE_COLORS[i % len(TWO_CURVE_COLORS)]
        median1 = rng.uniform(6, 36)
        median2 = median1 * rng.uniform(0.3, 0.85)

        spec = PlotSpec(
            curves=[
                CurveSpec(
                    label='Treatment',
                    median_survival=median1,
                    n_subjects=rng.integers(40, 250),
                    censoring_rate=rng.uniform(0.05, 0.4),
                    color=colors[0],
                    hazard_shape=rng.choice([0.7, 0.8, 1.0, 1.2, 1.5]),
                ),
                CurveSpec(
                    label='Control',
                    median_survival=median2,
                    n_subjects=rng.integers(40, 250),
                    censoring_rate=rng.uniform(0.05, 0.4),
                    color=colors[1],
                    hazard_shape=rng.choice([0.7, 0.8, 1.0, 1.2, 1.5]),
                ),
            ],
            show_censoring=True,
            show_ci=False,
            show_grid=bool(rng.choice([True, False])),
            dpi=rng.choice([100, 150, 200]),
            difficulty='medium',
        )
        benchmark_specs.append((f'medium_{idx:03d}', spec, seed + 100 + i))
        idx += 1

    for i in range(n_med_ci):
        rng = np.random.default_rng(seed + 200 + i)
        colors = TWO_CURVE_COLORS[i % len(TWO_CURVE_COLORS)]
        median1 = rng.uniform(6, 36)
        median2 = median1 * rng.uniform(0.3, 0.85)

        spec = PlotSpec(
            curves=[
                CurveSpec(
                    label='Treatment',
                    median_survival=median1,
                    n_subjects=rng.integers(40, 250),
                    censoring_rate=rng.uniform(0.05, 0.4),
                    color=colors[0],
                    hazard_shape=rng.choice([0.8, 1.0, 1.2]),
                ),
                CurveSpec(
                    label='Control',
                    median_survival=median2,
                    n_subjects=rng.integers(40, 250),
                    censoring_rate=rng.uniform(0.05, 0.4),
                    color=colors[1],
                    hazard_shape=rng.choice([0.8, 1.0, 1.2]),
                ),
            ],
            show_censoring=True,
            show_ci=True,
            show_grid=bool(rng.choice([True, False])),
            dpi=150,
            difficulty='medium',
        )
        benchmark_specs.append((f'medium_ci_{idx:03d}', spec, seed + 200 + i))
        idx += 1

    for i in range(n_med_atrisk):
        rng = np.random.default_rng(seed + 300 + i)
        colors = TWO_CURVE_COLORS[i % len(TWO_CURVE_COLORS)]
        spec = PlotSpec(
            curves=[
                CurveSpec(
                    label='Drug',
                    median_survival=rng.uniform(8, 36),
                    n_subjects=rng.integers(80, 300),
                    censoring_rate=rng.uniform(0.1, 0.4),
                    color=colors[0],
                    hazard_shape=rng.choice([0.8, 1.0, 1.3]),
                ),
                CurveSpec(
                    label='Placebo',
                    median_survival=rng.uniform(4, 20),
                    n_subjects=rng.integers(80, 300),
                    censoring_rate=rng.uniform(0.1, 0.4),
                    color=colors[1],
                    hazard_shape=rng.choice([0.8, 1.0, 1.3]),
                ),
            ],
            show_censoring=True,
            show_at_risk=True,
            show_legend=True,
            dpi=150,
            difficulty='medium',
        )
        benchmark_specs.append((f'medium_atrisk_{idx:03d}', spec, seed + 300 + i))
        idx += 1

    # ===== HARD: Multiple edge cases =====
    # Subdivide hard into categories
    n_hard_trunc = max(1, n_hard * 3 // 10)     # Truncated y-axis
    n_hard_lowres = max(1, n_hard * 2 // 10)     # Low resolution
    n_hard_three = max(1, n_hard * 2 // 10)      # Three curves
    n_hard_overlap = max(1, n_hard - n_hard_trunc - n_hard_lowres - n_hard_three)

    # Truncated y-axis with CIs
    for i in range(n_hard_trunc):
        rng = np.random.default_rng(seed + 400 + i)
        colors = TWO_CURVE_COLORS[i % len(TWO_CURVE_COLORS)]
        y_min = rng.choice([0.2, 0.3, 0.4, 0.5, 0.6])

        spec = PlotSpec(
            curves=[
                CurveSpec(
                    label='Arm A',
                    median_survival=rng.uniform(8, 48),
                    n_subjects=rng.integers(60, 250),
                    censoring_rate=rng.uniform(0.1, 0.35),
                    color=colors[0],
                    hazard_shape=rng.choice([0.7, 0.8, 1.0, 1.3]),
                ),
                CurveSpec(
                    label='Arm B',
                    median_survival=rng.uniform(6, 30),
                    n_subjects=rng.integers(60, 250),
                    censoring_rate=rng.uniform(0.1, 0.35),
                    color=colors[1],
                    hazard_shape=rng.choice([0.7, 0.8, 1.0, 1.3]),
                ),
            ],
            y_min=y_min,
            show_censoring=True,
            show_ci=True,
            dpi=150,
            difficulty='hard',
        )
        benchmark_specs.append((f'hard_trunc_{idx:03d}', spec, seed + 400 + i))
        idx += 1

    # Low resolution
    for i in range(n_hard_lowres):
        rng = np.random.default_rng(seed + 500 + i)
        colors = TWO_CURVE_COLORS[i % len(TWO_CURVE_COLORS)]
        spec = PlotSpec(
            curves=[
                CurveSpec(
                    label='Treatment',
                    median_survival=rng.uniform(6, 36),
                    n_subjects=rng.integers(30, 200),
                    censoring_rate=rng.uniform(0.1, 0.35),
                    color=colors[0],
                    hazard_shape=1.0,
                ),
                CurveSpec(
                    label='Control',
                    median_survival=rng.uniform(4, 24),
                    n_subjects=rng.integers(30, 200),
                    censoring_rate=rng.uniform(0.1, 0.35),
                    color=colors[1],
                    hazard_shape=1.0,
                ),
            ],
            show_censoring=True,
            figsize=(4, 3),
            dpi=72,
            difficulty='hard',
        )
        benchmark_specs.append((f'hard_lowres_{idx:03d}', spec, seed + 500 + i))
        idx += 1

    # Three curves
    for i in range(n_hard_three):
        rng = np.random.default_rng(seed + 600 + i)
        colors = THREE_CURVE_COLORS[i % len(THREE_CURVE_COLORS)]
        spec = PlotSpec(
            curves=[
                CurveSpec(
                    label='High Risk',
                    median_survival=rng.uniform(3, 15),
                    n_subjects=rng.integers(40, 150),
                    censoring_rate=rng.uniform(0.1, 0.35),
                    color=colors[0],
                    hazard_shape=rng.uniform(0.7, 1.8),
                ),
                CurveSpec(
                    label='Medium Risk',
                    median_survival=rng.uniform(10, 30),
                    n_subjects=rng.integers(40, 150),
                    censoring_rate=rng.uniform(0.1, 0.35),
                    color=colors[1],
                    hazard_shape=rng.uniform(0.7, 1.8),
                ),
                CurveSpec(
                    label='Low Risk',
                    median_survival=rng.uniform(20, 60),
                    n_subjects=rng.integers(40, 150),
                    censoring_rate=rng.uniform(0.1, 0.35),
                    color=colors[2],
                    hazard_shape=rng.uniform(0.7, 1.8),
                ),
            ],
            show_censoring=True,
            show_legend=True,
            show_grid=bool(rng.choice([True, False])),
            dpi=150,
            difficulty='hard',
        )
        benchmark_specs.append((f'hard_three_{idx:03d}', spec, seed + 600 + i))
        idx += 1

    # Closely overlapping curves with CIs
    for i in range(n_hard_overlap):
        rng = np.random.default_rng(seed + 700 + i)
        base_median = rng.uniform(8, 36)
        colors = TWO_CURVE_COLORS[i % len(TWO_CURVE_COLORS)]
        spec = PlotSpec(
            curves=[
                CurveSpec(
                    label='Group 1',
                    median_survival=base_median,
                    n_subjects=rng.integers(60, 250),
                    censoring_rate=rng.uniform(0.05, 0.3),
                    color=colors[0],
                    hazard_shape=rng.choice([0.8, 1.0, 1.2]),
                ),
                CurveSpec(
                    label='Group 2',
                    median_survival=base_median * rng.uniform(0.8, 0.97),
                    n_subjects=rng.integers(60, 250),
                    censoring_rate=rng.uniform(0.05, 0.3),
                    color=colors[1],
                    hazard_shape=rng.choice([0.8, 1.0, 1.2]),
                ),
            ],
            show_censoring=True,
            show_ci=True,
            dpi=150,
            difficulty='hard',
        )
        benchmark_specs.append((f'hard_overlap_{idx:03d}', spec, seed + 700 + i))
        idx += 1

    # ===== EXTREME: Intentionally nasty conditions =====
    n_ext_similar = max(1, n_extreme * 3 // 10)      # Similar colors
    n_ext_degraded = max(1, n_extreme * 3 // 10)     # Heavy image degradation
    n_ext_combo = max(1, n_extreme - n_ext_similar - n_ext_degraded)  # Both + more

    # Two curves with very similar colors (hard to segment)
    for i in range(n_ext_similar):
        rng = np.random.default_rng(seed + 800 + i)
        colors = SIMILAR_COLOR_PAIRS[i % len(SIMILAR_COLOR_PAIRS)]
        base_median = rng.uniform(8, 36)
        spec = PlotSpec(
            curves=[
                CurveSpec(
                    label='Arm A',
                    median_survival=base_median,
                    n_subjects=rng.integers(60, 200),
                    censoring_rate=rng.uniform(0.1, 0.35),
                    color=colors[0],
                    hazard_shape=rng.choice([0.8, 1.0, 1.2]),
                ),
                CurveSpec(
                    label='Arm B',
                    median_survival=base_median * rng.uniform(0.5, 0.9),
                    n_subjects=rng.integers(60, 200),
                    censoring_rate=rng.uniform(0.1, 0.35),
                    color=colors[1],
                    hazard_shape=rng.choice([0.8, 1.0, 1.2]),
                ),
            ],
            show_censoring=True,
            show_ci=True,
            dpi=150,
            difficulty='extreme',
        )
        benchmark_specs.append((f'extreme_similar_{idx:03d}', spec, seed + 800 + i))
        idx += 1

    # Heavy image degradation: JPEG compression + blur + noise
    for i in range(n_ext_degraded):
        rng = np.random.default_rng(seed + 900 + i)
        colors = TWO_CURVE_COLORS[i % len(TWO_CURVE_COLORS)]
        spec = PlotSpec(
            curves=[
                CurveSpec(
                    label='Treatment',
                    median_survival=rng.uniform(6, 36),
                    n_subjects=rng.integers(40, 200),
                    censoring_rate=rng.uniform(0.1, 0.35),
                    color=colors[0],
                    hazard_shape=1.0,
                ),
                CurveSpec(
                    label='Control',
                    median_survival=rng.uniform(4, 28),
                    n_subjects=rng.integers(40, 200),
                    censoring_rate=rng.uniform(0.1, 0.35),
                    color=colors[1],
                    hazard_shape=1.0,
                ),
            ],
            show_censoring=True,
            show_ci=bool(rng.choice([True, False])),
            figsize=(5, 4),
            dpi=100,
            difficulty='extreme',
            degrade_jpeg_quality=rng.integers(20, 40),
            degrade_blur_sigma=rng.uniform(0.8, 1.8),
            degrade_noise_std=rng.uniform(3, 8),
        )
        benchmark_specs.append((f'extreme_degraded_{idx:03d}', spec, seed + 900 + i))
        idx += 1

    # Combo: similar colors + low res + degradation
    for i in range(n_ext_combo):
        rng = np.random.default_rng(seed + 1000 + i)
        colors = SIMILAR_COLOR_PAIRS[(i + 1) % len(SIMILAR_COLOR_PAIRS)]
        y_min = rng.choice([0.0, 0.25, 0.4]) if rng.random() > 0.5 else 0.0
        spec = PlotSpec(
            curves=[
                CurveSpec(
                    label='Group 1',
                    median_survival=rng.uniform(8, 40),
                    n_subjects=rng.integers(50, 180),
                    censoring_rate=rng.uniform(0.1, 0.35),
                    color=colors[0],
                    hazard_shape=rng.choice([0.8, 1.0, 1.2]),
                ),
                CurveSpec(
                    label='Group 2',
                    median_survival=rng.uniform(6, 32) * rng.uniform(0.7, 0.95),
                    n_subjects=rng.integers(50, 180),
                    censoring_rate=rng.uniform(0.1, 0.35),
                    color=colors[1],
                    hazard_shape=rng.choice([0.8, 1.0, 1.2]),
                ),
            ],
            y_min=y_min,
            show_censoring=True,
            show_ci=True,
            figsize=(4, 3),
            dpi=72,
            difficulty='extreme',
            degrade_jpeg_quality=rng.integers(25, 45),
            degrade_blur_sigma=rng.uniform(0.5, 1.2),
            degrade_noise_std=rng.uniform(2, 6),
        )
        benchmark_specs.append((f'extreme_combo_{idx:03d}', spec, seed + 1000 + i))
        idx += 1

    # Generate all plots
    metadata = []
    for name, spec, s in benchmark_specs:
        img_path = str(output_dir / f'{name}.png')
        gt_path = str(output_dir / f'{name}_gt.json')
        gt = generate_plot(spec, img_path, gt_path, seed=s)
        if getattr(spec, 'degrade_jpeg_quality', None) is not None or getattr(spec, 'degrade_blur_sigma', None) is not None or getattr(spec, 'degrade_noise_std', None) is not None:
            apply_image_degradation(
                img_path,
                jpeg_quality=getattr(spec, 'degrade_jpeg_quality', None),
                blur_sigma=getattr(spec, 'degrade_blur_sigma', None),
                noise_std=getattr(spec, 'degrade_noise_std', None),
            )
        metadata.append({
            'name': name,
            'image_path': img_path,
            'ground_truth_path': gt_path,
            'difficulty': spec.difficulty,
            'n_curves': len(spec.curves),
            'y_min': spec.y_min,
            'dpi': spec.dpi,
            'show_ci': spec.show_ci,
        })
        print(f'  Generated: {name} ({spec.difficulty}, {len(spec.curves)} curves)')

    # Save metadata index
    meta_path = str(output_dir / 'benchmark_index.json')
    save_ground_truth(metadata, meta_path)

    return metadata

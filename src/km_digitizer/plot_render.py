from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import io
import numpy as np
from PIL import Image, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .simulate import SimArm, kaplan_meier_curve, sample_censor_marks


@dataclass
class RenderOptions:
    dpi: int = 150
    figsize: Tuple[float, float] = (5.0, 4.0)
    grid: bool = True
    legend: bool = True
    risk_table: bool = False
    censor_marks: bool = True
    truncate_y: bool = False
    y_min: float = 0.0
    linestyles: Optional[List[str]] = None  # e.g. ["-", "--", "-."]

    # degradation
    degrade: bool = False
    downsample_factor: float = 0.6  # 0.5 -> severe
    blur_radius: float = 0.8
    jpeg_quality: int = 60


def render_km_plot(
    arms: List[SimArm],
    out_path: str,
    opts: RenderOptions,
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Render a KM plot image to out_path.
    Returns ground truth (t_norm, s) curves per arm (normalized time).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig = plt.figure(figsize=opts.figsize, dpi=opts.dpi)

    if opts.risk_table:
        # Create a gridspec: plot on top, table on bottom
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
        ax = fig.add_subplot(gs[0, 0])
        ax_tbl = fig.add_subplot(gs[1, 0])
        ax_tbl.axis("off")
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax_tbl = None

    if opts.linestyles is None:
        linestyles = ["-", "--", "-."][: len(arms)]
    else:
        linestyles = opts.linestyles

    gt_curves: List[Tuple[np.ndarray, np.ndarray]] = []

    # Determine x max so curves share x-axis
    all_times = np.concatenate([a.event_times for a in arms]) if arms else np.array([1.0])
    tmax = float(np.max(all_times)) if all_times.size else 1.0
    tmax = max(tmax, 1e-3)

    for i, arm in enumerate(arms):
        km_t, km_s = kaplan_meier_curve(arm.event_times, arm.event_observed)

        # Normalize time to [0,1] for metrics
        t_norm = km_t / tmax
        gt_curves.append((t_norm.astype(np.float32), km_s.astype(np.float32)))

        ax.step(km_t, km_s, where="post", label=f"Arm {i+1}", linestyle=linestyles[i % len(linestyles)])

        if opts.censor_marks:
            t_marks, s_marks = sample_censor_marks(
                arm.event_times, arm.event_observed, km_t, km_s, max_marks=12, rng=rng
            )
            if t_marks.size:
                ax.plot(t_marks, s_marks, linestyle="None", marker="+", markersize=6)

    ax.set_xlabel("Time")
    ax.set_ylabel("Survival")
    if opts.grid:
        ax.grid(True, alpha=0.3)

    if opts.truncate_y:
        ax.set_ylim(opts.y_min, 1.0)
    else:
        ax.set_ylim(0.0, 1.0)

    ax.set_xlim(0.0, tmax)

    if opts.legend:
        ax.legend(loc="best", frameon=True, fontsize=8)

    if opts.risk_table and ax_tbl is not None:
        # Simple at-risk table at fixed time points
        time_pts = np.linspace(0, tmax, 5)
        cell_text = []
        row_labels = []
        col_labels = [f"{tp:.1f}" for tp in time_pts]
        for i, arm in enumerate(arms):
            row_labels.append(f"Arm {i+1}")
            # number at risk = count with observed time >= tp
            nar = [int(np.sum(arm.event_times >= tp)) for tp in time_pts]
            cell_text.append([str(x) for x in nar])

        ax_tbl.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        ax_tbl.set_title("Number at risk", fontsize=9)

    fig.tight_layout()

    # Save to bytes first
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB")

    if opts.degrade:
        # downsample / upsample
        w, h = img.size
        w2 = max(64, int(w * opts.downsample_factor))
        h2 = max(64, int(h * opts.downsample_factor))
        img = img.resize((w2, h2), resample=Image.BILINEAR).resize((w, h), resample=Image.BILINEAR)
        # blur
        if opts.blur_radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=opts.blur_radius))
        # jpeg compression roundtrip
        jb = io.BytesIO()
        img.save(jb, format="JPEG", quality=int(opts.jpeg_quality))
        jb.seek(0)
        img = Image.open(jb).convert("RGB")

    img.save(out_path)

    return gt_curves

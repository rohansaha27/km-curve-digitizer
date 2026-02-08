from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

from km_digitizer.simulate import simulate_survival_arm
from km_digitizer.plot_render import render_km_plot, RenderOptions
from km_digitizer.config import DigitizeConfig
from km_digitizer.digitize import digitize_km_curves
from km_digitizer.matching import match_curves_by_min_iae
from km_digitizer.metrics import curve_iae, curve_ae


@dataclass
class BenchRow:
    plot_id: int
    n_true: int
    n_pred: int
    matched: int
    median_iae: float
    median_ae: float
    fail: int


def make_plot_case(rng: np.random.Generator) -> Tuple[int, RenderOptions, DigitizeConfig, List[Tuple[float, float]]]:
    """
    Returns:
      n_arms, render_opts, digitize_cfg, hazards list [(event_hazard, censor_hazard), ...]
    """
    n_arms = int(rng.integers(1, 4))
    base = float(rng.uniform(0.05, 0.25))
    # create similar hazards to induce overlap sometimes
    hazards = []
    for i in range(n_arms):
        event_h = max(0.01, base + float(rng.normal(0, 0.03)))
        censor_h = float(rng.uniform(0.02, 0.20))
        hazards.append((event_h, censor_h))

    opts = RenderOptions(
        dpi=int(rng.choice([80, 120, 200])),
        grid=bool(rng.integers(0, 2)),
        legend=bool(rng.integers(0, 2)),
        risk_table=bool(rng.integers(0, 2)),
        censor_marks=bool(rng.integers(0, 2)),
        degrade=bool(rng.integers(0, 2)),
        truncate_y=bool(rng.integers(0, 2)),
        y_min=float(rng.uniform(0.25, 0.45)),
    )

    cfg = DigitizeConfig(
        debug=False,
        include_dark_pixels=True,
        enable_ocr_y_calibration=False,  # benchmark can toggle separately if desired
    )
    return n_arms, opts, cfg, hazards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_plots", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--enable_ocr_y", action="store_true", help="Enable OCR calibration in digitizer")
    args = ap.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    data_dir = os.path.join(outdir, "data")
    os.makedirs(data_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    rows: List[Dict[str, Any]] = []

    for plot_id in range(args.n_plots):
        n_arms, opts, cfg, hazards = make_plot_case(rng)
        cfg.enable_ocr_y_calibration = bool(args.enable_ocr_y)

        arms = []
        for i in range(n_arms):
            ev_h, ce_h = hazards[i]
            arm = simulate_survival_arm(n=140, hazard=ev_h, censor_hazard=ce_h, rng=rng)
            arm.name = f"Arm {i+1}"
            arms.append(arm)

        img_path = os.path.join(data_dir, f"plot_{plot_id:05d}.png")
        gt_curves = render_km_plot(arms, img_path, opts, rng)

        pred_curves_obj, _info = digitize_km_curves(img_path, cfg)

        pred_curves = [(c.t_norm, c.s) for c in pred_curves_obj]
        matches = match_curves_by_min_iae(gt_curves, pred_curves)

        # Compute per-match metrics
        iaes = []
        aes = []
        for (ti, pj, _cost) in matches:
            tt, yt = gt_curves[ti]
            tp, yp = pred_curves[pj]
            iaes.append(curve_iae(tt, yt, tp, yp))
            aes.append(curve_ae(tt, yt, tp, yp))

        fail = 1 if len(matches) < len(gt_curves) else 0
        row = {
            "plot_id": plot_id,
            "n_true": len(gt_curves),
            "n_pred": len(pred_curves),
            "matched": len(matches),
            "median_iae": float(np.median(iaes)) if iaes else float("nan"),
            "median_ae": float(np.median(aes)) if aes else float("nan"),
            "fail": fail,
            "dpi": opts.dpi,
            "grid": int(opts.grid),
            "legend": int(opts.legend),
            "risk_table": int(opts.risk_table),
            "censor_marks": int(opts.censor_marks),
            "degrade": int(opts.degrade),
            "truncate_y": int(opts.truncate_y),
            "y_min": float(opts.y_min),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    # Print summary
    ok = df[df["matched"] > 0]
    print(f"Plots: {args.n_plots}")
    print(f"Failure rate (missing >=1 true curve): {df['fail'].mean():.3f}")
    if len(ok) > 0:
        print(f"Median IAE (over plots, median-of-median): {ok['median_iae'].median():.4f}")
        print(f"Median AE  (over plots, median-of-median): {ok['median_ae'].median():.4f}")
    print(f"Wrote: {os.path.join(outdir, 'summary.csv')}")


if __name__ == "__main__":
    main()

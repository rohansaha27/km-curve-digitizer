#!/usr/bin/env python3
"""Digitize a single KM plot image (CV-only, with explicit axis ranges)."""
import json
import sys
from pathlib import Path

from .cv_only_digitizer import digitize_cv_only


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "my_plot.jpg"
    # Locoregional Control (%): x = 0–70 months, y = 0–100%
    # Two curves: black (Radiotherapy plus cetuximab), gray (Radiotherapy)
    x_range = (0.0, 70.0)
    y_range = (0.0, 100.0)
    curve_colors = ["#000000", "#606060"]  # black, dark gray (avoid grid)

    print(f"Digitizing: {image_path}")
    print(f"  X range: {x_range} months, Y range: {y_range}%")
    print()

    results = digitize_cv_only(
        image_path,
        x_range=x_range,
        y_range=y_range,
        curve_colors=curve_colors,
        n_eval_points=200,
        color_tolerance=45.0,
        verbose=True,
    )

    curves = results.get("curves", [])
    labels = ["Radiotherapy plus cetuximab", "Radiotherapy"]
    for i, curve in enumerate(curves):
        curve["label"] = labels[i] if i < len(labels) else curve.get("label", f"Curve {i+1}")

    print()
    print("Results:")
    for curve in curves:
        print(f"  {curve['label']}: {len(curve['step_times'])} steps")

    # Save JSON
    out_json = Path(image_path).stem + "_digitized.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "curves": [
                    {
                        "label": c["label"],
                        "eval_times": c["eval_times"],
                        "eval_survival": c["eval_survival"],
                        "step_times": c["step_times"],
                        "step_survival": c["step_survival"],
                    }
                    for c in curves
                ],
                "x_range": list(x_range),
                "y_range": list(y_range),
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {out_json}")

    # Save CSVs per curve
    for curve in curves:
        label = curve["label"].replace(" ", "_").replace("+", "plus")
        csv_path = Path(image_path).stem + f"_curve_{label}.csv"
        with open(csv_path, "w") as f:
            f.write("months,locoregional_control_pct\n")
            for t, s in zip(curve["eval_times"], curve["eval_survival"]):
                f.write(f"{t:.2f},{s:.4f}\n")
        print(f"Saved: {csv_path}")

    return results


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
import pandas as pd

from km_digitizer.config import DigitizeConfig
from km_digitizer.digitize import digitize_km_curves
from km_digitizer.debug import save_debug_overlay, curves_to_csv_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to KM plot image (png/jpg)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--debug", action="store_true", help="Save debug overlay image")
    ap.add_argument("--ocr_y", action="store_true", help="Enable OCR y-axis calibration (needs tesseract)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cfg = DigitizeConfig(debug=args.debug, enable_ocr_y_calibration=args.ocr_y)

    curves, info = digitize_km_curves(args.image, cfg)

    rows = curves_to_csv_rows(curves)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "curves.csv"), index=False)

    if args.debug and "debug_overlay_bgr" in info:
        save_debug_overlay(os.path.join(args.outdir, "debug_overlay.png"), info["debug_overlay_bgr"])

    print(f"Digitized {len(curves)} curve(s). Output in: {args.outdir}")


if __name__ == "__main__":
    main()

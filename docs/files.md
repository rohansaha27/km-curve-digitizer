# Project file reference

A short description of what each file and directory does.

---
## Root

|------|-------------|
| **app.py** | Streamlit web dashboard. Run with `streamlit run app.py`. Lets you browse synthetic/real-world data, run benchmarks, view comparison images and reports, 
| **run_benchmark.py** | Synthetic benchmark runner. Generates KM plots with known ground truth, runs the digitizer (hybrid or CV-only), and evaluates metrics. Supports `--cv-only`, `--generate-only`, `--single`, and sizes like `default` / `large` / `xl`. |
| **run_real_world_benchmark.py** | Real-world benchmark. Runs the digitizer on published KM images in `data/real_world/`, compares extracted median survival and survival-at-time to values in `annotations.json`, and writes results and overlay images to `data/results_real_world/`. |
| **requirements.txt** | Python dependencies: OpenCV, NumPy, matplotlib, Pillow, anthropic, lifelines, pandas, scikit-learn, scipy, tqdm, streamlit. |

---

## `src/` — Core library

| File | Description |
|------|-------------|
| **\_\_init\_\_.py** | Package marker and short docstring for the KM Curve Digitizer package. |
| **digitizer.py** | Main **hybrid LLM + CV** pipeline. `KMDigitizer` uses the LLM to get plot metadata (axes, curve colors, legend), then CV to trace curves and post-process (monotonicity, step function). Entry point: `KMDigitizer().digitize(image_path)`. |
| **cv_only_digitizer.py** | **CV-only** pipeline (no API key). Uses heuristics or provided hints for axis ranges and curve colors, then the same CV extraction. Exposes `digitize_cv_only()` and `digitize_with_ground_truth_hints()` for benchmarks. |
| **llm_reader.py** | LLM-based plot reader. Uses Claude vision to extract axis ranges, curve colors, legend, plot bbox, and optional survival values. Provides `extract_plot_metadata()` and helpers to turn metadata into extraction params. |
| **cv_extractor.py** | Computer vision curve extraction. OpenCV-based: plot bbox detection (ticks, projections, Hough), color segmentation (CIELAB), scan-line tracing, step-function extraction. Used by both hybrid and CV-only pipelines. |
| **synthesizer.py** | Synthetic KM plot generator. Builds plots from `CurveSpec` / `PlotSpec` with lifelines, supports difficulty tiers (easy → extreme), CIs, censoring, at-risk tables, and optional image degradation. Used by `run_benchmark.py` to create benchmark suites. |
| **metrics.py** | Evaluation metrics for digitized curves: MAE, RMSE, max error, integrated absolute error (IAE), concordance correlation coefficient (CCC), accuracy at time points, median survival error. Can generate text and visual evaluation reports. |
| **utils.py** | Shared helpers: pixel ↔ data coordinate conversion, monotonicity enforcement, step-function interpolation, color conversion (hex ↔ RGB, CIELAB distance), ground-truth load/save. |
| **digitize_single.py** | CLI to digitize a **single** image. CV-only with hardcoded axis ranges and curve colors; edit the script for your plot. Run: `python -m src.digitize_single path/to/image.png`. Writes `*_digitized.json`. |

---

## `data/`

| Path | Description |
|------|-------------|
| **data/synthetic/** | Generated synthetic KM images and ground-truth JSON (created by `run_benchmark.py`). Contains per-plot images and a `benchmark_index.json` (or similar) listing plot specs. Often gitignored. |
| **data/results/** | Output of the **synthetic** benchmark: per-plot metrics, comparison images, aggregate report and plots (e.g. `benchmark_report.txt`, `aggregate_performance.png`). Created by `run_benchmark.py`. |
| **data/real_world/** | Real published KM plot images and annotations. `annotations.json` describes each figure (axis ranges, curve colors, reported median survival and survival-at-time from the papers). Images: e.g. `rcc_fig001.png`, `plos_fig1.png`, `ntrk_fig002.png`. |
| **data/results_real_world/** | Output of the **real-world** benchmark: extracted curves, comparison to reported values, overlay images. Created by `run_real_world_benchmark.py`. |


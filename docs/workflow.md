# Workflow and design decisions

## Pipeline overview

1. **Metadata** — Get axis ranges, plot bounds, and curve colors (and optionally labels).
2. **Tracing** — For each curve color, extract pixel coordinates in the plot region, then map pixels → data coordinates.
3. **Step function** — Turn traced points into a KM step function (flat segments, drops at events).
4. **Post-process** — Enforce monotonicity (survival never increases), clip to axis range, sample at fixed evaluation times.

Output: per-curve step and evaluated survival, plus metadata.

---

## Decision 1: Hybrid LLM + CV (not LLM-only or CV-only for everything)

- **LLM-only**: Vision models can read approximate values from a plot but are not precise enough for sub-pixel curve tracing; numeric outputs are noisy and not reliable for downstream analysis.
- **CV-only**: Axis ranges and curve colors are layout/semantic (tick labels, legend). Heuristics (projection profiles, Hough, dominant color) work on clean synthetic plots but fail on real figures (truncated axes, similar colors, CIs, grids).
- **Choice**: Use the **LLM for semantics** (axis min/max, curve count, colors, legend labels) and **CV for geometry** (plot bbox, pixel-level curve tracing). LLM gives one structured prompt; CV does deterministic, reproducible extraction. This matches how KM-GPT and similar tools split “understanding” vs “measuring.”

---

## Decision 2: LLM for metadata only

- The LLM returns a single JSON: `x_axis`/`y_axis` (min, max, ticks), `curves` (label, color_hex, line_style), `plot_region` (fracs), and flags (CI, censoring, grid). No raw curve coordinates from the LLM.
- **Why not LLM for coordinates?** Reading exact (time, survival) from pixels is error-prone and varies by model/run; we need repeatable, comparable numbers for benchmarks and downstream use.
- **Optional LLM validation**: We can sample a few time points and ask the LLM for survival values to cross-check CV output (e.g. agreement within 5%). This is off by default and used only as a sanity check, not as the source of truth.

---

## Decision 3: CV for plot bbox and curve tracing

- **Bbox**: We use projection profiles (column/row sums of dark pixels) plus Hough lines to find axis lines. In the hybrid pipeline we **merge** LLM bbox with CV bbox: if they disagree by more than ~15% of image size, we take a weighted average (e.g. 40% LLM, 60% CV) so that CV can correct obvious LLM misestimates.
- **Curve extraction**: For each target color we build a **color mask** (pixels within a distance of the target in **CIELAB**), then **scan-line tracing**: for each vertical column in the plot we find the bottommost matching pixel (so we get the “lower” edge of the line, which is the survival value at that time). Raw (pixel_x, pixel_y) are converted to (time, survival) via the known axis ranges and plot bbox. This avoids contour-following and handles crossing curves as long as colors differ.
- **CIELAB**: Color matching uses LAB distance instead of RGB so that tolerance is perceptually uniform (same numeric tolerance for “red” vs “blue” in terms of perceived difference). Reduces false picks from JPEG/compression and slight shading.

---

## Decision 4: Step function and monotonicity

- KM curves are **step functions**: flat until an event, then a drop. We don’t fit smooth curves; we detect steps from the scan-line data (e.g. median filter + threshold on vertical change) and output (step_times, step_survival).
- **Monotonicity**: Survival is non-increasing. We enforce this in post-processing (e.g. take running minimum) so that noise or overlap with CIs/grid doesn’t produce upward bumps. We also clip to the declared y range and optionally anchor the first step at t=0 and S=1 (or y_max).

---

## Decision 5: CV-only mode (no API key)

- For benchmarks, reproducibility, and cost we support a **CV-only** path: no LLM call. It requires **hints** (axis ranges and curve colors) from somewhere—e.g. from synthetic ground truth JSON or from a manual annotations file for real-world figures.
- **Use cases**: (1) Synthetic benchmark with known truth. (2) Real-world benchmark where we have `annotations.json` with reported axis ranges and curve colors from the paper. (3) Scripts where the user sets ranges/colors in code (e.g. `digitize_single.py`).
- We do **not** rely on CV-only to *discover* axis ranges or curve colors on arbitrary real plots; that’s what the LLM is for. CV-only with hints gives a fair, comparable evaluation and allows running without an API key.

---

## Decision 6: Evaluation metrics

- We compare digitized curves to ground truth (synthetic) or to reported values (real-world) using metrics aligned with the literature (e.g. KM-GPT):
  - **MAE / RMSE** of survival probabilities at common evaluation times.
  - **IAE** (integrated absolute error): area between curves, normalized by time span—single number for “how far off overall.”
  - **CCC** (concordance correlation): agreement in level and trend, not just correlation.
  - **Accuracy at 5%**: fraction of evaluation points with error &lt; 0.05.
  - **Median survival error**: for real-world, compare extracted median to reported median.
- Synthetic benchmarks use pixel-level ground truth; real-world uses reported medians and survival-at-time from the paper text, since we don’t have pixel truth.

---

## Decision 7: Synthetic vs real-world benchmarks

- **Synthetic**: We generate KM plots (lifelines + matplotlib) with known step functions, then run the digitizer and compare to that truth. Pros: controlled difficulty (single/multi-curve, CIs, truncated axis, low DPI, similar colors), no API needed in CV-only mode, many samples. Used to tune tolerance, report MAE/IAE/CCC, and compare to other tools.
- **Real-world**: We take published figures and `annotations.json` (axis ranges, curve colors, reported medians and survival-at-time from the paper). We digitize and compare to those reported values. Pros: validity on real data; cons: no pixel-level truth, only author-reported numbers. Used to check that the pipeline works on real literature and that median/survival-at-time errors are acceptable.

---

## Summary / TLDR

| Question | Decision | Rationale |
|----------|----------|-----------|
| Who gets axis/colors? | LLM (hybrid) or hints (CV-only) | Semantics and layout; CV heuristics unreliable on real figures. |
| Who traces the curve? | CV only | Sub-pixel precision and repeatability; LLM not reliable for coordinates. |
| Color space for matching? | CIELAB | Perceptually uniform; more robust to compression and lighting. |
| Curve representation? | Step function + monotonicity | KM is a step function; post-processing removes non-physical bumps. |
| How to evaluate? | MAE, IAE, CCC, accuracy@5%, median error | Match literature and capture both pointwise and overall agreement. |
| How to test without API? | CV-only + ground-truth or annotation hints | Reproducible benchmarks and real-world eval without LLM cost. |

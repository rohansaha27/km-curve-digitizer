# KM Curve Digitizer

Extracts numerical survival data from Kaplan-Meier plot images using a hybrid LLM + computer vision pipeline.

## How It Works

1. **LLM reads the plot** (Claude vision) -- extracts axis ranges, curve colors, and legend labels
2. **CV traces each curve** (OpenCV) -- color segmentation in CIELAB space, scan-line tracing, step-function reconstruction
3. **Post-processing** -- monotonicity enforcement, median filtering, coordinate mapping

A **CV-only mode** is available without an API key (requires manual axis ranges).

## Results

**Synthetic benchmark** (200 plots -- single/multi-arm, CIs, truncated axes, low-res):

- **Easy:** single curve, full y-axis, clean (optional grid/censor marks).
- **Medium:** two curves, full y-axis; adds confidence intervals and/or number-at-risk tables.
- **Hard:** truncated y-axis, low resolution (DPI 72), three curves, or closely overlapping curves with CIs.
- **Extreme:** intentionally difficult — similar-color curves (e.g. two blues), heavy image degradation (JPEG compression, blur, noise), or both combined with low res/truncated axis.

| Metric | Value |
|--------|-------|
| Median MAE | 0.0073 |
| Median IAE | 0.0074 |
| Accuracy @5% | 99.5% |
| Concordance (CCC) | 0.9944 |

Compared to **KM-GPT** (tested on 538 single-arm curves): our IAE is **2.4x better** (0.0074 vs 0.018) on harder multi-arm conditions.

**Real-world** (4 published PLOS ONE plots): median survival error ~1.5 months, 100% curve detection.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here  # optional, for hybrid mode
```

## Usage

**Web UI:**
```bash
streamlit run app.py
```

**Python:**
```python
from src.digitizer import KMDigitizer

results = KMDigitizer().digitize("path/to/km_plot.png")

# CV-only (no API key):
from src.cv_only_digitizer import digitize_cv_only
results = digitize_cv_only("plot.png", x_range=(0, 48), y_range=(0, 1))
```

**CLI:**
```bash
python -m src.digitize_single path/to/km_plot.png   # single image (CV-only, edit axis ranges in script)
python run_benchmark.py --cv-only                    # default (40 plots)
python run_benchmark.py --cv-only --size large        # 200 plots
python run_real_world_benchmark.py                    # published data
```

## Project Structure

```
src/
├── digitizer.py           # Hybrid LLM+CV pipeline
├── digitize_single.py     # Single-image CLI (CV-only)
├── cv_only_digitizer.py   # CV-only pipeline
├── llm_reader.py          # Claude vision extraction
├── cv_extractor.py        # OpenCV curve tracing
├── synthesizer.py         # Synthetic data generation
├── metrics.py             # Evaluation metrics
└── utils.py               # Shared utilities

app.py                     # Streamlit dashboard
run_benchmark.py           # Synthetic benchmark runner
run_real_world_benchmark.py # Real-world benchmark runner
```

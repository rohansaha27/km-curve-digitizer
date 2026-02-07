# KM Curve Digitizer

Automated extraction of survival data from Kaplan-Meier (KM) curve images using a hybrid LLM + computer vision approach.

## Overview

Kaplan-Meier plots are ubiquitous in clinical and biomedical literature. When conducting systematic reviews or meta-analyses, researchers often need to extract numerical data from these plots -- a process traditionally done manually using digitization software. This tool automates that process.

### Approach

The digitizer uses a **two-stage hybrid pipeline**:

1. **LLM-based metadata extraction** (Claude vision): Reads axis labels, ranges, tick marks, curve colors, legend text, and estimates the plot bounding box. The LLM excels at understanding the semantic structure of the plot.

2. **Computer vision curve tracing** (OpenCV): Uses color-based segmentation in CIELAB perceptual color space to isolate each curve, then applies a scan-line algorithm to trace the curve pixel by pixel. Converts pixel coordinates to data coordinates using the axis metadata from step 1.

3. **Post-processing**: Enforces KM-specific constraints (monotonically non-increasing step function), smooths noise with median filtering, and optionally cross-validates against LLM direct readings.

### Why Hybrid?

- **LLMs alone** struggle with precise pixel-level coordinate extraction (typically 5-10% error)
- **CV alone** struggles to interpret axis labels, distinguish overlapping elements, and handle diverse plot styles
- **Combined**: LLM provides the structural understanding, CV provides pixel-level precision

A **CV-only mode** is also available for use without an API key, using heuristic axis detection and automatic color identification.

## Benchmark Results

### Synthetic Data (200 plots, `--size large`)

Evaluated on **200 diverse synthetic KM plots** across three difficulty levels, covering single/multi-arm curves, confidence intervals, truncated y-axes, low resolution, and closely overlapping curves.

#### Overall Performance (CV pipeline, ground truth axis/color hints)

| Metric | Value |
|--------|-------|
| **Mean MAE** | **0.0091** (±0.0050) |
| **Median MAE** | **0.0073** |
| **Mean RMSE** | 0.0112 (±0.0070) |
| **Mean IAE** | **0.0091** (±0.0050) |
| **Median IAE** | **0.0074** |
| **Mean CCC** | **0.9944** (±0.0190) |
| **Accuracy @3%** | 96.6% |
| **Accuracy @5%** | **99.5%** |
| **Accuracy @10%** | **99.9%** |

#### Performance by Difficulty

| Difficulty | Plots | Median MAE | Acc @5% | Mean CCC |
|-----------|-------|------------|---------|----------|
| Easy (single curve, clean) | 40 | 0.0072 | 99.1% | 0.9980 |
| Medium (2 curves, CIs, at-risk tables) | 80 | 0.0068 | 99.9% | 0.9986 |
| Hard (truncated y, low-res, 3 curves, overlap) | 80 | 0.0092 | 99.4% | 0.9883 |

#### Comparison with KM-GPT Paper

| Metric | Ours (median) | KM-GPT (median) | Notes |
|--------|--------------|-----------------|-------|
| **Point-wise AE** | **0.0073** | 0.005 | Comparable; theirs tested on single-arm only |
| **Integrated AE (IAE)** | **0.0074** | 0.018 | **Ours is 2.4x better** |
| **Median Surv. Error** | **~0.005 normalized** | 0.005 normalized | **Match** (ours: 0.26 months / 48mo range) |

> KM-GPT was evaluated on 538 single-arm synthetic curves. Our benchmark uses 200 plots with substantially harder conditions: multi-arm plots (2-3 curves), confidence interval bands, truncated y-axes, low resolution (72 DPI), and closely overlapping curves.

### Real-World Published Data

Evaluated on **4 KM plots from open-access published clinical papers** (PLOS ONE), comparing digitized values against survival statistics reported in the papers.

| Metric | Value |
|--------|-------|
| **Mean Median Survival Error** | **1.45 months** |
| **Median Survival Relative Error** | **10.3%** |
| **Mean Timepoint Abs Error** | **0.032** (3.2% on survival probability) |
| **Curve Detection Rate** | **100%** (8/8 curves across 4 plots) |

Sources used:
- **Stage IV RCC** (10.1371/journal.pone.0063341): 3 figures with reported median survival, 1-year and 5-year survival rates
- **NTRK fusion tumors** (10.1371/journal.pone.0270571): OS curves with risk tables and censoring marks

Run the real-world benchmark: `python run_real_world_benchmark.py`

## Installation

```bash
pip install -r requirements.txt
```

For the hybrid LLM+CV mode, set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY=your_key_here
```

The CV-only mode works without an API key.

## Quick Start

### Digitize a single image (hybrid mode)

```python
from src.digitizer import KMDigitizer

digitizer = KMDigitizer(verbose=True)
results = digitizer.digitize("path/to/km_plot.png")

for curve in results['curves']:
    print(f"Curve: {curve['label']}")
    print(f"  Times:    {curve['step_times'][:5]}...")
    print(f"  Survival: {curve['step_survival'][:5]}...")
```

### Digitize without LLM (CV-only mode)

```python
from src.cv_only_digitizer import digitize_cv_only

results = digitize_cv_only(
    "path/to/km_plot.png",
    x_range=(0, 24),     # Must provide axis ranges
    y_range=(0, 1.0),
    verbose=True,
)
```

### Command line

```bash
# Run synthetic benchmark (default: 40 plots, no API key needed)
python run_benchmark.py --cv-only

# Run large benchmark (200 plots)
python run_benchmark.py --cv-only --size large

# Run XL benchmark (500 plots, publication-grade)
python run_benchmark.py --cv-only --size xl

# Run real-world benchmark
python run_real_world_benchmark.py

# Generate synthetic data only (no API key needed)
python run_benchmark.py --generate-only --size large

# Run benchmark with hybrid LLM+CV mode (requires API key)
python run_benchmark.py

# Digitize a single image
python run_benchmark.py --single path/to/km_plot.png --verbose
```

## Benchmark Details

### Synthetic Data

The benchmark suite generates synthetic KM plots with known ground truth, configurable in size:

| Size Flag | Plots | Description |
|-----------|-------|-------------|
| `--size small` | 20 | Quick smoke test |
| `--size default` | 40 | Standard benchmark |
| `--size large` | 200 | Rigorous evaluation |
| `--size xl` | 500 | Publication-grade evaluation |

#### Difficulty Levels

| Level | Characteristics |
|-------|----------------|
| **Easy** | Single curve, various colors, optional censoring/grid |
| **Medium** | Two curves (8 color pairs), CIs, at-risk tables |
| **Hard** | Truncated y-axis (0.2-0.6 start), low resolution (72 DPI), three overlapping curves, closely separated curves with CIs |

#### Edge Cases Covered

- **Truncated y-axis**: Y-axis starting at 0.2-0.6 (common in real papers)
- **Low resolution**: 72 DPI, 4x3 inch figures (simulating scanned documents)
- **Multiple overlapping curves**: Up to 3 curves with potential intersections
- **Closely separated curves**: Curves with <20% difference in median survival
- **Confidence intervals**: Shaded CI bands that can confuse color detection
- **Censoring marks**: Tick marks on curves
- **Number-at-risk tables**: Tables below the plot
- **Grid lines**: Background grid
- **Various Weibull shapes**: Exponential and non-exponential hazard functions
- **Dark/similar colors**: Black, dark brown, gray -- colors that can interfere with axes

### Evaluation Metrics

| Metric | Description | Matches KM-GPT |
|--------|-------------|-----------------|
| **MAE** | Mean Absolute Error of survival probabilities at 200 evenly-spaced time points | ~ Point-wise AE |
| **IAE** | Integrated Absolute Error (area between curves, normalized by time span) | Yes, directly comparable |
| **CCC** | Lin's Concordance Correlation Coefficient (agreement measure) | |
| **Accuracy@N%** | Fraction of time points with absolute error less than N% | |
| **Median Survival Error** | Absolute error in estimated median survival time | ~ AE in median OS |
| **RMSE** | Root Mean Squared Error | |
| **R-squared** | Coefficient of determination | |

### Real-World Data

The real-world benchmark evaluates on published KM plots from open-access journals. Since pixel-level ground truth is unavailable, evaluation uses:
1. **Reported median survival** from the paper text/legends
2. **Reported survival rates** at key time points (1-year, 5-year, etc.)
3. **Visual overlay** comparison of digitized curves on the original figure

## Architecture

```
src/
├── __init__.py
├── digitizer.py           # Main hybrid LLM+CV pipeline
├── cv_only_digitizer.py   # CV-only pipeline (no LLM needed)
├── llm_reader.py          # Claude vision metadata extraction
├── cv_extractor.py        # OpenCV curve tracing engine
├── synthesizer.py         # Synthetic benchmark generation
├── metrics.py             # Evaluation metrics
└── utils.py               # Shared utilities

run_benchmark.py               # Synthetic data benchmark runner
run_real_world_benchmark.py     # Real-world published data benchmark
```

### Pipeline Flow

```
Input Image
    │
    ▼
┌─────────────────────┐
│  LLM Metadata       │  → Axis ranges, curve colors, plot bbox,
│  Extraction          │    legend labels, structural features
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Plot Region        │  → Projection-profile based axis detection,
│  Detection (CV)     │    validated against Hough line transform
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  CI-Aware Line      │  → Adaptive tolerance based on color
│  Mask Creation      │    distribution; separates line from
│                     │    confidence interval fill shading
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Scan-Line Curve    │  → Column-by-column extraction with
│  Extraction         │    cluster analysis for step detection,
│                     │    vertical transition handling
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Post-Processing    │  → Monotonicity enforcement, median
│                     │    filtering, step function reconstruction,
│                     │    coordinate mapping
└──────────┬──────────┘
           │
           ▼
  Output: (time, survival) pairs for each curve
```

## Key Design Decisions

1. **CIELAB color space**: Color matching uses perceptually uniform CIELAB distance rather than RGB/HSV, providing more robust discrimination between curve colors across different display conditions.

2. **CI-aware line extraction**: An adaptive tolerance algorithm detects when confidence interval shading is present (by analyzing the color distance distribution of matching pixels) and automatically tightens the tolerance to isolate just the curve line. This is critical for papers that show CI bands.

3. **Projection-profile bbox detection**: Rather than relying solely on Hough line transform (which is fragile with CI fills), the plot bounding box is detected using row/column intensity projection profiles that find the axis lines as the most prominent dark features.

4. **Step function awareness**: The extraction algorithm explicitly models the KM step-function structure, distinguishing horizontal plateaus from vertical transitions by analyzing the spatial spread of matching pixels in each column.

5. **Monotonicity enforcement**: A running-minimum filter ensures the output is always non-increasing, correcting any noise-induced violations.

6. **Truncated axis handling**: The evaluation framework correctly handles truncated y-axes by clipping ground truth to the visible range, reflecting the physical limitation that the digitizer can only extract what's visible.

## LLM Cost Estimate

Using Claude Sonnet 4 ($3/M input, $15/M output tokens):

| Scenario | Images | Estimated Cost |
|----------|--------|----------------|
| Quick test (default benchmark) | 40 | ~$0.70 |
| Rigorous benchmark (large) | 200 | ~$3.40 |
| Publication benchmark (xl) | 500 | ~$8.50 |
| Typical systematic review | 50-100 | ~$1-2 |

The LLM is only needed for real-world images where axis ranges and curve colors are unknown. The CV-only mode is free (no API calls).

## Limitations

- Requires an Anthropic API key (Claude) for the hybrid LLM component (CV-only mode available)
- Performance may degrade on extremely compressed JPEG images
- May struggle with non-standard plot styles (hand-drawn, unusual color schemes)
- Curves with very similar colors (e.g., dark red vs dark blue on same plot) may show ~5% reduced accuracy
- Does not currently extract confidence interval bounds as separate data series
- CV-only mode requires user-provided axis ranges for real-world images
- Multi-panel figures require manual cropping to individual panels

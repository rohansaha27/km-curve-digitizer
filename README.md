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

Evaluated on 40 synthetic KM plots across three difficulty levels:

### Overall Performance (CV pipeline, ground truth axis/color hints)

| Metric | Value |
|--------|-------|
| **Mean MAE** | **0.0085** (±0.0037) |
| **Mean RMSE** | 0.0107 (±0.0046) |
| **Mean CCC** | **0.9969** (±0.0055) |
| **Mean R-squared** | 0.9940 (±0.0109) |
| **Accuracy @3%** | 97.8% |
| **Accuracy @5%** | **99.6%** |
| **Accuracy @10%** | **100.0%** |

### Performance by Difficulty

| Difficulty | Plots | Mean MAE | Acc @5% | Mean CCC |
|-----------|-------|----------|---------|----------|
| Easy (single curve, clean) | 8 | 0.0083 | 99.9% | 0.9989 |
| Medium (2 curves, CIs, various colors) | 13 | 0.0064 | 100.0% | 0.9992 |
| Hard (truncated y, low-res, 3 curves, overlap) | 19 | 0.0100 | 99.3% | 0.9946 |

All 40 plots achieve MAE < 0.025 and CCC > 0.96. The worst-case plot (hard_trunc_021) has MAE = 0.021.

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
# Generate synthetic benchmark data (no API key needed)
python run_benchmark.py --generate-only

# Run benchmark with CV-only mode (no API key needed)
python run_benchmark.py --cv-only

# Run benchmark with hybrid LLM+CV mode
python run_benchmark.py

# Digitize a single image
python run_benchmark.py --single path/to/km_plot.png --verbose

# Run with limited plots for quick testing
python run_benchmark.py --max-plots 5 --verbose
```

## Benchmark Details

The benchmark suite generates **40 synthetic KM plots** with known ground truth:

### Difficulty Levels

| Level | Count | Characteristics |
|-------|-------|----------------|
| **Easy** | 8 | Single curve, various colors (blue/red/green/orange/purple/brown/pink/cyan), optional censoring marks and grid |
| **Medium** | 13 | Two curves with 5 different color pairs, confidence intervals, number-at-risk tables, censoring marks |
| **Hard** | 19 | Truncated y-axis (starting at 0.2-0.5), low resolution (72 DPI at 4x3"), three overlapping curves, closely separated curves with CIs |

### Edge Cases Covered

- **Truncated y-axis**: Y-axis starting at 0.2-0.5 (common in real papers)
- **Low resolution**: 72 DPI, 4x3 inch figures (simulating scanned documents)
- **Multiple overlapping curves**: Up to 3 curves with potential intersections
- **Closely separated curves**: Curves with <15% difference in median survival
- **Confidence intervals**: Shaded CI bands that can confuse color detection
- **Censoring marks**: Tick marks on curves
- **Number-at-risk tables**: Tables below the plot
- **Grid lines**: Background grid
- **Various Weibull shapes**: Exponential and non-exponential hazard functions
- **Black curves**: That can interfere with axis detection
- **Brown/pastel colors**: That are harder to detect

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error of survival probabilities at 200 evenly-spaced time points |
| **RMSE** | Root Mean Squared Error |
| **IAE** | Integrated Absolute Error (area between curves, normalized) |
| **CCC** | Lin's Concordance Correlation Coefficient (agreement) |
| **R-squared** | Coefficient of determination |
| **Accuracy@N%** | Fraction of time points with absolute error less than N% |
| **Median Survival Error** | Absolute error in estimated median survival time |

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

## Limitations

- Requires an Anthropic API key (Claude) for the hybrid LLM component (CV-only mode available)
- Performance may degrade on extremely compressed JPEG images
- May struggle with non-standard plot styles (hand-drawn, unusual color schemes)
- Curves with very similar colors (e.g., light blue vs cyan) may be confused
- Does not currently extract confidence interval bounds as separate data series
- CV-only mode requires user-provided axis ranges for real-world images

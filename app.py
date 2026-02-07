#!/usr/bin/env python3
"""
KM Curve Digitizer — Streamlit Dashboard

Launch with:
    streamlit run app.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="KM Curve Digitizer",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SYNTH_DIR = ROOT / "data" / "synthetic"
RESULTS_DIR = ROOT / "data" / "results"
REAL_DIR = ROOT / "data" / "real_world"
REAL_RESULTS_DIR = ROOT / "data" / "results_real_world"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _benchmark_exists(directory: Path) -> bool:
    return directory.exists() and any(directory.glob("*.json"))


def _result_images(directory: Path):
    """Return comparison PNG paths sorted by name."""
    if not directory.exists():
        return []
    return sorted(directory.glob("*_comparison.png"))


def _run_command(cmd: list[str], placeholder) -> bool:
    """Run a subprocess, streaming output to a Streamlit placeholder."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(ROOT),
    )
    output_lines = []
    for line in proc.stdout:
        output_lines.append(line)
        # Show last 30 lines
        placeholder.code("".join(output_lines[-30:]), language="text")
    proc.wait()
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("KM Curve Digitizer")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Run Benchmark", "Synthetic Results", "Real-World Results"],
    index=0,
)

# ===================================================================
# PAGE: Overview
# ===================================================================
if page == "Overview":
    st.title("KM Curve Digitizer")
    st.markdown("""
    Automated extraction of survival data from Kaplan-Meier curve images
    using a **hybrid LLM + computer vision** pipeline.

    ---

    ### Quick Start

    Use the sidebar to navigate:

    | Page | What it does |
    |------|-------------|
    | **Run Benchmark** | Generate synthetic data and run the evaluation pipeline |
    | **Synthetic Results** | View metrics, charts, and per-plot comparisons from the synthetic benchmark |
    | **Real-World Results** | View digitization results on published clinical KM plots |
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        synth_ready = _benchmark_exists(RESULTS_DIR)
        st.metric("Synthetic Benchmark", "Ready" if synth_ready else "Not run yet")
    with col2:
        real_ready = _benchmark_exists(REAL_RESULTS_DIR)
        st.metric("Real-World Benchmark", "Ready" if real_ready else "Not run yet")
    with col3:
        n_synth = len(list(SYNTH_DIR.glob("*.png"))) if SYNTH_DIR.exists() else 0
        st.metric("Synthetic Plots Generated", n_synth)

    if synth_ready:
        report_path = RESULTS_DIR / "benchmark_report.txt"
        if report_path.exists():
            st.subheader("Latest Synthetic Benchmark Report")
            st.code(report_path.read_text(), language="text")


# ===================================================================
# PAGE: Run Benchmark
# ===================================================================
elif page == "Run Benchmark":
    st.title("Run Benchmark")

    tab_synth, tab_real = st.tabs(["Synthetic Benchmark", "Real-World Benchmark"])

    # --- Synthetic ---
    with tab_synth:
        st.markdown("Generate synthetic KM plots and evaluate the CV digitizer against known ground truth.")

        col1, col2 = st.columns(2)
        with col1:
            size = st.selectbox("Benchmark size", ["small (20)", "default (40)", "large (200)", "xl (500)"], index=1)
            size_flag = size.split(" ")[0]
        with col2:
            regen = st.checkbox("Regenerate data (delete existing)", value=False)

        if st.button("Run Synthetic Benchmark", type="primary"):
            if regen and SYNTH_DIR.exists():
                import shutil
                shutil.rmtree(SYNTH_DIR, ignore_errors=True)
                if RESULTS_DIR.exists():
                    shutil.rmtree(RESULTS_DIR, ignore_errors=True)
                st.info("Cleared existing data.")

            cmd = [
                sys.executable, "run_benchmark.py",
                "--cv-only", "--size", size_flag,
            ]
            if not regen and _benchmark_exists(SYNTH_DIR):
                cmd.append("--skip-generate")

            st.info(f"Running: `{' '.join(cmd)}`")
            placeholder = st.empty()
            t0 = time.time()
            ok = _run_command(cmd, placeholder)
            elapsed = time.time() - t0

            if ok:
                st.success(f"Benchmark completed in {elapsed:.1f}s")
                st.balloons()
            else:
                st.error("Benchmark failed. Check output above.")

    # --- Real-World ---
    with tab_real:
        st.markdown("Evaluate the digitizer on published KM plots from open-access papers.")

        has_real_images = REAL_DIR.exists() and any(REAL_DIR.glob("*.png"))
        if not has_real_images:
            st.warning("No real-world images found in `data/real_world/`. Run the download first.")
        else:
            n_images = len(list(REAL_DIR.glob("*.png")))
            st.info(f"Found {n_images} real-world KM plot images.")

        if st.button("Run Real-World Benchmark", type="primary", disabled=not has_real_images):
            cmd = [sys.executable, "run_real_world_benchmark.py"]
            st.info(f"Running: `{' '.join(cmd)}`")
            placeholder = st.empty()
            t0 = time.time()
            ok = _run_command(cmd, placeholder)
            elapsed = time.time() - t0

            if ok:
                st.success(f"Real-world benchmark completed in {elapsed:.1f}s")
            else:
                st.error("Benchmark failed. Check output above.")


# ===================================================================
# PAGE: Synthetic Results
# ===================================================================
elif page == "Synthetic Results":
    st.title("Synthetic Benchmark Results")

    if not _benchmark_exists(RESULTS_DIR):
        st.warning("No benchmark results found. Go to **Run Benchmark** first.")
        st.stop()

    # --- Summary Report ---
    report_path = RESULTS_DIR / "benchmark_report.txt"
    if report_path.exists():
        report_text = report_path.read_text()

        # Parse key metrics from report
        lines = report_text.split("\n")
        metrics = {}
        for line in lines:
            if "Mean MAE:" in line and "Median" not in line:
                metrics["Mean MAE"] = line.split(":")[1].strip().split(" ")[0]
            elif "Median IAE:" in line and "DIFFICULTY" not in report_text[report_text.index(line) - 100:report_text.index(line)]:
                try:
                    metrics["Median IAE"] = line.split(":")[1].strip().split(" ")[0]
                except Exception:
                    pass
            elif "Mean CCC:" in line and len(metrics) < 4:
                metrics["Mean CCC"] = line.split(":")[1].strip().split(" ")[0]
            elif "Accuracy (<5% err):" in line and len(metrics) < 5:
                metrics["Accuracy @5%"] = line.split(":")[1].strip()

        # Top-level metric cards
        cols = st.columns(4)
        metric_items = list(metrics.items())[:4]
        for i, (name, val) in enumerate(metric_items):
            with cols[i]:
                st.metric(name, val)

    # --- Aggregate Charts ---
    agg_path = RESULTS_DIR / "aggregate_performance.png"
    scatter_path = RESULTS_DIR / "per_plot_mae.png"

    if agg_path.exists() or scatter_path.exists():
        st.subheader("Performance Distribution")
        col1, col2 = st.columns(2)
        if agg_path.exists():
            with col1:
                st.image(str(agg_path), use_container_width=True)
        if scatter_path.exists():
            with col2:
                st.image(str(scatter_path), use_container_width=True)

    # --- Per-Plot Comparison Browser ---
    st.subheader("Per-Plot Comparison")
    comparisons = _result_images(RESULTS_DIR)
    if comparisons:
        # Build name list
        names = [p.stem.replace("_comparison", "") for p in comparisons]

        # Filter by difficulty
        all_difficulties = ["all"]
        for n in names:
            if n.startswith("easy"):
                d = "easy"
            elif n.startswith("medium"):
                d = "medium"
            elif n.startswith("hard"):
                d = "hard"
            else:
                d = "other"
            if d not in all_difficulties:
                all_difficulties.append(d)

        selected_diff = st.selectbox("Filter by difficulty", all_difficulties)

        filtered = []
        for name, path in zip(names, comparisons):
            if selected_diff == "all":
                filtered.append((name, path))
            elif name.startswith(selected_diff):
                filtered.append((name, path))

        if filtered:
            # Slider to browse
            idx = st.slider(
                "Plot",
                0,
                len(filtered) - 1,
                0,
                format=f"Plot %d of {len(filtered)}",
            )
            name, img_path = filtered[idx]

            # Load result JSON for this plot
            result_path = RESULTS_DIR / f"{name}_result.json"
            col_img, col_info = st.columns([3, 1])

            with col_img:
                st.image(str(img_path), caption=name, use_container_width=True)

            with col_info:
                st.markdown(f"**{name}**")
                if result_path.exists():
                    result = _load_json(result_path)
                    agg = result.get("aggregate", {})
                    st.metric("MAE", f"{agg.get('mean_mae', 0):.4f}")
                    st.metric("IAE", f"{agg.get('mean_iae', 0):.4f}")
                    st.metric("Acc @5%", f"{agg.get('mean_accuracy_5pct', 0)*100:.1f}%")
                    st.metric("CCC", f"{agg.get('mean_concordance_correlation', 0):.4f}")
                    st.metric("Curves", f"{agg.get('n_curves_detected', 0)}/{agg.get('n_curves_expected', 0)}")
    else:
        st.info("No comparison images found. Run the benchmark first.")

    # --- Full Report ---
    if report_path.exists():
        with st.expander("Full Benchmark Report"):
            st.code(report_text, language="text")


# ===================================================================
# PAGE: Real-World Results
# ===================================================================
elif page == "Real-World Results":
    st.title("Real-World Benchmark Results")

    if not _benchmark_exists(REAL_RESULTS_DIR):
        st.warning("No real-world results found. Go to **Run Benchmark** first.")
        st.stop()

    # Report
    report_path = REAL_RESULTS_DIR / "real_world_report.txt"
    if report_path.exists():
        report_text = report_path.read_text()

        # Summary metrics
        for line in report_text.split("\n"):
            if "Mean:" in line and "months" in line and "Median Survival" in report_text[:report_text.index(line)]:
                pass  # Let the report handle it

        st.subheader("Summary")
        st.code(report_text[:report_text.index("PER-PLOT DETAILS") if "PER-PLOT DETAILS" in report_text else len(report_text)], language="text")

    # Overlay images
    st.subheader("Overlay Visualizations")
    overlays = sorted(REAL_RESULTS_DIR.glob("*_overlay.png"))
    if overlays:
        for overlay in overlays:
            name = overlay.stem.replace("_overlay", "")
            st.markdown(f"**{name}**")
            st.image(str(overlay), use_container_width=True)
            st.divider()
    else:
        st.info("No overlay images found.")

    # Full report
    if report_path.exists():
        with st.expander("Full Report"):
            st.code(report_text, language="text")

    # JSON results
    json_path = REAL_RESULTS_DIR / "real_world_results.json"
    if json_path.exists():
        with st.expander("Raw JSON Results"):
            st.json(_load_json(json_path))

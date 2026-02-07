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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    ["Overview", "Run Benchmark", "Synthetic Results", "Real-World Results", "Digitize Image"],
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
    | **Digitize Image** | Upload your own KM plot and digitize it |
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


# ===================================================================
# PAGE: Digitize Image
# ===================================================================
elif page == "Digitize Image":
    st.title("Digitize a KM Plot")
    st.markdown("Upload a KM survival curve image and extract the survival data.")

    uploaded = st.file_uploader(
        "Upload a KM plot image",
        type=["png", "jpg", "jpeg", "gif", "webp"],
    )

    if uploaded:
        # Show the uploaded image
        st.image(uploaded, caption="Uploaded image", use_container_width=True)

        st.subheader("Extraction Settings")

        mode = st.radio(
            "Mode",
            ["CV-only (provide axis ranges manually)", "Hybrid LLM+CV (requires API key)"],
            index=0,
        )

        if "CV-only" in mode:
            col1, col2 = st.columns(2)
            with col1:
                x_min = st.number_input("X-axis min", value=0.0)
                x_max = st.number_input("X-axis max", value=48.0)
            with col2:
                y_min = st.number_input("Y-axis min", value=0.0)
                y_max = st.number_input("Y-axis max", value=1.0)

            color_tolerance = st.slider("Color tolerance", 20.0, 60.0, 35.0, 1.0)

            if st.button("Digitize", type="primary"):
                import tempfile
                import cv2
                from src.cv_only_digitizer import digitize_cv_only
                from src.utils import interpolate_step_function

                # Save uploaded file to temp
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name

                with st.spinner("Running CV extraction..."):
                    try:
                        results = digitize_cv_only(
                            tmp_path,
                            x_range=(x_min, x_max),
                            y_range=(y_min, y_max),
                            color_tolerance=color_tolerance,
                            verbose=False,
                        )

                        if results.get("curves"):
                            st.success(f"Detected {len(results['curves'])} curve(s)")

                            # Plot the extracted curves
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']

                            for i, curve in enumerate(results["curves"]):
                                t = np.array(curve["eval_times"])
                                s = np.array(curve["eval_survival"])
                                c = colors[i % len(colors)]
                                label = curve.get("label", f"Curve {i+1}")
                                ax.step(t, s, where="post", color=c, linewidth=2, label=label)

                            ax.set_xlabel("Time")
                            ax.set_ylabel("Survival Probability")
                            ax.set_title("Digitized Curves")
                            ax.legend()
                            ax.set_xlim(x_min, x_max)
                            ax.set_ylim(y_min, y_max)
                            ax.grid(True, alpha=0.2)
                            st.pyplot(fig)
                            plt.close(fig)

                            # Data table
                            st.subheader("Extracted Data")
                            for i, curve in enumerate(results["curves"]):
                                label = curve.get("label", f"Curve {i+1}")
                                with st.expander(f"Curve: {label}"):
                                    import pandas as pd
                                    df = pd.DataFrame({
                                        "Time": curve["eval_times"],
                                        "Survival": [f"{v:.4f}" for v in curve["eval_survival"]],
                                    })
                                    st.dataframe(df, use_container_width=True, height=300)

                                    # Download button
                                    csv = df.to_csv(index=False)
                                    st.download_button(
                                        f"Download CSV ({label})",
                                        csv,
                                        f"km_curve_{label}.csv",
                                        "text/csv",
                                    )
                        else:
                            st.warning("No curves detected. Try adjusting the color tolerance or axis ranges.")

                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)

        else:
            api_key = st.text_input("Anthropic API Key", type="password")
            if st.button("Digitize", type="primary"):
                if not api_key:
                    st.error("Please provide an API key.")
                else:
                    import tempfile
                    import os
                    os.environ["ANTHROPIC_API_KEY"] = api_key

                    from src.digitizer import KMDigitizer

                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp_path = tmp.name

                    with st.spinner("Running hybrid LLM+CV extraction..."):
                        try:
                            digitizer = KMDigitizer(verbose=False)
                            results = digitizer.digitize(tmp_path)

                            if results.get("curves"):
                                st.success(f"Detected {len(results['curves'])} curve(s)")

                                fig, ax = plt.subplots(figsize=(10, 6))
                                colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']

                                for i, curve in enumerate(results["curves"]):
                                    t = np.array(curve["eval_times"])
                                    s = np.array(curve["eval_survival"])
                                    c = colors[i % len(colors)]
                                    label = curve.get("label", f"Curve {i+1}")
                                    ax.step(t, s, where="post", color=c, linewidth=2, label=label)

                                ax.set_xlabel("Time")
                                ax.set_ylabel("Survival Probability")
                                ax.set_title("Digitized Curves")
                                ax.legend()
                                ax.grid(True, alpha=0.2)
                                st.pyplot(fig)
                                plt.close(fig)

                                # Metadata from LLM
                                if results.get("metadata"):
                                    with st.expander("LLM Metadata"):
                                        st.json(results["metadata"])

                                # Data table
                                st.subheader("Extracted Data")
                                for i, curve in enumerate(results["curves"]):
                                    label = curve.get("label", f"Curve {i+1}")
                                    with st.expander(f"Curve: {label}"):
                                        import pandas as pd
                                        df = pd.DataFrame({
                                            "Time": curve["eval_times"],
                                            "Survival": [f"{v:.4f}" for v in curve["eval_survival"]],
                                        })
                                        st.dataframe(df, use_container_width=True, height=300)
                                        csv = df.to_csv(index=False)
                                        st.download_button(
                                            f"Download CSV ({label})",
                                            csv,
                                            f"km_curve_{label}.csv",
                                            "text/csv",
                                        )
                            else:
                                st.warning("No curves detected.")

                        except Exception as e:
                            st.error(f"Error: {e}")
                        finally:
                            Path(tmp_path).unlink(missing_ok=True)

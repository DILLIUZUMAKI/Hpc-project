import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from hpc_optimizer.config import ROOT, BENCHMARKS, ExperimentConfig, cpu_limit, hardware_info
from hpc_optimizer.engine import Engine, load_source
from hpc_optimizer.llm import LLMClient, LLMConfig
from hpc_optimizer.sandbox import DockerSandbox, SandboxUnavailable
from hpc_optimizer.storage import Store, to_csv, report
from hpc_optimizer.charts import (
    chart, table, candidate_color,
    COLOR_SERIAL, COLOR_REFERENCE, COLOR_AI_BASE, COLOR_AI_ACCEPTED, COLOR_AI_REJECTED,
    COLOR_EMERALD, COLOR_MINT, COLOR_TEAL, COLOR_AMBER
)

load_dotenv(ROOT / ".env", override=False)

st.set_page_config(
    page_title="ParallelLab · HPC Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- MODERN, SPACIOUS, UNCLUSTERED CSS -----------------
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #F1F5F9;
}

code, pre, .stCodeBlock, [data-testid="stMetricValue"], [data-testid="stDataFrame"], .mono-font {
    font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace !important;
}

/* Deep Obsidian Canvas with subtle glow */
.stApp {
    background-color: #080B10 !important;
    background-image: radial-gradient(circle at 10% 0%, rgba(16, 185, 129, 0.05) 0%, transparent 28%),
                      radial-gradient(circle at 90% 0%, rgba(56, 189, 248, 0.04) 0%, transparent 28%) !important;
    color: #F1F5F9 !important;
}

/* Clean Custom Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #080B10;
}
::-webkit-scrollbar-thumb {
    background: #1A2433;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #10B981;
}

/* Completely remove the black top header bar & Streamlit toolbar */
header[data-testid="stHeader"],
[data-testid="stHeader"],
.stAppHeader {
    display: none !important;
    height: 0px !important;
    padding: 0px !important;
    background: transparent !important;
}

[data-testid="stToolbar"],
[data-testid="stDecoration"],
.stAppDeployButton,
#MainMenu,
footer {
    display: none !important;
    visibility: hidden !important;
}

/* Generous breathing room */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1400px;
}

/* Sleek Minimalist Sidebar */
[data-testid="stSidebar"] {
    background-color: #06080D !important;
    border-right: 1px solid #141C27 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #141C27 !important;
    margin: 0.85rem 0 !important;
}

/* Sidebar Brand Header */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 2px;
}
.sidebar-title {
    font-size: 17px;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.02em;
}
.sidebar-badge {
    font-size: 10px;
    font-weight: 700;
    color: #10B981;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 1px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
}
.sidebar-subtitle {
    font-size: 10.5px;
    color: #94A3B8;
    margin-bottom: 12px;
}

/* Hardware Telemetry Strip in Sidebar */
.telemetry-strip {
    background: #0D131C;
    border: 1px solid #182230;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 12px;
}
.telemetry-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    margin-bottom: 4px;
}
.telemetry-label {
    color: #94A3B8;
    font-weight: 500;
}
.telemetry-value {
    color: #F8FAFC;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}

/* Card Containers */
.hpc-card {
    background: #0E1521;
    border: 1px solid #1A2434;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 16px;
    transition: border-color 0.15s ease;
}
.hpc-card:hover {
    border-color: #243348;
}

/* Hero Header Banner */
.hero-banner {
    background: linear-gradient(135deg, #0F1725 0%, #0B111B 100%);
    border: 1px solid #1B2637;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}
.hero-category {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #10B981;
    text-transform: uppercase;
    margin-bottom: 3px;
}
.hero-heading {
    font-size: 20px;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.02em;
    margin: 0 0 3px 0;
}
.hero-sub {
    font-size: 12.5px;
    color: #94A3B8;
    margin: 0;
    line-height: 1.4;
}

/* Clean KPI Stats Cards */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 10px;
    margin-bottom: 16px;
}
.kpi-card {
    background: #0E1521;
    border: 1px solid #1A2434;
    border-radius: 8px;
    padding: 14px 16px;
    border-left: 3px solid #10B981;
}
.kpi-card-cyan { border-left-color: #38BDF8; }
.kpi-card-amber { border-left-color: #F59E0B; }
.kpi-card-emerald { border-left-color: #10B981; }

.kpi-title {
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 4px;
}
.kpi-number {
    font-size: 24px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #FFFFFF;
    margin-bottom: 3px;
}
.kpi-footer {
    font-size: 10.5px;
    font-family: 'JetBrains Mono', monospace;
    color: #10B981;
    display: flex;
    align-items: center;
    gap: 5px;
}
.kpi-footer-muted { color: #64748B; }

/* Status Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 9999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.badge-emerald {
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.4);
}
.badge-cyan {
    background: rgba(56, 189, 248, 0.14);
    color: #38BDF8;
    border: 1px solid rgba(56, 189, 248, 0.35);
}
.badge-amber {
    background: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}
.badge-rose {
    background: rgba(244, 63, 94, 0.15);
    color: #FB7185;
    border: 1px solid rgba(244, 63, 94, 0.35);
}
.badge-neutral {
    background: rgba(148, 163, 184, 0.1);
    color: #94A3B8;
    border: 1px solid rgba(148, 163, 184, 0.2);
}

/* Callout Box */
.hpc-callout {
    background: #0D131C;
    border: 1px solid #1A2434;
    border-left: 3px solid #10B981;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
}
.hpc-callout-amber { border-left-color: #F59E0B; }
.hpc-callout-rose { border-left-color: #F43F5E; }
.hpc-callout-head {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #10B981;
    margin-bottom: 3px;
}
.hpc-callout-amber .hpc-callout-head { color: #FBBF24; }
.hpc-callout-rose .hpc-callout-head { color: #FB7185; }
.hpc-callout-body {
    font-size: 12.5px;
    color: #E2E8F0;
    line-height: 1.45;
}

/* Clean Empty State */
.empty-box {
    background: #0E1521;
    border: 1px dashed #1E2B3E;
    border-radius: 12px;
    padding: 40px 24px;
    text-align: center;
    margin: 16px 0;
}
.empty-icon {
    font-size: 30px;
    margin-bottom: 8px;
    color: #10B981;
}
.empty-title {
    font-size: 15px;
    font-weight: 600;
    color: #F8FAFC;
    margin-bottom: 4px;
}
.empty-desc {
    font-size: 12.5px;
    color: #94A3B8;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.5;
}

/* Buttons */
.stButton > button[kind="primary"], .stButton > button:first-child[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 7px 16px !important;
    font-size: 13px !important;
    box-shadow: 0 2px 6px rgba(16, 185, 129, 0.25) !important;
    transition: all 0.15s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    box-shadow: 0 4px 10px rgba(16, 185, 129, 0.35) !important;
}
.stButton > button[kind="secondary"] {
    background: #0E1521 !important;
    border: 1px solid #1A2434 !important;
    color: #E2E8F0 !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    font-size: 12.5px !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #10B981 !important;
    color: #FFFFFF !important;
}

/* Modern Streamlit Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid #1A2434;
    padding-bottom: 2px;
    margin-bottom: 14px;
}
.stTabs [data-baseweb="tab"] {
    height: 36px;
    border-radius: 6px;
    padding: 0 14px;
    font-size: 12.5px;
    font-weight: 500;
    color: #94A3B8;
    background-color: transparent;
    border: none;
    transition: all 0.15s ease;
}
.stTabs [aria-selected="true"] {
    background-color: #131E2D !important;
    color: #10B981 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #10B981 !important;
}

/* Form Inputs */
div[data-baseweb="input"], div[data-baseweb="select"] > div {
    background-color: #0B1018 !important;
    border: 1px solid #1A2434 !important;
    border-radius: 6px !important;
    color: #FFFFFF !important;
}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {
    border-color: #10B981 !important;
    box-shadow: 0 0 0 1px #10B981 !important;
}
</style>""", unsafe_allow_html=True)


# ----------------- DATA STORE & SESSION STATE -----------------
store = Store(os.getenv("HPC_RESULTS_DIR", str(ROOT / "experiments")))
history = store.list()

# Check docker status once
if "docker_ready" not in st.session_state:
    try:
        DockerSandbox().preflight()
        st.session_state.docker_ready = True
    except Exception:
        st.session_state.docker_ready = False

# ----------------- SIDEBAR NAVIGATION -----------------
st.sidebar.markdown(f"""<div class="sidebar-brand">
    <span class="sidebar-title">⚡ ParallelLab</span>
    <span class="sidebar-badge">HPC</span>
</div>
<div class="sidebar-subtitle">Profiling & Optimization Workbench</div>

<div class="telemetry-strip">
    <div class="telemetry-row">
        <span class="telemetry-label">SANDBOX</span>
        <span class="badge {'badge-emerald' if st.session_state.docker_ready else 'badge-amber'}">
            ● {'READY' if st.session_state.docker_ready else 'CHECK DOCKER'}
        </span>
    </div>
    <div class="telemetry-row" style="margin-bottom:0;">
        <span class="telemetry-label">CPU BUDGET</span>
        <span class="telemetry-value">{cpu_limit()} CORES</span>
    </div>
</div>""", unsafe_allow_html=True)

# Standard page list maintaining compatibility with test_dashboard.py
PAGES = [
    "Overview",
    "Code & benchmarks",
    "AI generation",
    "Compilation & correctness",
    "Benchmarking",
    "Profiling",
    "Iterative optimization",
    "Results comparison",
    "Experiment history",
    "Documentation"
]

page = st.sidebar.radio("Workspace", PAGES, key="nav")
st.sidebar.divider()

# Active Experiment Selector in Sidebar
experiment = None
if history:
    all_ids = [e["id"] for e in history]
    pending = st.session_state.pop("pending_run", None)
    if pending in all_ids:
        st.session_state.selected_run = pending
    elif st.session_state.get("selected_run") not in all_ids:
        st.session_state.selected_run = all_ids[0]

    st.sidebar.markdown("<div style='font-size:11px;font-weight:600;letter-spacing:0.06em;color:#94A3B8;text-transform:uppercase;margin-bottom:6px;'>Active Experiment</div>", unsafe_allow_html=True)
    
    def format_run_label(rid):
        exp_item = next((e for e in history if e["id"] == rid), None)
        if not exp_item:
            return rid
        bench = exp_item.get("config", {}).get("benchmark", "run").upper()
        short_hash = rid.split("-")[-1][:6] if "-" in rid else rid[:6]
        created = exp_item.get("created_at", "")
        if "T" in created:
            time_part = created.split("T")[1][:5]
            date_part = created.split("T")[0][5:]
            time_str = f"{date_part} {time_part}"
        elif "T" in rid and "Z" in rid:
            t_str = rid.split("T")[1].split("Z")[0]
            time_str = f"{t_str[:2]}:{t_str[2:4]}"
        else:
            time_str = rid[:10]
        return f"{bench} · {time_str} ({short_hash})"

    selected_id = st.sidebar.selectbox(
        "Active experiment",
        all_ids,
        format_func=format_run_label,
        label_visibility="collapsed",
        key="selected_run"
    )
    experiment = next(e for e in history if e["id"] == selected_id)
    
    # Active run summary card in sidebar
    conf = experiment.get("config", {})
    meas = [m for c in experiment.get("candidates", []) for m in c.get("measurements", [])]
    peak_sp = max([m.get("speedup", 1.0) for m in meas], default=1.0)
    
    st.sidebar.markdown(f"""<div style="background:#0D131C;border:1px solid #1A2434;border-radius:8px;padding:9px 11px;margin-top:6px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">
            <span class="badge badge-emerald">{conf.get('benchmark', '').upper()}</span>
            <span class="badge badge-neutral">{conf.get('paradigm', '').upper()}</span>
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#94A3B8;margin-top:4px;">
            Peak Speedup: <b style="color:#10B981;">{peak_sp:.2f}x</b>
        </div>
    </div>""", unsafe_allow_html=True)
else:
    st.sidebar.info("No experiments yet. Start in Code & benchmarks.")


# ----------------- HELPER COMPONENTS -----------------
def render_header(title, subtitle, category="PARALLELLAB", exp=None):
    chips_html = ""
    if exp:
        c = exp.get("config", {})
        st_val = exp.get("status", "completed")
        badge_cls = "badge-emerald" if st_val == "completed" else "badge-rose" if "fail" in st_val else "badge-amber"
        chips_html = f"""<div style="display:flex;gap:6px;align-items:center;">
            <span class="badge badge-cyan">{c.get('benchmark', '').upper()}</span>
            <span class="badge badge-neutral">{c.get('paradigm', '').upper()}</span>
            <span class="badge {badge_cls}">● {st_val.upper()}</span>
        </div>"""
        
    st.markdown(f"""<div class="hero-banner">
        <div>
            <div class="hero-category">{category}</div>
            <h1 class="hero-heading">{title}</h1>
            <p class="hero-sub">{subtitle}</p>
        </div>
        {chips_html}
    </div>""", unsafe_allow_html=True)


def require_run():
    if experiment is None:
        st.info("No measured results yet. Open Code & benchmarks to run your first experiment.")
        st.stop()


def candidate_picker(key):
    candidates = experiment.get("candidates", []) if experiment else []
    if not candidates:
        st.info("This experiment has no candidates yet.")
        st.stop()
    selected = st.selectbox("Implementation", [c["id"] for c in candidates], key=key)
    return next(c for c in candidates if c["id"] == selected)


def show_chart(kind):
    fig = chart(experiment, kind)
    if fig is None:
        st.info("No valid measurements available for this chart.")
    else:
        st.plotly_chart(fig, use_container_width=True)


def metrics():
    candidates = experiment.get("candidates", []) if experiment else []
    compiled = [c for c in candidates if c.get("compilation")]
    passing = sum(c.get("compilation", {}).get("returncode") == 0 and not c.get("compilation", {}).get("failure") for c in compiled)
    valid = sum(c.get("status") == "valid" for c in candidates)
    ai = [c for c in candidates if c.get("origin", "").startswith("llm")]
    
    st.markdown(f"""<div class="kpi-container">
        <div class="kpi-card kpi-card-emerald">
            <div class="kpi-title">Compilation Pass</div>
            <div class="kpi-number">{passing}/{len(compiled)}</div>
            <div class="kpi-footer">● Build Verification</div>
        </div>
        <div class="kpi-card kpi-card-cyan">
            <div class="kpi-title">Oracle Verified</div>
            <div class="kpi-number">{valid}/{len(candidates)}</div>
            <div class="kpi-footer kpi-footer-muted">Numerical delta = 0</div>
        </div>
        <div class="kpi-card kpi-card-amber">
            <div class="kpi-title">AI Iterations</div>
            <div class="kpi-number">{max(0, len(ai) - 1) if ai else 0}</div>
            <div class="kpi-footer kpi-footer-muted">Feedback refinement</div>
        </div>
        <div class="kpi-card kpi-card-cyan">
            <div class="kpi-title">Active Selection</div>
            <div class="kpi-number" style="font-size:18px;padding-top:4px;">{experiment.get('best_ai') or 'reference'}</div>
            <div class="kpi-footer">▲ Peak scaling kernel</div>
        </div>
    </div>""", unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: OVERVIEW (DECLUTTERED & SIMPLE)
# ==============================================================================
if page == "Overview":
    render_header("Performance Workbench Overview",
                  "Containerized microarchitecture profiling, verified correctness oracles, and scaling benchmarks for OpenMP & MPI.",
                  "PARALLELLAB / OVERVIEW", experiment)
    
    # Clean system telemetry metrics
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.markdown(f"""<div class="kpi-card kpi-card-emerald">
            <div class="kpi-title">CPU Budget</div>
            <div class="kpi-number">{cpu_limit()} Cores</div>
            <div class="kpi-footer">Detected hardware concurrency</div>
        </div>""", unsafe_allow_html=True)
    with c_m2:
        st.markdown(f"""<div class="kpi-card kpi-card-cyan">
            <div class="kpi-title">Sandbox Environment</div>
            <div class="kpi-number">Docker Isolated</div>
            <div class="kpi-footer">Linux cgroup process limits</div>
        </div>""", unsafe_allow_html=True)
    with c_m3:
        st.markdown(f"""<div class="kpi-card kpi-card-amber">
            <div class="kpi-title">Total Experiments</div>
            <div class="kpi-number">{len(history)} Completed</div>
            <div class="kpi-footer">Stored locally in experiments/</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.5, 1.0], gap="large")
    
    with left:
        st.markdown("<div style='font-size:13px;font-weight:600;letter-spacing:0.04em;color:#94A3B8;text-transform:uppercase;margin-bottom:8px;'>Supported HPC Computational Kernels</div>", unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame([{"Workload": b["name"], "Default Size": f"{b['default']:,}", "Unit": b["unit"], "Bounds": f"{b['min']:,} - {b['max']:,}"} for b in BENCHMARKS.values()]),
            hide_index=True,
            use_container_width=True
        )
    
    with right:
        st.markdown("<div style='font-size:13px;font-weight:600;letter-spacing:0.04em;color:#94A3B8;text-transform:uppercase;margin-bottom:8px;'>Docker Sandbox Health</div>", unsafe_allow_html=True)
        st.markdown(f"""<div class="hpc-callout">
            <div class="hpc-callout-head">Sandbox Verification</div>
            <div class="hpc-callout-body">Preflight checks ensure GCC, OpenMPI, and optional profilers are accessible inside the execution sandbox container.</div>
        </div>""", unsafe_allow_html=True)

        if st.button("Check Docker environment", use_container_width=True):
            try:
                info = DockerSandbox().preflight()
                st.success("Sandbox is ready.")
                st.json(info)
            except Exception as exc:
                st.error(str(exc))

    if experiment:
        st.markdown("<div style='font-size:13px;font-weight:600;letter-spacing:0.04em;color:#94A3B8;text-transform:uppercase;margin-top:16px;margin-bottom:8px;'>Active Experiment Scaling Preview</div>", unsafe_allow_html=True)
        show_chart("runtime")


# ==============================================================================
# PAGE 2: CODE & BENCHMARKS (CLEAN, INTUITIVE RUNNER)
# ==============================================================================
elif page == "Code & benchmarks":
    render_header("Workload Configuration & Execution",
                  "Define computational kernels, concurrency scaling, and run container-isolated benchmarking.",
                  "CONFIGURATION", experiment)
    
    col_config, col_action = st.columns([1.5, 1.0], gap="large")
    
    with col_config:
        st.markdown("<div style='font-size:12.5px;font-weight:600;letter-spacing:0.05em;color:#10B981;text-transform:uppercase;margin-bottom:8px;'>1. Kernel & Parallel Model</div>", unsafe_allow_html=True)
        c_k1, c_k2 = st.columns([1.2, 1.0])
        with c_k1:
            benchmark = st.selectbox("Workload", list(BENCHMARKS), format_func=lambda x: BENCHMARKS[x]["name"])
        with c_k2:
            paradigm = st.radio("Parallel programming model", ["openmp", "mpi"], format_func=str.upper, horizontal=True)
            
        b = BENCHMARKS[benchmark]

        st.markdown("<div style='font-size:12.5px;font-weight:600;letter-spacing:0.05em;color:#10B981;text-transform:uppercase;margin-top:14px;margin-bottom:8px;'>2. Problem Size & Threads</div>", unsafe_allow_html=True)
        c_s1, c_s2 = st.columns([1.0, 1.2])
        with c_s1:
            size = st.number_input(f"Input size ({b['unit']})", min_value=b["min"], max_value=b["max"], value=b["default"], key="size_" + benchmark)
            steps = 20
            if benchmark == "jacobi":
                steps = st.number_input("Jacobi steps", 1, 1000, 20)
            else:
                steps = 20
        with c_s2:
            workers = [p for p in [1, 2, 4, 8, 16, 32] if p <= cpu_limit()]
            counts = st.multiselect("Thread / process counts", workers, default=workers[:min(3, len(workers))])

        # Advanced Settings in clean accordion (Zero clutter by default!)
        with st.expander("⚙️ Advanced Tuning & Profiler (Optional)", expanded=False):
            adv1, adv2 = st.columns(2)
            with adv1:
                repeats = st.number_input("Measured repetitions per configuration", 3, 30, 5)
                warmups = st.number_input("Warm-up runs", 0, 3, 1)
                seed = st.number_input("Deterministic seed", 0, 4294967295, 42)
            with adv2:
                timeout = st.number_input("Timeout per program run (seconds)", 1, 120, 30)
                memory = st.select_slider("Container memory limit (MB)", [512, 1024, 2048, 4096, 8192], value=1024)
                profiler = st.selectbox("Profiler", ["basic", "perf", "cachegrind"])
                threshold = st.slider("Minimum measured improvement to accept a revision (%)", 0.0, 20.0, 2.0, 0.5)

        # Custom C Source in clean accordion
        with st.expander("📄 Custom Serial C Replacement (Optional)", expanded=False):
            uploaded = st.file_uploader("Upload custom serial C file", type=["c"])
            serial_source = None
            if uploaded:
                try:
                    if uploaded.size > 256000:
                        raise ValueError("Maximum source size is 256 KB.")
                    serial_source = uploaded.getvalue().decode("utf-8")
                    st.success("Loaded custom source.")
                except (UnicodeError, ValueError) as exc:
                    st.error(str(exc))
                    st.stop()
            with st.expander("Preview serial C kernel", expanded=False):
                st.code(serial_source or load_source(benchmark, "serial"), language="c", line_numbers=True)

    with col_action:
        st.markdown("<div style='font-size:12.5px;font-weight:600;letter-spacing:0.05em;color:#94A3B8;text-transform:uppercase;margin-bottom:8px;'>AI Synthesis & Execution</div>", unsafe_allow_html=True)
        use_ai = st.toggle("Enable AI generation and optimization", value=False)
        
        env = LLMConfig.from_env()
        endpoint = env.base_url
        model = env.model
        key = env.api_key
        iterations = 1
        consent = False

        if use_ai:
            with st.container():
                endpoint = st.text_input("API base URL", value=env.base_url)
                model = st.text_input("Model ID", value=env.model)
                key = st.text_input("API key (session only)", value=env.api_key, type="password")
                iterations = st.slider("Optimization iterations", 0, 3, 1)
                consent = st.checkbox("I agree to transfer serial source and profile feedback to the API provider.")

        st.markdown(f"""<div class="hpc-card" style="padding:14px;margin-top:10px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:11.5px;color:#94A3B8;line-height:1.6;">
                • Container: <span style="color:#FFFFFF;">Disposable Linux Sandbox</span><br>
                • Network: <span style="color:#10B981;">Disabled (Air-gapped)</span><br>
                • CPU Limit: <span style="color:#FFFFFF;">{cpu_limit()} Cores</span><br>
                • Mode: <span style="color:{'#10B981' if use_ai else '#94A3B8'};">{'AI Feedback Optimization' if use_ai else 'Reference Benchmark'}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        if st.button("Run experiment", type="primary", use_container_width=True):
            if not counts:
                st.error("Please select at least one worker count.")
            else:
                try:
                    config = ExperimentConfig(
                        benchmark=benchmark,
                        paradigm=paradigm,
                        size=int(size),
                        steps=int(steps),
                        seed=int(seed) if 'seed' in locals() else 42,
                        counts=counts,
                        repeats=int(repeats) if 'repeats' in locals() else 5,
                        warmups=int(warmups) if 'warmups' in locals() else 1,
                        iterations=int(iterations) if 'iterations' in locals() else 1,
                        timeout=int(timeout) if 'timeout' in locals() else 30,
                        memory_mb=int(memory) if 'memory' in locals() else 1024,
                        profiler=profiler if 'profiler' in locals() else "basic",
                        min_improvement=(threshold / 100) if 'threshold' in locals() else 0.02,
                        use_ai=use_ai,
                        allow_source_transfer=consent
                    )
                    config.validate()
                    llm = LLMClient(LLMConfig(endpoint, model, key)) if use_ai else None
                    with st.status("Running experiment…", expanded=True) as status:
                        result = Engine(store, DockerSandbox(config.image), llm, progress=st.write).run(config, serial_source)
                        status.update(label="Experiment " + result["status"], state="complete" if result["status"] == "completed" else "error")
                    st.session_state.pending_run = result["id"]
                    if result["status"] == "completed":
                        st.success("Saved. Open Results comparison or Experiment history.")
                    else:
                        st.error(result.get("error", "A candidate failed; inspect the saved diagnostics."))
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))


# ==============================================================================
# PAGE 3: AI GENERATION
# ==============================================================================
elif page == "AI generation":
    render_header("AI Code Generation & Inspector",
                  "Inspect synthesized parallel kernels side-by-side with serial baseline and curated reference code.",
                  "AI GENERATION", experiment)
    require_run()
    
    ai = [c for c in experiment["candidates"] if c.get("origin", "").startswith("llm")]
    serial_cand = next((c for c in experiment["candidates"] if c["id"] == "serial"), None)
    ref_cand = next((c for c in experiment["candidates"] if c["id"] == "reference"), None)
    ai_base_cand = next((c for c in experiment["candidates"] if c["id"] == "ai_baseline"), None)
    ai_iterations = [c for c in experiment["candidates"] if c.get("origin") == "llm_feedback"]

    if not ai:
        st.markdown("""<div class="empty-box">
            <div class="empty-icon">⚙</div>
            <div class="empty-title">Reference-Only Run</div>
            <div class="empty-desc">This experiment contains no AI-generated code. Enable AI generation in <b>Code & benchmarks</b> to trigger automated parallel kernel synthesis.</div>
        </div>""", unsafe_allow_html=True)
    else:
        insp_mode = st.radio("Inspector View", ["Side-by-Side Comparison", "Focused Single Candidate"], horizontal=True, key="ai_insp_mode")
        
        if insp_mode == "Side-by-Side Comparison":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"#### 1. Serial Baseline", unsafe_allow_html=True)
                if serial_cand:
                    st.caption(f"Lines: {len(serial_cand['source'].splitlines())}")
                    st.code(serial_cand["source"], language="c", line_numbers=True)
                    st.download_button("Download serial.c", serial_cand["source"], "serial.c", "text/x-c", use_container_width=True)
            with col2:
                st.markdown(f"#### 2. AI Baseline", unsafe_allow_html=True)
                if ai_base_cand:
                    st.caption(f"Lines: {len(ai_base_cand['source'].splitlines())}")
                    st.code(ai_base_cand["source"], language="c", line_numbers=True)
                    st.download_button("Download ai_baseline.c", ai_base_cand["source"], "ai_baseline.c", "text/x-c", use_container_width=True)
                    if ai_base_cand.get("explanation"):
                        with st.expander("Model Strategy"):
                            st.write(ai_base_cand["explanation"])
            with col3:
                if ai_iterations:
                    selected_iter = st.selectbox("Select Iteration", [c["id"] for c in ai_iterations], key="side_iter_sel")
                    rev_cand = next(c for c in ai_iterations if c["id"] == selected_iter)
                    st.markdown(f"#### 3. Revision: `{selected_iter}`")
                    st.caption(f"Lines: {len(rev_cand['source'].splitlines())}")
                    st.code(rev_cand["source"], language="c", line_numbers=True)
                    st.download_button(f"Download {selected_iter}.c", rev_cand["source"], f"{selected_iter}.c", "text/x-c", use_container_width=True)
                elif ref_cand:
                    st.markdown(f"#### 3. Curated Reference")
                    st.caption(f"Lines: {len(ref_cand['source'].splitlines())}")
                    st.code(ref_cand["source"], language="c", line_numbers=True)
                    st.download_button("Download reference.c", ref_cand["source"], "reference.c", "text/x-c", use_container_width=True)
        else:
            candidate = candidate_picker("ai_code")
            st.markdown(f"### `{candidate['id']}` · Status: {candidate['status'].upper()}")
            if candidate.get("explanation"):
                st.markdown(f"""<div class="hpc-callout">
                    <div class="hpc-callout-head">Model Strategy & Architecture</div>
                    <div class="hpc-callout-body">{candidate['explanation']}</div>
                </div>""", unsafe_allow_html=True)
            st.code(candidate["source"], language="c", line_numbers=True)
            st.download_button(f"Download {candidate['id']}.c", candidate["source"], candidate["id"] + ".c", "text/x-c")


# ==============================================================================
# PAGE 4: COMPILATION & CORRECTNESS
# ==============================================================================
elif page == "Compilation & correctness":
    render_header("Compilation & Numerical Correctness",
                  "Automated GCC build logs, independent Python oracle full-vector comparison, and thread race detection.",
                  "VERIFICATION", experiment)
    require_run()
    metrics()
    
    candidate = candidate_picker("correctness")
    comp = candidate.get("compilation", {"status": "not_run"})
    is_comp_ok = comp.get("returncode") == 0 and not comp.get("failure")
    checks = candidate.get("correctness", [])
    
    # 2 Summary Cards
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(f"""<div class="hpc-card" style="padding:16px;">
            <div style="font-size:11px;font-weight:600;color:#94A3B8;text-transform:uppercase;">Compiler Status</div>
            <div style="font-size:22px;font-weight:700;color:{'#10B981' if is_comp_ok else '#FB7185'};margin:4px 0;">
                {'● BUILD PASSED' if is_comp_ok else '⚠ BUILD FAILED'}
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#94A3B8;">Build Time: {comp.get('seconds', 0):.3f}s</div>
        </div>""", unsafe_allow_html=True)
    with sc2:
        n_passed = sum(1 for c in checks if c.get("passed"))
        st.markdown(f"""<div class="hpc-card" style="padding:16px;">
            <div style="font-size:11px;font-weight:600;color:#94A3B8;text-transform:uppercase;">Numerical Oracle Verification</div>
            <div style="font-size:22px;font-weight:700;color:{'#10B981' if n_passed == len(checks) and checks else '#FB7185'};margin:4px 0;">
                {f'{n_passed}/{len(checks)} PASSED' if checks else 'NO CHECKS'}
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#94A3B8;">Tolerance: 1e-8 absolute</div>
        </div>""", unsafe_allow_html=True)

    # Correctness matrix
    if checks:
        st.markdown("<div style='font-size:13px;font-weight:600;letter-spacing:0.04em;color:#94A3B8;text-transform:uppercase;margin-top:10px;margin-bottom:8px;'>Test Case Verification Matrix</div>", unsafe_allow_html=True)
        matrix_rows = []
        for idx, c in enumerate(checks, start=1):
            r_info = c.get("run", {})
            sec = r_info.get("seconds")
            matrix_rows.append({
                "Test": f"#{idx:02d}",
                "Workers": c.get("workers", 1),
                "Size": c.get("size", "—"),
                "Result": "PASSED" if c.get("passed") else "FAILED",
                "Execution Time": f"{sec*1000:.2f} ms" if sec else "—",
                "Detail": c.get("detail", "Numerical comparison passed.")
            })
        st.dataframe(pd.DataFrame(matrix_rows), hide_index=True, use_container_width=True)

    with st.expander("🔍 Compiler Logs & Diagnostic Streams", expanded=False):
        st.markdown("**Build Command:**")
        cmd_str = " ".join(comp.get("command", [])) if isinstance(comp.get("command"), list) else str(comp.get("command", "None"))
        st.code(cmd_str, language="bash")
        if comp.get("stderr"):
            st.markdown("**Standard Error:**")
            st.code(comp["stderr"], language="text")


# ==============================================================================
# PAGE 5: BENCHMARKING
# ==============================================================================
elif page == "Benchmarking":
    render_header("Performance Telemetry & Scaling Benchmarks",
                  "Warmups are excluded. Every measured sample is saved. Error bars show sample standard deviation.",
                  "BENCHMARKING", experiment)
    require_run()
    
    b_tab1, b_tab2, b_tab3 = st.tabs(["Median Wall Time Scaling", "Speedup vs Linear Reference (S=p)", "Run-to-Run Variance"])
    with b_tab1:
        show_chart("runtime")
    with b_tab2:
        show_chart("speedup")
    with b_tab3:
        show_chart("variance")
        
    st.markdown("<div style='font-size:13px;font-weight:600;letter-spacing:0.04em;color:#94A3B8;text-transform:uppercase;margin-top:16px;margin-bottom:8px;'>Configuration Measurement Table</div>", unsafe_allow_html=True)
    st.dataframe(table(experiment), hide_index=True, use_container_width=True)


# ==============================================================================
# PAGE 6: PROFILING
# ==============================================================================
elif page == "Profiling":
    render_header("Microarchitecture Profiling & Event Counters",
                  "Hardware performance counters, cache miss rates, and kernel vs communication timing breakdowns.",
                  "PROFILING", experiment)
    require_run()
    candidate = candidate_picker("profile")
    
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        fig_bd = chart(experiment, "profile_breakdown")
        if fig_bd:
            st.plotly_chart(fig_bd, use_container_width=True)
        else:
            show_chart("runtime")
    with p_col2:
        fig_hw = chart(experiment, "profile_counters")
        if fig_hw:
            st.plotly_chart(fig_hw, use_container_width=True)
        else:
            show_chart("variance")

    analysis = candidate.get("analysis", {})
    botts = analysis.get("observed_bottlenecks", [])
    suggs = analysis.get("optimization_suggestions", [])
    
    if botts or suggs:
        col_an1, col_an2 = st.columns(2)
        with col_an1:
            st.markdown("#### Observed Diagnostics")
            for b in (botts or ["No automated bottlenecks detected."]):
                st.markdown(f"""<div class="hpc-callout hpc-callout-amber">
                    <div class="hpc-callout-body">{b}</div>
                </div>""", unsafe_allow_html=True)
        with col_an2:
            st.markdown("#### Recommendations")
            for s in (suggs or ["No specific suggestions recorded."]):
                st.markdown(f"""<div class="hpc-callout">
                    <div class="hpc-callout-body">{s}</div>
                </div>""", unsafe_allow_html=True)


# ==============================================================================
# PAGE 7: ITERATIVE OPTIMIZATION
# ==============================================================================
elif page == "Iterative optimization":
    render_header("Iterative Search & Candidate Trajectory",
                  "Profiling feedback drives multi-turn code revisions. Selection minimizes geometric-mean runtime across tested worker counts.",
                  "ITERATIVE SEARCH", experiment)
    require_run()
    
    show_chart("iterations")
    
    ai_feedback_candidates = [c for c in experiment.get("candidates", []) if c.get("origin") == "llm_feedback"]
    if not ai_feedback_candidates:
        st.info("No iterative feedback revisions in this experiment. Enable AI mode with >= 1 iteration in Code & benchmarks.")
    else:
        for cand in ai_feedback_candidates:
            is_acc = cand.get("accepted", False)
            with st.expander(f"{cand['id']} · {'▲ ACCEPTED' if is_acc else '✕ REJECTED'}", expanded=True):
                st.markdown(f"**Decision**: `{cand.get('decision', 'N/A')}`")
                if cand.get("explanation"):
                    st.write(cand["explanation"])
                st.code(cand["source"], language="c", line_numbers=True)


# ==============================================================================
# PAGE 8: RESULTS COMPARISON
# ==============================================================================
elif page == "Results comparison":
    render_header("Performance Comparison & Benchmark Matrix",
                  "Comprehensive multi-dimensional evaluation: speedup scaling, parallel efficiency, run-to-run noise, and candidate exports.",
                  "RESULTS MATRIX", experiment)
    require_run()
    metrics()
    
    c1, c2 = st.columns(2)
    with c1:
        show_chart("runtime")
    with c2:
        show_chart("speedup")
        
    st.markdown("<div style='font-size:13px;font-weight:600;letter-spacing:0.04em;color:#94A3B8;text-transform:uppercase;margin-top:14px;margin-bottom:8px;'>Implementation Roster</div>", unsafe_allow_html=True)
    roster_rows = []
    for c in experiment.get("candidates", []):
        meas = c.get("measurements", [])
        best_sp = max([m.get("speedup", 1.0) for m in meas], default=1.0)
        roster_rows.append({
            "Implementation": c["id"],
            "Origin": c.get("origin", "—"),
            "Status": c.get("status", "—"),
            "Peak Speedup": f"{best_sp:.2f}x" if meas else "—",
            "Lines": len(c.get("source", "").splitlines())
        })
    st.dataframe(pd.DataFrame(roster_rows), hide_index=True, use_container_width=True)
    
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.download_button("Export CSV", to_csv(experiment), experiment["id"] + ".csv", "text/csv", use_container_width=True)
    b.download_button("Export full JSON", json.dumps(experiment, indent=2), experiment["id"] + ".json", "application/json", use_container_width=True)
    c.download_button("Export report", report(experiment), experiment["id"] + ".md", "text/markdown", use_container_width=True)


# ==============================================================================
# PAGE 9: EXPERIMENT HISTORY
# ==============================================================================
elif page == "Experiment history":
    render_header("Experiment Run Archive & Telemetry Log",
                  "Chronological archive of benchmarking sessions, active candidate versions, and detailed event traces.",
                  "HISTORY", experiment)
    if not history:
        st.info("Your first experiment will appear here.")
    else:
        st.dataframe(
            pd.DataFrame([{"ID": e["id"], "Date": e.get("created_at", "")[:19].replace("T", " "), "Workload": e["config"]["benchmark"].upper(),
                           "Model": e["config"]["paradigm"].upper(), "Status": e["status"].upper(), "Best AI": e.get("best_ai") or "reference"} for e in history]),
            hide_index=True,
            use_container_width=True
        )
        
        c_hist1, c_hist2 = st.columns(2)
        with c_hist1:
            st.markdown(f"""<div class="kpi-card kpi-card-emerald">
                <div class="kpi-title">Total Experiments Executed</div>
                <div class="kpi-number">{len(history)}</div>
            </div>""", unsafe_allow_html=True)
        with c_hist2:
            unique_w = len({e["config"]["benchmark"] for e in history})
            st.markdown(f"""<div class="kpi-card kpi-card-cyan">
                <div class="kpi-title">Distinct Workloads Evaluated</div>
                <div class="kpi-number">{unique_w}</div>
            </div>""", unsafe_allow_html=True)


# ==============================================================================
# PAGE 10: DOCUMENTATION
# ==============================================================================
else:
    render_header("Technical Documentation & Guides",
                  "Methodological principles, contract requirements, sandboxing architecture, and troubleshooting procedures.",
                  "DOCUMENTATION", experiment)
    doc = st.selectbox("Guide", ["QUICKSTART.md", "docs/METHODOLOGY.md", "docs/PROGRAM_CONTRACT.md", "docs/SECURITY.md", "docs/TROUBLESHOOTING.md", "docs/VALIDATION.md"])
    path = ROOT / doc
    if path.exists():
        st.markdown(f"""<div class="hpc-card" style="padding:22px;">
            {path.read_text()}
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("Documentation file not found.")

import os
import sys
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

# -------------------------------------------------------------------
# Make `src/` importable when Streamlit runs from app/ folder
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run as run_pipeline  # noqa: E402


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "retail_sales_dataset.csv"
DEFAULT_OUT = PROJECT_ROOT / "outputs"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "insights.md"
DEFAULT_FIGS = PROJECT_ROOT / "reports" / "figures"

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def safe_read_csv(path: Path, parse_dates=None) -> pd.DataFrame | None:
    try:
        if not path.exists():
            return None
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception as e:
        st.error(f"Failed reading {path.name}: {e}")
        return None


def safe_read_json(path: Path) -> dict | None:
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed reading {path.name}: {e}")
        return None


def fmt_money(x: float) -> str:
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)


# -------------------------------------------------------------------
# App
# -------------------------------------------------------------------
st.set_page_config(page_title="Retail Calendar Pattern Finder", layout="wide")

st.title("Retail Calendar Pattern Finder")
st.caption("Seasonality + spike days + driver explanations (transactions, units, AOV, category mix).")

with st.sidebar:
    st.header("Run pipeline")

    # Option: file upload (nice UX)
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    input_path = st.text_input(
        "Or CSV path",
        value=str(DEFAULT_INPUT),
        help="If you upload a file above, it will be used instead of this path.",
    )

    top_n = st.slider("Top spike days", min_value=5, max_value=50, value=15, step=5)

    run_btn = st.button("Run / Refresh Pipeline")


# If user uploaded a CSV, save it to a temp file under project outputs (so pipeline can read it)
effective_input_path = Path(input_path)

if uploaded is not None:
    temp_dir = PROJECT_ROOT / "data" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / "uploaded_retail_sales.csv"
    temp_path.write_bytes(uploaded.getbuffer())
    effective_input_path = temp_path

if run_btn:
    with st.spinner("Running pipeline..."):
        run_pipeline(
            input_path=str(effective_input_path),
            out_dir=str(DEFAULT_OUT),
            report_path=str(DEFAULT_REPORT),
            figures_dir=str(DEFAULT_FIGS),
            top_n=top_n,
        )
    st.success("Done! Outputs + report generated.")


# -------------------------------------------------------------------
# Load outputs
# -------------------------------------------------------------------
daily_path = DEFAULT_OUT / "daily_metrics.csv"
scored_path = DEFAULT_OUT / "daily_scored.csv"
spikes_path = DEFAULT_OUT / "spike_days.csv"
mix_path = DEFAULT_OUT / "daily_category_mix.csv"
cards_path = DEFAULT_OUT / "spike_cards.json"
quality_path = DEFAULT_OUT / "quality_summary.json"

daily = safe_read_csv(daily_path, parse_dates=["date"])
scored = safe_read_csv(scored_path, parse_dates=["date"])
spikes = safe_read_csv(spikes_path, parse_dates=["date"])
mix = safe_read_csv(mix_path, parse_dates=["date"])
cards = safe_read_json(cards_path) or []
quality = safe_read_json(quality_path)

if daily is None or scored is None or spikes is None or mix is None:
    st.info("Run the pipeline from the sidebar to generate outputs.")
    st.stop()


# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
tab_overview, tab_seasonality, tab_spikes, tab_quality, tab_report = st.tabs(
    ["Overview", "Seasonality", "Spike Explorer", "Data Quality", "Report"]
)

# =========================
# Overview
# =========================
with tab_overview:
    total_revenue = float(daily["revenue"].sum())
    total_txns = int(daily["txns"].sum())
    total_units = float(daily["units"].sum())
    avg_aov = float(daily["aov"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total revenue", fmt_money(total_revenue))
    c2.metric("Total transactions", f"{total_txns:,}")
    c3.metric("Total units", fmt_money(total_units))
    c4.metric("Avg AOV", fmt_money(avg_aov))

    st.subheader("Daily revenue trend")
    fig = px.line(daily.sort_values("date"), x="date", y="revenue")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Daily residuals (actual − expected)")
    fig2 = px.scatter(scored.sort_values("date"), x="date", y="residual", hover_data=["dow", "month"])
    fig2.add_hline(y=0)
    st.plotly_chart(fig2, width="stretch")


# =========================
# Seasonality
# =========================
with tab_seasonality:
    st.subheader("Day-of-week patterns")

    dow = (
        daily.groupby("dow", as_index=False)
        .agg(
            avg_daily_revenue=("revenue", "mean"),
            median_daily_revenue=("revenue", "median"),
            avg_txns_per_day=("txns", "mean"),
            avg_units_per_day=("units", "mean"),
            avg_aov=("aov", "mean"),
            n_days=("date", "count"),
        )
    )
    dow["dow"] = pd.Categorical(dow["dow"], categories=DOW_ORDER, ordered=True)
    dow = dow.sort_values("dow")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Average daily revenue by day-of-week**")
        st.plotly_chart(px.bar(dow, x="dow", y="avg_daily_revenue"), width="stretch")

    with colB:
        st.markdown("**Median daily revenue by day-of-week**")
        st.plotly_chart(px.bar(dow, x="dow", y="median_daily_revenue"), width="stretch")

    st.subheader("Monthly patterns (observed transaction-days only)")

    month = (
        daily.groupby("month", as_index=False)
        .agg(revenue=("revenue", "sum"), n_days=("date", "count"), aov=("aov", "mean"))
        .sort_values("month")
    )

    colC, colD = st.columns(2)
    with colC:
        st.markdown("**Monthly revenue**")
        st.plotly_chart(px.bar(month, x="month", y="revenue"), width="stretch")

    with colD:
        st.markdown("**Monthly average AOV**")
        st.plotly_chart(px.line(month, x="month", y="aov", markers=True), width="stretch")

    st.subheader("Month × Day-of-week heatmap (avg daily revenue)")
    heat = daily.copy()
    heat["dow"] = pd.Categorical(heat["dow"], categories=DOW_ORDER, ordered=True)

    # Fix FutureWarning by making observed explicit
    pivot = heat.pivot_table(
        index="month",
        columns="dow",
        values="revenue",
        aggfunc="mean",
        observed=False,   # <- important
    )

    pivot_reset = pivot.reset_index().melt(id_vars="month", var_name="dow", value_name="avg_revenue").dropna()
    fig_h = px.density_heatmap(pivot_reset, x="dow", y="month", z="avg_revenue")
    st.plotly_chart(fig_h, width="stretch")


# =========================
# Spike Explorer
# =========================
with tab_spikes:
    st.subheader("Top spike days (ranked by residual z-score)")

    st.dataframe(spikes, width="stretch", hide_index=True)

    spike_dates = spikes["date"].dt.strftime("%Y-%m-%d").tolist()
    if not spike_dates:
        st.warning("No spikes found (unexpected). Try increasing top_n and rerun pipeline.")
        st.stop()

    chosen = st.selectbox("Choose a spike date", spike_dates)
    chosen_date = pd.to_datetime(chosen)

    row = scored[scored["date"] == chosen_date]
    if row.empty:
        st.error("Selected date not found in scored daily table.")
        st.stop()
    row = row.iloc[0]

    # Baselines for quick driver deltas
    baseline_txns = float(scored["txns"].mean())
    baseline_units = float(scored["units"].mean())
    baseline_aov = float(scored["aov"].mean())

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Revenue", fmt_money(row["revenue"]), f"{row['residual']:,.0f} vs expected")
    d2.metric("Transactions", f"{row['txns']:.0f}", f"{row['txns']-baseline_txns:+.1f} vs mean")
    d3.metric("Units", f"{row['units']:.0f}", f"{row['units']-baseline_units:+.1f} vs mean")
    d4.metric("AOV", fmt_money(row["aov"]), f"{row['aov']-baseline_aov:+.1f} vs mean")

    st.markdown("### Category contribution (revenue)")
    day_mix = mix[mix["date"] == chosen_date].sort_values("category_revenue", ascending=False)
    if day_mix.empty:
        st.info("No category mix rows found for this date.")
    else:
        st.plotly_chart(px.bar(day_mix, x="product_category", y="category_revenue"), width="stretch")

    st.markdown("### Spike Card (export)")
    card = next((c for c in cards if c.get("date") == chosen), None)
    if card:
        st.code(json.dumps(card, indent=2), language="json")
    else:
        st.info("Spike card not found (rerun pipeline to regenerate spike_cards.json).")


# =========================
# Data Quality
# =========================
with tab_quality:
    st.subheader("Quality Summary")
    if quality:
        st.json(quality)
    else:
        st.info("quality_summary.json not found. Rerun pipeline.")


# =========================
# Report
# =========================
with tab_report:
    st.subheader("Generated Report (Markdown)")
    if DEFAULT_REPORT.exists():
        report_text = DEFAULT_REPORT.read_text(encoding="utf-8")
        st.markdown(report_text)
        st.caption("Note: Markdown images may not render inline depending on local paths; you can open reports/insights.md directly too.")
    else:
        st.info("Report not found. Run pipeline first.")

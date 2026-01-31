from __future__ import annotations

import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from .io import load_retail_csv
from .quality_checks import basic_sanity_checks, check_total_amount_identity
from .aggregate_daily import daily_metrics, daily_category_mix
from .aggregate_weekly import weekly_metrics
from .seasonality import dow_summary, month_summary, month_dow_heatmap_data, DOW_ORDER
from .baseline import expected_revenue_month_dow
from .spike_detection import add_spike_scores, top_spike_days
from .spike_explain import build_spike_cards, save_cards
from .reporting import write_markdown_report

def _save_fig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def _plot_dow(dow_df: pd.DataFrame, out_path: str):
    x = dow_df["dow"].astype(str).tolist()
    y = dow_df["avg_daily_revenue"].to_numpy()
    plt.figure(figsize=(10,4))
    plt.bar(x, y)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Avg daily revenue")
    plt.title("Average Daily Revenue by Day-of-Week")
    _save_fig(out_path)

def _plot_month(month_df: pd.DataFrame, out_path: str):
    x = month_df["month"].astype(str).tolist()
    y = month_df["revenue"].to_numpy()
    plt.figure(figsize=(10,4))
    plt.bar(x, y)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Total revenue (observed days only)")
    plt.title("Monthly Revenue (Observed Transaction-Days)")
    _save_fig(out_path)

def _plot_heatmap(mean_pivot: pd.DataFrame, out_path: str):
    # simple matplotlib imshow
    plt.figure(figsize=(10,6))
    data = mean_pivot.to_numpy()
    plt.imshow(data, aspect="auto")
    plt.yticks(range(len(mean_pivot.index)), mean_pivot.index.tolist())
    plt.xticks(range(len(mean_pivot.columns)), [str(c) for c in mean_pivot.columns], rotation=30, ha="right")
    plt.colorbar(label="Avg daily revenue")
    plt.title("Month × Day-of-Week Heatmap (Avg Daily Revenue)")
    _save_fig(out_path)

def _plot_spike_scatter(scored: pd.DataFrame, out_path: str):
    plt.figure(figsize=(10,4))
    x = pd.to_datetime(scored["date"])
    y = scored["residual"].to_numpy()
    plt.plot(x, y, marker="o", linestyle="")
    plt.axhline(0, linewidth=1)
    plt.ylabel("Residual (actual - expected)")
    plt.title("Spike Residuals Over Time")
    plt.xticks(rotation=30, ha="right")
    _save_fig(out_path)

def run(input_path: str, out_dir: str, report_path: str, figures_dir: str, top_n: int = 15):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    df = load_retail_csv(input_path)

    # Quality checks
    sanity = basic_sanity_checks(df)
    identity = check_total_amount_identity(df)
    mismatches = int(identity["is_mismatch"].sum())

    # Aggregations
    daily = daily_metrics(df)
    weekly = weekly_metrics(df)
    mix = daily_category_mix(df)

    # Seasonality
    dow = dow_summary(daily)
    month = month_summary(daily)

    # Heatmap data
    mean_pivot, cnt_pivot = month_dow_heatmap_data(daily)

    # Baseline + spikes
    expected = expected_revenue_month_dow(daily, min_cell_days=2)
    scored = add_spike_scores(daily, expected)
    spikes = top_spike_days(scored, n=top_n, by="robust_z_residual")

    # Spike cards
    cards = build_spike_cards(scored, mix, top_n=top_n, score_col="robust_z_residual")

    # Save outputs
    daily.to_csv(os.path.join(out_dir, "daily_metrics.csv"), index=False)
    weekly.to_csv(os.path.join(out_dir, "weekly_metrics.csv"), index=False)
    mix.to_csv(os.path.join(out_dir, "daily_category_mix.csv"), index=False)
    scored.to_csv(os.path.join(out_dir, "daily_scored.csv"), index=False)
    spikes.to_csv(os.path.join(out_dir, "spike_days.csv"), index=False)
    save_cards(cards, os.path.join(out_dir, "spike_cards.json"))

    # Figures
    fig_paths = {
        "dow_revenue": os.path.join(figures_dir, "dow_revenue.png"),
        "month_revenue": os.path.join(figures_dir, "month_revenue.png"),
        "heatmap": os.path.join(figures_dir, "heatmap_month_dow.png"),
        "spike_scatter": os.path.join(figures_dir, "spike_scatter.png"),
    }
    _plot_dow(dow, fig_paths["dow_revenue"])
    _plot_month(month, fig_paths["month_revenue"])
    _plot_heatmap(mean_pivot, fig_paths["heatmap"])
    _plot_spike_scatter(scored, fig_paths["spike_scatter"])

    # Report
    dataset_profile = {
        "n_rows": int(len(df)),
        "n_days": int(df["date"].nunique()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "categories": sorted(df["product_category"].unique().tolist()),
        "total_amount_mismatches": mismatches,
    }
    write_markdown_report(
        path=report_path,
        dataset_profile=dataset_profile,
        dow_summary=dow,
        month_summary=month,
        top_spikes=spikes,
        figure_paths={
            "dow_revenue": os.path.relpath(fig_paths["dow_revenue"], os.path.dirname(report_path)),
            "month_revenue": os.path.relpath(fig_paths["month_revenue"], os.path.dirname(report_path)),
            "heatmap": os.path.relpath(fig_paths["heatmap"], os.path.dirname(report_path)),
            "spike_scatter": os.path.relpath(fig_paths["spike_scatter"], os.path.dirname(report_path)),
        },
    )

    # Save a quick quality summary json
    with open(os.path.join(out_dir, "quality_summary.json"), "w", encoding="utf-8") as f:
        json.dump({**sanity, "total_amount_mismatches": mismatches}, f, indent=2)

    return {
        "out_dir": out_dir,
        "report_path": report_path,
        "figures_dir": figures_dir,
        "n_spikes": len(spikes),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to the retail sales CSV")
    ap.add_argument("--out", default="outputs", help="Output directory")
    ap.add_argument("--report", default=os.path.join("reports", "insights.md"), help="Report path (markdown)")
    ap.add_argument("--figures", default=os.path.join("reports", "figures"), help="Figures directory")
    ap.add_argument("--top-n", type=int, default=15, help="Top N spike days to export")
    args = ap.parse_args()

    result = run(args.input, args.out, args.report, args.figures, top_n=args.top_n)
    print("\nDone! Project outputs created.", flush=True)
    print(f"Outputs folder: {result['out_dir']}", flush=True)
    print(f"Report: {result['report_path']}", flush=True)
    print(f"Figures: {result['figures_dir']}", flush=True)
    print(f"Spike days exported: {result['n_spikes']}\n", flush=True)

if __name__ == "__main__":
    main()

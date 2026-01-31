from __future__ import annotations

import pandas as pd

DOW_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

def dow_summary(daily: pd.DataFrame) -> pd.DataFrame:
    s = (daily.groupby("dow", as_index=False)
         .agg(
             avg_daily_revenue=("revenue", "mean"),
             median_daily_revenue=("revenue", "median"),
             avg_txns_per_day=("txns", "mean"),
             avg_units_per_day=("units", "mean"),
             avg_aov=("aov", "mean"),
             n_days=("date", "count"),
         ))
    s["dow"] = pd.Categorical(s["dow"], categories=DOW_ORDER, ordered=True)
    return s.sort_values("dow").reset_index(drop=True)

def month_summary(daily: pd.DataFrame) -> pd.DataFrame:
    s = (daily.groupby("month", as_index=False)
         .agg(
             revenue=("revenue", "sum"),
             txns=("txns", "sum"),
             units=("units", "sum"),
             aov=("aov", "mean"),
             n_days=("date", "count"),
         ))
    return s.sort_values("month").reset_index(drop=True)

def month_dow_heatmap_data(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (mean_revenue, counts) pivot tables (month x dow)."""
    tmp = daily.copy()
    tmp["dow"] = pd.Categorical(tmp["dow"], categories=DOW_ORDER, ordered=True)

    mean_pivot = tmp.pivot_table(index="month", columns="dow", values="revenue", aggfunc="mean", observed=False)
    cnt_pivot = tmp.pivot_table(index="month", columns="dow", values="revenue", aggfunc="count", observed=False)
    return mean_pivot, cnt_pivot

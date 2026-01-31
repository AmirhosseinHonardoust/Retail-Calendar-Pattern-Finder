from __future__ import annotations

import pandas as pd
import numpy as np

def expected_revenue_month_dow(daily: pd.DataFrame, min_cell_days: int = 2) -> pd.Series:
    """Expected revenue per day using a simple seasonal mean hierarchy:
    - month+dow cell mean if cell has >= min_cell_days observations
    - else month mean if month has >= min_cell_days
    - else dow mean if dow has >= min_cell_days
    - else overall mean

    Returns a pd.Series aligned with 'daily' index.
    """
    overall = float(daily["revenue"].mean())

    # Precompute means & counts
    cell = daily.groupby(["month","dow"])["revenue"].agg(["mean","count"]).reset_index()
    month = daily.groupby("month")["revenue"].agg(["mean","count"]).reset_index()
    dow = daily.groupby("dow")["revenue"].agg(["mean","count"]).reset_index()

    # Merge expectations
    out = daily[["month","dow"]].copy()
    out = out.merge(cell, on=["month","dow"], how="left", suffixes=("","_cell"))
    out = out.rename(columns={"mean":"cell_mean","count":"cell_count"})
    out = out.merge(month.rename(columns={"mean":"month_mean","count":"month_count"}), on="month", how="left")
    out = out.merge(dow.rename(columns={"mean":"dow_mean","count":"dow_count"}), on="dow", how="left")

    exp = np.full(len(out), overall, dtype=float)

    # Prefer cell
    m = out["cell_count"].fillna(0).astype(int) >= min_cell_days
    exp[m] = out.loc[m, "cell_mean"].astype(float)

    # Fallback month
    m2 = ~m & (out["month_count"].fillna(0).astype(int) >= min_cell_days)
    exp[m2] = out.loc[m2, "month_mean"].astype(float)

    # Fallback dow
    m3 = ~(m | m2) & (out["dow_count"].fillna(0).astype(int) >= min_cell_days)
    exp[m3] = out.loc[m3, "dow_mean"].astype(float)

    return pd.Series(exp, index=daily.index, name="expected_revenue")

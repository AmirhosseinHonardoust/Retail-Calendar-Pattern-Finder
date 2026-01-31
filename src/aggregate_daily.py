from __future__ import annotations

import pandas as pd

def daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("date", as_index=False).agg(
        revenue=("total_amount", "sum"),
        txns=("transaction_id", "nunique"),
        units=("quantity", "sum"),
        aov=("total_amount", "mean"),
    )
    g["dow"] = g["date"].dt.day_name()
    g["month"] = g["date"].dt.to_period("M").astype(str)
    g["week"] = g["date"].dt.to_period("W-SUN").astype(str)
    return g

def daily_category_mix(df: pd.DataFrame) -> pd.DataFrame:
    cat = df.groupby(["date", "product_category"], as_index=False)["total_amount"].sum()
    cat = cat.rename(columns={"total_amount": "category_revenue"})
    totals = cat.groupby("date", as_index=False)["category_revenue"].sum().rename(columns={"category_revenue": "day_revenue"})
    out = cat.merge(totals, on="date", how="left")
    out["category_share"] = out["category_revenue"] / out["day_revenue"]
    return out

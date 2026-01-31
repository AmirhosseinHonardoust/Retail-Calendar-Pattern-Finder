from __future__ import annotations

import pandas as pd

def weekly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # week label is precomputed in df["week"] (W-SUN)
    g = df.groupby("week", as_index=False).agg(
        revenue=("total_amount", "sum"),
        txns=("transaction_id", "nunique"),
        units=("quantity", "sum"),
        aov=("total_amount", "mean"),
        n_days=("date", "nunique"),
    )
    return g

from __future__ import annotations

import numpy as np
import pandas as pd

def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad

def add_spike_scores(daily: pd.DataFrame, expected: pd.Series) -> pd.DataFrame:
    out = daily.copy()
    out["expected_revenue"] = expected.values
    out["residual"] = out["revenue"] - out["expected_revenue"]
    out["robust_z_revenue"] = robust_zscore(out["revenue"].to_numpy())
    out["robust_z_residual"] = robust_zscore(out["residual"].to_numpy())
    return out

def top_spike_days(scored: pd.DataFrame, n: int = 15, by: str = "robust_z_residual") -> pd.DataFrame:
    cols = [
        "date","dow","month","week",
        "revenue","expected_revenue","residual",
        "txns","units","aov",
        "robust_z_revenue","robust_z_residual",
    ]
    return scored.sort_values(by, ascending=False).head(n)[cols].reset_index(drop=True)

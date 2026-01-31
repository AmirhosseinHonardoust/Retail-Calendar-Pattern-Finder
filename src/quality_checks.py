from __future__ import annotations

import pandas as pd

def check_total_amount_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Check if total_amount == quantity * price_per_unit."""
    chk = df.copy()
    chk["expected_total"] = chk["quantity"] * chk["price_per_unit"]
    chk["abs_error"] = (chk["total_amount"] - chk["expected_total"]).abs()
    # floating tolerant threshold
    chk["is_mismatch"] = chk["abs_error"] > 1e-6
    return chk[["transaction_id", "date", "quantity", "price_per_unit", "total_amount", "expected_total", "abs_error", "is_mismatch"]]

def basic_sanity_checks(df: pd.DataFrame) -> dict:
    """Return a dict of simple quality stats."""
    out = {}
    out["n_rows"] = int(len(df))
    out["n_missing_any"] = int(df.isna().any(axis=1).sum())
    out["n_negative_quantity"] = int((df["quantity"] <= 0).sum())
    out["n_negative_price"] = int((df["price_per_unit"] <= 0).sum())
    out["n_negative_total"] = int((df["total_amount"] <= 0).sum())
    out["date_min"] = str(df["date"].min().date())
    out["date_max"] = str(df["date"].max().date())
    return out

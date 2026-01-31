from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "date",
    "transaction_id",
    "customer_id",
    "gender",
    "age",
    "product_category",
    "quantity",
    "price_per_unit",
    "total_amount",
]

def _to_snake_case(name: str) -> str:
    return (
        name.strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .lower()
    )

def load_retail_csv(path: str) -> pd.DataFrame:
    """Load the Kaggle retail sales CSV and standardize column names."""
    df = pd.read_csv(path)

    # Standardize columns
    df.columns = [_to_snake_case(c) for c in df.columns]

    # Date parsing
    if "date" not in df.columns:
        raise ValueError("Expected a 'Date' column in the input CSV.")
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()

    # Basic sanity checks
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Type casts
    df["transaction_id"] = df["transaction_id"].astype(str)
    df["customer_id"] = df["customer_id"].astype(str)
    df["gender"] = df["gender"].astype(str)
    df["product_category"] = df["product_category"].astype(str)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price_per_unit"] = pd.to_numeric(df["price_per_unit"], errors="coerce")
    df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")

    # Derived calendar fields
    df["dow"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["week"] = df["date"].dt.to_period("W-SUN").astype(str)  # week ending Sunday

    return df

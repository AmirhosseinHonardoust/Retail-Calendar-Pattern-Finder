from __future__ import annotations

import json
import pandas as pd
import numpy as np

def _as_float(x):
    try:
        return float(x)
    except Exception:
        return None

def build_spike_cards(
    scored_daily: pd.DataFrame,
    daily_mix: pd.DataFrame,
    top_n: int = 15,
    score_col: str = "robust_z_residual",
) -> list[dict]:
    """Create exportable 'Spike Cards' for the top-N spike days."""
    top = scored_daily.sort_values(score_col, ascending=False).head(top_n).copy()

    # Expected drivers using identity: revenue ~ txns * aov
    overall_aov = float(scored_daily["aov"].mean())
    overall_txns = float(scored_daily["txns"].mean())
    overall_units = float(scored_daily["units"].mean())

    cards = []
    for _, r in top.iterrows():
        day = r["date"]
        day_mix = daily_mix[daily_mix["date"] == day].sort_values("category_revenue", ascending=False)
        if len(day_mix) > 0:
            top_cat = day_mix.iloc[0]
            top_cat_name = str(top_cat["product_category"])
            top_cat_rev = float(top_cat["category_revenue"])
            top_cat_share = float(top_cat["category_share"])
        else:
            top_cat_name, top_cat_rev, top_cat_share = None, None, None

        expected_rev = _as_float(r.get("expected_revenue"))
        actual_rev = _as_float(r.get("revenue"))
        delta = None if (expected_rev is None or actual_rev is None) else (actual_rev - expected_rev)
        delta_pct = None if (expected_rev in (None, 0.0) or delta is None) else (delta / expected_rev * 100.0)

        # crude expected drivers: use seasonal expected revenue but baseline txns/units/aov from global means
        # (keeps the "driver" section present even if you later replace with model-based expectations)
        card = {
            "date": str(pd.to_datetime(day).date()),
            "dow": str(r.get("dow")),
            "actual_revenue": actual_rev,
            "expected_revenue": expected_rev,
            "delta_revenue": delta,
            "delta_pct": delta_pct,
            "score": _as_float(r.get(score_col)),
            "drivers": {
                "txns": {"actual": _as_float(r.get("txns")), "baseline_mean": overall_txns},
                "units": {"actual": _as_float(r.get("units")), "baseline_mean": overall_units},
                "aov": {"actual": _as_float(r.get("aov")), "baseline_mean": overall_aov},
            },
            "top_category": {
                "name": top_cat_name,
                "revenue": top_cat_rev,
                "share": top_cat_share,
            },
            "notes": [
                "Baseline expectations use a simple seasonal hierarchy (month+dow, then month, then dow).",
                "Treat missing dates as unknown coverage (dataset includes only transaction-days).",
            ],
        }
        cards.append(card)

    return cards

def save_cards(cards: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)

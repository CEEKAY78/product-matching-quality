"""Data and mapping quality diagnostics.

Two kinds of checks:
- profile_catalog: completeness / duplicates inside one catalog;
- mapping_quality: referential integrity and cardinality of the gold mapping
  (dangling ids, one-to-many links, coverage of each catalog).
"""

from __future__ import annotations

import pandas as pd


def _pct(numerator: float, denominator: float) -> float:
    return round(100.0 * numerator / denominator, 1) if denominator else 0.0


def profile_catalog(frame: pd.DataFrame) -> dict[str, float | int]:
    n = len(frame)
    names = frame["name"].fillna("").str.strip()
    dup_mask = names.ne("") & names.duplicated(keep=False)
    return {
        "rows": n,
        "missing_price_pct": _pct(frame["price"].isna().sum(), n),
        "missing_description_pct": _pct(frame["description"].fillna("").eq("").sum(), n),
        "missing_manufacturer_pct": _pct(frame["manufacturer"].fillna("").eq("").sum(), n),
        "empty_name_rows": int(names.eq("").sum()),
        "duplicate_name_rows": int(dup_mask.sum()),
        "median_price": float(frame["price"].median()) if frame["price"].notna().any() else 0.0,
    }


def mapping_quality(
    abt: pd.DataFrame, buy: pd.DataFrame, mapping: pd.DataFrame
) -> dict[str, float | int]:
    abt_ids = set(abt["id"])
    buy_ids = set(buy["id"])
    pairs = mapping.drop_duplicates()
    per_abt = pairs.groupby("idAbt")["idBuy"].nunique()
    per_buy = pairs.groupby("idBuy")["idAbt"].nunique()
    return {
        "pairs": int(len(mapping)),
        "duplicate_pairs": int(len(mapping) - len(pairs)),
        "dangling_abt_ids": int((~pairs["idAbt"].isin(abt_ids)).sum()),
        "dangling_buy_ids": int((~pairs["idBuy"].isin(buy_ids)).sum()),
        "abt_ids_with_many_buy": int((per_abt > 1).sum()),
        "buy_ids_with_many_abt": int((per_buy > 1).sum()),
        "abt_coverage_pct": _pct(len(set(pairs["idAbt"]) & abt_ids), len(abt_ids)),
        "buy_coverage_pct": _pct(len(set(pairs["idBuy"]) & buy_ids), len(buy_ids)),
    }

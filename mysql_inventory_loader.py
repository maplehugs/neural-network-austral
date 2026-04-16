"""
Utilities to load daily inventory snapshots from MySQL and convert them
into the schema expected by stockout_forecasting.py:

Required columns in output:
- sku (str)
- fecha (datetime64)
- stock (float)
- consumo (float, inferred from stock deltas)
"""

from __future__ import annotations

import os
import re
import unicodedata
from datetime import timedelta
from typing import Optional

import pandas as pd

from import_excel import TABLE_NAME, get_engine


def _canonical_name(col_name: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(col_name))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_only.lower()).strip("_")


def _pick_column(columns: list[str], candidates: list[str], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"No se encontro columna para '{label}'. Disponibles: {columns}")


def _normalize_schema(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("La tabla esta vacia en MySQL.")

    canonical_map = {_canonical_name(col): col for col in raw_df.columns}
    canonical_cols = list(canonical_map.keys())

    date_col = _pick_column(canonical_cols, ["fecha", "date", "dia"], "fecha")
    sku_col = _pick_column(
        canonical_cols,
        ["codigo", "cdigo", "sku", "item", "producto", "id_producto"],
        "sku",
    )
    stock_col = _pick_column(
        canonical_cols,
        ["t_unid", "stock", "existencia", "cantidad", "units", "inventario"],
        "stock",
    )

    df = raw_df[[canonical_map[date_col], canonical_map[sku_col], canonical_map[stock_col]]].copy()
    df.columns = ["fecha", "sku", "stock"]

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["stock"] = pd.to_numeric(df["stock"], errors="coerce")
    df["sku"] = df["sku"].astype(str).str.strip()

    df = df.dropna(subset=["fecha", "sku", "stock"])
    df = df[df["stock"] >= 0]

    # If there are repeated snapshots for the same sku/day, keep max stock of that day.
    df = (
        df.groupby(["sku", "fecha"], as_index=False)["stock"]
        .max()
        .sort_values(["sku", "fecha"])
        .reset_index(drop=True)
    )
    return df


def _fill_missing_days_and_infer_consumption(df: pd.DataFrame) -> pd.DataFrame:
    records: list[pd.DataFrame] = []

    for sku, grp in df.groupby("sku"):
        grp = grp.sort_values("fecha").copy()
        full_range = pd.date_range(grp["fecha"].min(), grp["fecha"].max(), freq="D")

        g = grp.set_index("fecha").reindex(full_range)
        g.index.name = "fecha"
        g["sku"] = sku

        # Assume missing snapshots keep last known stock level.
        g["stock"] = g["stock"].ffill().fillna(0)

        # Daily consumption inferred from stock drops; restocks are clipped to zero consumption.
        g["consumo"] = (g["stock"].shift(1) - g["stock"]).clip(lower=0).fillna(0)

        records.append(g.reset_index())

    final_df = pd.concat(records, ignore_index=True)
    final_df["fecha"] = pd.to_datetime(final_df["fecha"])
    return final_df[["sku", "fecha", "stock", "consumo"]]


def _filter_skus(df: pd.DataFrame, min_points: int = 10) -> pd.DataFrame:
    """Keep SKUs with enough historical points before daily reindexing."""
    stats = df.groupby("sku")["stock"].agg(["count"])
    valid_skus = stats[stats["count"] >= min_points].index
    return df[df["sku"].isin(valid_skus)].copy()


def _filter_after_consumption(
    df: pd.DataFrame,
    min_consumo_sum: float = 0.0,
    min_consumo_mean: float = 0.01,
) -> pd.DataFrame:
    """Drop SKUs that never sell or have near-zero average demand."""
    stats = df.groupby("sku")["consumo"].agg(["sum", "mean"])
    valid = stats[
        (stats["sum"] > min_consumo_sum)
        & (stats["mean"] > min_consumo_mean)
    ].index
    return df[df["sku"].isin(valid)].copy()


def load_inventory_for_forecasting(
    days_back: int = 30,
    use_all_history: bool = True,
    table_name: Optional[str] = None,
) -> pd.DataFrame:
    table = table_name or TABLE_NAME
    engine = get_engine()

    with engine.connect() as connection:
        raw_df = pd.read_sql_query(f"SELECT * FROM {table}", con=connection)

    df = _normalize_schema(raw_df)

    if not use_all_history:
        max_date = df["fecha"].max()
        start_date = max_date - timedelta(days=max(1, days_back) - 1)
        df = df[df["fecha"] >= start_date].copy()

    min_points = int(os.getenv("STOCKOUT_MIN_POINTS_PER_SKU", "10"))
    min_consumo_mean = float(os.getenv("STOCKOUT_MIN_CONSUMO_MEAN", "0.01"))

    before_pre = df["sku"].nunique()
    df = _filter_skus(df, min_points=min_points)
    after_pre = df["sku"].nunique()

    out = _fill_missing_days_and_infer_consumption(df)

    before_post = out["sku"].nunique()
    out = _filter_after_consumption(
        out,
        min_consumo_sum=0.0,
        min_consumo_mean=min_consumo_mean,
    )
    after_post = out["sku"].nunique()

    print(
        "[CLEAN] SKUs pre_filter: "
        f"{before_pre} -> {after_pre} | post_consumption: {before_post} -> {after_post}"
    )

    out = out.sort_values(["sku", "fecha"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    preview = load_inventory_for_forecasting(days_back=30, use_all_history=False)
    print(preview.head().to_string(index=False))
    print(f"Rows: {len(preview)} | SKUs: {preview['sku'].nunique()}")


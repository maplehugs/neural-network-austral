"""
Load daily inventory snapshots directly from Excel files in a folder and
convert them to the forecasting schema expected by stockout_forecasting.py.

Output columns:
- sku (str)
- fecha (datetime64)
- stock (float)
- consumo (float)
"""

from __future__ import annotations

import os
import re
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


def parse_date_from_filename(filename: str) -> date | None:
    match = re.match(r"(\d{1,2})_(\d{1,2})_(\d{4})", filename)
    if not match:
        return None
    d, mth, y = int(match.group(1)), int(match.group(2)), int(match.group(3))
    try:
        return date(y, mth, d)
    except ValueError:
        return None


def _canonical_name(col_name: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(col_name))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_only.lower()).strip("_")


def _pick_column(columns: list[str], candidates: list[str], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"No se encontro columna para '{label}'. Disponibles: {columns}")


def _normalize_snapshot(df_raw: pd.DataFrame, snapshot_date: date) -> pd.DataFrame:
    canonical_map = {_canonical_name(col): col for col in df_raw.columns}
    canonical_cols = list(canonical_map.keys())

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

    df = df_raw[[canonical_map[sku_col], canonical_map[stock_col]]].copy()
    df.columns = ["sku", "stock"]
    df["sku"] = df["sku"].astype(str).str.strip()
    df["stock"] = pd.to_numeric(df["stock"], errors="coerce")

    df = df.dropna(subset=["sku", "stock"])
    df = df[df["stock"] >= 0]
    df["fecha"] = pd.to_datetime(snapshot_date)

    # If a file contains duplicated codes, keep the highest stock for that day.
    df = df.groupby(["sku", "fecha"], as_index=False)["stock"].max()
    return df[["sku", "fecha", "stock"]]


def _fill_missing_days_and_infer_consumption(df: pd.DataFrame) -> pd.DataFrame:
    records: list[pd.DataFrame] = []

    for sku, grp in df.groupby("sku"):
        grp = grp.sort_values("fecha").copy()
        full_range = pd.date_range(grp["fecha"].min(), grp["fecha"].max(), freq="D")

        g = grp.set_index("fecha").reindex(full_range)
        g.index.name = "fecha"
        g["sku"] = sku
        g["stock"] = g["stock"].ffill().fillna(0)
        g["consumo"] = (g["stock"].shift(1) - g["stock"]).clip(lower=0).fillna(0)

        records.append(g.reset_index())

    out = pd.concat(records, ignore_index=True)
    out["fecha"] = pd.to_datetime(out["fecha"])
    return out[["sku", "fecha", "stock", "consumo"]]


def _filter_skus(df: pd.DataFrame, min_points: int = 10) -> pd.DataFrame:
    stats = df.groupby("sku")["stock"].agg(["count"])
    valid = stats[stats["count"] >= min_points].index
    return df[df["sku"].isin(valid)].copy()


def _filter_after_consumption(
    df: pd.DataFrame,
    min_consumo_sum: float = 0.0,
    min_consumo_mean: float = 0.01,
) -> pd.DataFrame:
    stats = df.groupby("sku")["consumo"].agg(["sum", "mean"])
    valid = stats[
        (stats["sum"] > min_consumo_sum)
        & (stats["mean"] > min_consumo_mean)
    ].index
    return df[df["sku"].isin(valid)].copy()


def load_inventory_from_excels(
    data_dir: str = "data",
    days_back: int = 30,
    use_all_history: bool = True,
    max_files: int | None = None,
) -> pd.DataFrame:
    folder = Path(data_dir)
    files = sorted(
        folder.glob("*.xlsx"),
        key=lambda p: parse_date_from_filename(p.name) or date.min,
    )

    if max_files is not None and max_files > 0:
        files = files[:max_files]

    if not files:
        raise ValueError(f"No se encontraron archivos .xlsx en: {folder.resolve()}")

    frames: list[pd.DataFrame] = []
    for file_path in files:
        snapshot_date = parse_date_from_filename(file_path.name)
        if snapshot_date is None:
            continue

        try:
            raw = pd.read_excel(file_path)
            norm = _normalize_snapshot(raw, snapshot_date)
            if not norm.empty:
                frames.append(norm)
        except Exception as exc:
            print(f"[WARN] Omitiendo {file_path.name}: {exc}")

    if not frames:
        raise ValueError("No hubo archivos Excel validos para construir el dataset.")

    df = pd.concat(frames, ignore_index=True)
    df = df.groupby(["sku", "fecha"], as_index=False)["stock"].max()

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
    preview = load_inventory_from_excels(data_dir="data", use_all_history=False, days_back=30)
    print(preview.head().to_string(index=False))
    print(f"Rows: {len(preview)} | SKUs: {preview['sku'].nunique()}")


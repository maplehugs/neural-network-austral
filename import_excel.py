"""
import_inventario_mysql.py
--------------------------
Lee archivos .xlsx y los sube a MySQL

Requisitos:
    pip install pandas openpyxl sqlalchemy mysql-connector-python
"""

import os
import re
import unicodedata
from pathlib import Path
from datetime import date

import pandas as pd
from sqlalchemy import create_engine

# ── CONFIGURACIÓN ─────────────────────────────────────────────
XLSX_DIR = Path("data")

DB_USER = "maple"
DB_PASS = "maple"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "inventario_db"

TABLE_NAME = "inventario"
WRITE_MODE = os.getenv("MYSQL_IMPORT_MODE", "append").strip().lower()
if WRITE_MODE not in {"append", "replace"}:
    WRITE_MODE = "append"
# ─────────────────────────────────────────────────────────────


def get_engine():
    return create_engine(
        f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


def parse_date_from_filename(filename: str):
    m = re.match(r"(\d{1,2})_(\d{1,2})_(\d{4})", filename)
    if not m:
        return None
    d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return date(y, mth, d)
    except:
        return None


def _canonical_name(col_name: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(col_name))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_only.lower()).strip("_")


def _resolve_codigo_column(columns):
    canonical_map = {_canonical_name(c): c for c in columns}
    for candidate in ["codigo", "cdigo", "sku", "item"]:
        if candidate in canonical_map:
            return canonical_map[candidate]
    raise ValueError(f"No se encontro columna de codigo en: {list(columns)}")


def load_xlsx(path: Path, fecha):
    df = pd.read_excel(path)

    codigo_col = _resolve_codigo_column(df.columns)
    df = df[df[codigo_col].notna()].copy()
    df[codigo_col] = pd.to_numeric(df[codigo_col], errors="coerce")
    df = df[df[codigo_col].notna()].copy()
    df[codigo_col] = df[codigo_col].astype(int)

    df.insert(0, "fecha", pd.to_datetime(fecha))

    return df, codigo_col


def main():
    files = sorted(
        XLSX_DIR.glob("*.xlsx"),
        key=lambda p: parse_date_from_filename(p.name) or date.min
    )

    if not files:
        print("No hay archivos")
        return

    frames = []
    code_col_for_sort = None

    for f in files:
        fecha = parse_date_from_filename(f.name)
        if not fecha:
            continue

        try:
            df, codigo_col = load_xlsx(f, fecha)
        except Exception as exc:
            print(f"[WARN] {f.name} omitido: {exc}")
            continue

        code_col_for_sort = codigo_col
        frames.append(df)
        print(f"[OK] {f.name} -> {fecha}")

    if not frames:
        print("No se encontraron archivos validos para importar")
        return

    combined = pd.concat(frames, ignore_index=True)
    if code_col_for_sort and code_col_for_sort in combined.columns:
        combined.sort_values(["fecha", code_col_for_sort], inplace=True)
    else:
        combined.sort_values(["fecha"], inplace=True)

    engine = get_engine()

    combined.to_sql(
        TABLE_NAME,
        con=engine,
        if_exists=WRITE_MODE,
        index=False,
        chunksize=1000
    )

    print(f"\nDatos subidos a MySQL ({DB_NAME}.{TABLE_NAME})")
    print(f"Modo escritura: {WRITE_MODE}")
    print(f"Filas: {len(combined)}")


if __name__ == "__main__":
    main()
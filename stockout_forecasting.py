"""
=============================================================================
PREDICCIÓN DE AGOTAMIENTO DE STOCK — DEMANDA INTERMITENTE
Proyecto de grado | Enfoque híbrido GRU + Croston/SBA
=============================================================================

Arquitectura:
  - GRU dual-output: clasif. (¿hubo venta?) + regresión (¿cuánto?)
  - Croston/SBA como benchmark estadístico
  - Simulación Monte Carlo a 30 días por SKU

Dependencias:
  pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from pathlib import Path

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, mean_absolute_error,
                             brier_score_loss, precision_recall_curve, confusion_matrix)
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import Adam

# Loader imports are intentionally lazy in the data-source block so that
# Colab runs from /data do not require MySQL dependencies.

tf.random.set_seed(42)
np.random.seed(42)

QUICK_MODE = os.getenv("STOCKOUT_QUICK_MODE", "0").strip() == "1"
TRAIN_EPOCHS = int(os.getenv("STOCKOUT_TRAIN_EPOCHS", "60" if not QUICK_MODE else "6"))
BENCHMARK_MAX_SKUS = int(os.getenv("STOCKOUT_BENCHMARK_MAX_SKUS", "0"))
MC_TARGET_SIMS = int(os.getenv("STOCKOUT_MC_TARGET_SIMS", "500" if not QUICK_MODE else "80"))
MC_ALL_SIMS = int(os.getenv("STOCKOUT_MC_ALL_SIMS", "300" if not QUICK_MODE else "40"))

ARTIFACT_DIR = Path(os.getenv("STOCKOUT_ARTIFACT_DIR", "artifacts"))
LOAD_EXISTING_MODEL = os.getenv("STOCKOUT_LOAD_MODEL", "0").strip() == "1"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

EPSILON = 1e-6
CROSTON_SEED_OFFSET = 1000
TRAIN_CUTOFF_RATIO = 0.85

# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERACIÓN DE DATOS SINTÉTICOS
#    (si ya tienes un CSV, salta al bloque "CARGA DE DATOS REALES")
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(n_skus: int = 20,
                             days: int = 365,
                             seed: int = 42) -> pd.DataFrame:
    """
    Genera un DataFrame con columnas: sku, fecha, stock, consumo.
    Simula tres tipos de SKUs:
      - 'fast': venta casi diaria (ej. lentes estándar populares)
      - 'slow': venta cada 2-4 semanas (ej. graduaciones especiales)
      - 'erratic': consumo en ráfagas esporádicas (ej. lentes de nicho)
    """
    rng = np.random.default_rng(seed)
    records = []

    for sku_id in range(n_skus):
        tipo = rng.choice(["fast", "slow", "erratic"],
                          p=[0.4, 0.35, 0.25])
        stock = rng.integers(50, 200)
        fecha_inicio = pd.Timestamp("2023-01-01")

        for day in range(days):
            fecha = fecha_inicio + timedelta(days=day)

            # Probabilidad de venta según tipo
            if tipo == "fast":
                prob_venta = 0.75
                qty_if_sale = rng.integers(1, 5)
            elif tipo == "slow":
                prob_venta = 0.15
                qty_if_sale = rng.integers(1, 3)
            else:  # erratic: ráfagas
                prob_venta = 0.08
                qty_if_sale = rng.integers(3, 15)

            hubo_venta = rng.random() < prob_venta
            consumo = int(qty_if_sale) if hubo_venta and stock > 0 else 0
            consumo = min(consumo, stock)
            stock = max(0, stock - consumo)

            # Reposición aleatoria
            if stock < 20 and rng.random() < 0.3:
                stock += rng.integers(30, 80)

            records.append({
                "sku": f"SKU_{sku_id:03d}",
                "tipo": tipo,
                "fecha": fecha,
                "stock": stock,
                "consumo": consumo,
            })

    df = pd.DataFrame(records)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS REALES (Excel/MySQL) + fallback sintético
# ─────────────────────────────────────────────────────────────────────────────
# Variables de entorno opcionales:
#   STOCKOUT_DATA_SOURCE=excel|mysql|synthetic
#   STOCKOUT_DATA_DIR=data
#   STOCKOUT_DAYS_BACK=30
#   STOCKOUT_ALL_HISTORY=1|0
DATA_SOURCE = os.getenv("STOCKOUT_DATA_SOURCE", "excel").strip().lower()
DATA_DIR = os.getenv("STOCKOUT_DATA_DIR", "data").strip()
DAYS_BACK = int(os.getenv("STOCKOUT_DAYS_BACK", "30"))
USE_ALL_HISTORY = os.getenv("STOCKOUT_ALL_HISTORY", "1").strip() == "1"

if DATA_SOURCE == "excel":
    try:
        from excel_inventory_loader import load_inventory_from_excels

        df = load_inventory_from_excels(
            data_dir=DATA_DIR,
            days_back=DAYS_BACK,
            use_all_history=USE_ALL_HISTORY,
        )
        if df.empty:
            raise ValueError("La carga desde Excel no devolvio filas utiles.")

        n_days = df["fecha"].nunique()
        if n_days < 14:
            raise ValueError(
                f"Historial insuficiente ({n_days} dias). Se recomienda >= 14 dias por SKU."
            )

        print("[DATA] Fuente: Excel")
        print(f"[DATA] Carpeta: {DATA_DIR}")
        print(f"[DATA] Shape: {df.shape} | SKUs: {df['sku'].nunique()} | Dias unicos: {n_days}")
    except Exception as exc:
        print(f"[WARN] No se pudo usar Excel ({DATA_DIR}): {exc}")
        print("[WARN] Usando datos sinteticos como fallback...")
        df = generate_synthetic_data(n_skus=20, days=365)
elif DATA_SOURCE == "mysql":
    try:
        from mysql_inventory_loader import load_inventory_for_forecasting

        df = load_inventory_for_forecasting(
            days_back=DAYS_BACK,
            use_all_history=USE_ALL_HISTORY,
        )
        if df.empty:
            raise ValueError("La consulta a MySQL no devolvio filas utiles.")

        n_days = df["fecha"].nunique()
        if n_days < 14:
            raise ValueError(
                f"Historial insuficiente ({n_days} dias). Se recomienda >= 14 dias por SKU."
            )

        print("[DATA] Fuente: MySQL")
        print(f"[DATA] Shape: {df.shape} | SKUs: {df['sku'].nunique()} | Dias unicos: {n_days}")
    except Exception as exc:
        print(f"[WARN] No se pudo usar MySQL: {exc}")
        print("[WARN] Usando datos sinteticos como fallback...")
        df = generate_synthetic_data(n_skus=20, days=365)
else:
    df = generate_synthetic_data(n_skus=20, days=365)
    print("[DATA] Fuente: sintetica (forzada por configuracion)")

print(df.head())


# ─────────────────────────────────────────────────────────────────────────────
# 2. INGENIERÍA DE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame,
                   seq_len: int = 30) -> pd.DataFrame:
    """
    Construye variables de entrada a nivel diario por SKU.
    Retorna un DataFrame enriquecido.
    """
    df = df.sort_values(["sku", "fecha"]).copy()

    # Lags y ventanas rodantes
    df["consumo_1d"]  = df.groupby("sku")["consumo"].shift(1).fillna(0)
    df["consumo_7d"]  = (df.groupby("sku")["consumo"]
                           .transform(lambda x: x.shift(1).rolling(7, min_periods=1).sum()))
    df["consumo_30d"] = (df.groupby("sku")["consumo"]
                           .transform(lambda x: x.shift(1).rolling(30, min_periods=1).sum()))
    df["consumo_avg7"] = (df.groupby("sku")["consumo"]
                            .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean()))

    # Días sin venta consecutivos
    def dias_sin_venta(series):
        result = []
        count = 0
        for v in series:
            if v == 0:
                count += 1
            else:
                count = 0
            result.append(count)
        return pd.Series(result, index=series.index)

    df["dias_sin_venta"] = df.groupby("sku")["consumo"].transform(dias_sin_venta)

    # Velocidad de consumo (stock / consumo_avg, evitando div 0)
    df["cobertura_dias"] = np.where(
        df["consumo_avg7"] > 0,
        df["stock"] / df["consumo_avg7"],
        999
    )
    df["cobertura_dias"] = df["cobertura_dias"].clip(upper=999)

    # Calendario
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["mes"]        = df["fecha"].dt.month
    df["dia_mes"]    = df["fecha"].dt.day

    # Etiqueta clasificación: ¿hubo consumo HOY?
    df["y_clf"]  = (df["consumo"] > 0).astype(int)
    # Etiqueta regresión: cuánto se consumió (0 si no hubo)
    df["y_reg"]  = df["consumo"].astype(float)

    df = df.fillna(0)
    return df


df_feat = build_features(df)

FEATURE_COLS = [
    "consumo_1d", "consumo_7d", "consumo_30d", "consumo_avg7",
    "dias_sin_venta", "cobertura_dias", "stock",
    "dia_semana", "mes", "dia_mes"
]

print(f"\n[FEATURES] Columnas: {FEATURE_COLS}")
print(df_feat[FEATURE_COLS + ["y_clf", "y_reg"]].describe())


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONSTRUCCIÓN DE SECUENCIAS PARA LA GRU
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(df: pd.DataFrame,
                    feature_cols: list,
                    seq_len: int = 14) -> tuple:
    """
    Para cada fila t en la serie de un SKU, construye una ventana
    [t-seq_len, ..., t-1] como input y el valor en t como target.

    Returns:
        X: (n_samples, seq_len, n_features)
        y_clf: (n_samples,)  — clasificación
        y_reg: (n_samples,)  — regresión
    """
    X_list, y_clf_list, y_reg_list = [], [], []

    for sku, grp in df.groupby("sku"):
        grp = grp.sort_values("fecha").reset_index(drop=True)
        feats = grp[feature_cols].values
        clf   = grp["y_clf"].values
        reg   = grp["y_reg"].values

        for i in range(seq_len, len(grp)):
            X_list.append(feats[i - seq_len:i])
            y_clf_list.append(clf[i])
            y_reg_list.append(reg[i])

    X      = np.array(X_list, dtype=np.float32)
    y_clf  = np.array(y_clf_list, dtype=np.float32)
    y_reg  = np.array(y_reg_list, dtype=np.float32)
    return X, y_clf, y_reg


SEQ_LEN = 14   # Ventana de 14 días hacia atrás

X_all, y_clf_all, y_reg_all = build_sequences(df_feat, FEATURE_COLS, SEQ_LEN)
print(f"\n[SEQUENCES] X: {X_all.shape} | y_clf: {y_clf_all.shape} | y_reg: {y_reg_all.shape}")
print(f"  Tasa de venta (clase 1): {y_clf_all.mean():.2%}")

if len(X_all) == 0:
    raise ValueError(
        "No se pudieron construir secuencias. Usa mas historial (STOCKOUT_ALL_HISTORY=1) "
        "o baja SEQ_LEN."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. ESCALADO + SPLIT TEMPORAL
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split(X, y_clf, y_reg,
                   val_ratio=0.15, test_ratio=0.15):
    """
    Split respetando el orden temporal:
      train | val | test  (sin shuffle)
    """
    n = len(X)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    n_train = n - n_val - n_test

    X_tr,   y_clf_tr,  y_reg_tr  = X[:n_train], y_clf[:n_train], y_reg[:n_train]
    X_val,  y_clf_val, y_reg_val = X[n_train:n_train+n_val], y_clf[n_train:n_train+n_val], y_reg[n_train:n_train+n_val]
    X_te,   y_clf_te,  y_reg_te  = X[n_train+n_val:], y_clf[n_train+n_val:], y_reg[n_train+n_val:]

    print(f"\n[SPLIT] Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}")
    return (X_tr, y_clf_tr, y_reg_tr,
            X_val, y_clf_val, y_reg_val,
            X_te, y_clf_te, y_reg_te)


(X_tr, y_clf_tr, y_reg_tr,
 X_val, y_clf_val, y_reg_val,
 X_te, y_clf_te, y_reg_te) = temporal_split(X_all, y_clf_all, y_reg_all)

# Escalar features (fit solo en train para no filtrar info futura)
scaler = StandardScaler()
n_tr, seq, n_feat = X_tr.shape
X_tr_raw, X_val_raw, X_te_raw = X_tr.copy(), X_val.copy(), X_te.copy()

def transform_sequences(X_data, scaler_obj, n_feat_local, seq_local):
    return scaler_obj.transform(X_data.reshape(-1, n_feat_local)).reshape(-1, seq_local, n_feat_local).astype(np.float32)


X_tr = scaler.fit_transform(X_tr_raw.reshape(-1, n_feat)).reshape(-1, seq, n_feat).astype(np.float32)
X_val = transform_sequences(X_val_raw, scaler, n_feat, seq)
X_te = transform_sequences(X_te_raw, scaler, n_feat, seq)


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODELO GRU DUAL-OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
# Elegimos GRU sobre LSTM porque:
#   - Tiene menos parámetros → menos sobreajuste con datos escasos
#   - Captura dependencias temporales sin asumir suavidad en la serie
#   - Converge más rápido que LSTM en sequences cortas (seq_len=14)
#
# Arquitectura dual-output:
#   Trunk GRU compartido → cabeza clf (sigmoid) + cabeza reg (relu/linear)
#   La cabeza clf predice P(venta hoy)
#   La cabeza reg predice cantidad IF hay venta (interpretado junto con clf)
# ─────────────────────────────────────────────────────────────────────────────

def build_gru_model(seq_len: int,
                    n_features: int,
                    gru_units: int = 64,
                    dropout_rate: float = 0.3) -> Model:
    """
    GRU con dos cabezas de salida:
      - output_clf: probabilidad de venta (sigmoid)
      - output_reg: cantidad consumida esperada (relu)
    """
    inp = layers.Input(shape=(seq_len, n_features), name="input_seq")

    # Trunk compartido: dos capas GRU apiladas
    x = layers.GRU(gru_units, return_sequences=True,
                   recurrent_dropout=0.1, name="gru_1")(inp)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.GRU(gru_units // 2, return_sequences=False,
                   recurrent_dropout=0.1, name="gru_2")(x)
    x = layers.Dropout(dropout_rate)(x)

    # Capa densa compartida
    shared = layers.Dense(32, activation="relu", name="dense_shared")(x)

    # Cabeza clasificación: ¿hubo venta?
    out_clf = layers.Dense(1, activation="sigmoid",
                           name="output_clf")(shared)

    # Cabeza regresión: ¿cuánto se consumió? (solo relevante si clf > umbral)
    out_reg = layers.Dense(1, activation="relu",
                           name="output_reg")(shared)

    model = Model(inputs=inp, outputs=[out_clf, out_reg])
    return model


model = build_gru_model(SEQ_LEN, len(FEATURE_COLS))
model.summary()

# Compilación con pérdidas ponderadas
#   - Binary crossentropy para clasificación
#   - Huber loss para regresión (más robusta a outliers que MSE)
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss={
        "output_clf": "binary_crossentropy",
        "output_reg": tf.keras.losses.Huber(delta=2.0),
    },
    loss_weights={"output_clf": 1.0, "output_reg": 0.5},
    metrics={"output_clf": ["accuracy"], "output_reg": ["mae"]}
)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

early_stop = callbacks.EarlyStopping(
    monitor="val_output_clf_accuracy",
    patience=10,
    restore_best_weights=True,
    mode="max"
)

lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
)

print("\n[TRAIN] Entrenando GRU dual-output...")

def save_artifacts(model_obj, scaler_obj, metadata: dict):
    model_obj.save(ARTIFACT_DIR / "gru_model.keras")
    with open(ARTIFACT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler_obj, f)
    with open(ARTIFACT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_artifacts():
    model_path = ARTIFACT_DIR / "gru_model.keras"
    scaler_path = ARTIFACT_DIR / "scaler.pkl"
    meta_path = ARTIFACT_DIR / "metadata.json"
    if not (model_path.exists() and scaler_path.exists() and meta_path.exists()):
        return None, None, None
    loaded_model = tf.keras.models.load_model(model_path)
    with open(scaler_path, "rb") as f:
        loaded_scaler = pickle.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return loaded_model, loaded_scaler, metadata


history = None
loaded_ok = False
if LOAD_EXISTING_MODEL:
    loaded_model, loaded_scaler, metadata = load_artifacts()
    if loaded_model is not None:
        print(f"[LOAD] Modelo y scaler cargados desde {ARTIFACT_DIR.resolve()}")
        model = loaded_model
        scaler = loaded_scaler
        X_tr = transform_sequences(X_tr_raw, scaler, n_feat, seq)
        X_val = transform_sequences(X_val_raw, scaler, n_feat, seq)
        X_te = transform_sequences(X_te_raw, scaler, n_feat, seq)
        loaded_ok = True
    else:
        print(f"[LOAD] No hay artefactos en {ARTIFACT_DIR.resolve()}, se entrena desde cero.")

if not loaded_ok:
    history = model.fit(
        X_tr,
        {"output_clf": y_clf_tr, "output_reg": y_reg_tr},
        validation_data=(X_val, {"output_clf": y_clf_val, "output_reg": y_reg_val}),
        epochs=TRAIN_EPOCHS,
        batch_size=128,
        callbacks=[early_stop, lr_scheduler],
        verbose=1
    )
    save_artifacts(model, scaler, {
        "feature_cols": FEATURE_COLS,
        "seq_len": SEQ_LEN,
        "train_epochs": TRAIN_EPOCHS,
        "quick_mode": QUICK_MODE,
    })
    print(f"[SAVE] Artefactos guardados en {ARTIFACT_DIR.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUACIÓN EN TEST
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_te, y_clf_te, y_reg_te,
                   clf_threshold: float = 0.5):
    pred_clf, pred_reg = model.predict(X_te, verbose=0)
    pred_clf = pred_clf.flatten()
    pred_reg = pred_reg.flatten()

    pred_labels = (pred_clf >= clf_threshold).astype(int)

    print("\n" + "="*55)
    print("MÉTRICAS EN TEST — CABEZA CLASIFICACIÓN")
    print("="*55)
    print(f"  Accuracy  : {accuracy_score(y_clf_te, pred_labels):.4f}")
    print(f"  Precision : {precision_score(y_clf_te, pred_labels, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_clf_te, pred_labels, zero_division=0):.4f}")
    print(f"  F1        : {f1_score(y_clf_te, pred_labels, zero_division=0):.4f}")
    try:
        print(f"  ROC-AUC   : {roc_auc_score(y_clf_te, pred_clf):.4f}")
    except Exception:
        pass

    mask_venta = y_clf_te == 1
    if mask_venta.sum() > 0:
        mae_reg = mean_absolute_error(y_reg_te[mask_venta], pred_reg[mask_venta])
        print(f"\nMÉTRICAS EN TEST — CABEZA REGRESIÓN (solo días con venta)")
        print(f"  MAE consumo: {mae_reg:.4f} unidades")

    return pred_clf, pred_reg


def sweep_thresholds(y_true, prob, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    rows = []
    for t in thresholds:
        pred = (prob >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 2),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def fit_calibrators(y_val, p_val):
    p_val = np.clip(np.array(p_val).astype(float), EPSILON, 1 - EPSILON)
    y_val = np.array(y_val).astype(int)

    platt = LogisticRegression(max_iter=1000)
    platt.fit(p_val.reshape(-1, 1), y_val)
    p_platt = platt.predict_proba(p_val.reshape(-1, 1))[:, 1]

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(p_val, y_val)
    p_iso = isotonic.predict(p_val)

    scores = {
        "raw": brier_score_loss(y_val, p_val),
        "platt": brier_score_loss(y_val, p_platt),
        "isotonic": brier_score_loss(y_val, p_iso),
    }
    best = min(scores, key=scores.get)
    calibrator = None if best == "raw" else (platt if best == "platt" else isotonic)
    return calibrator, best, scores


def apply_calibration(prob, calibrator, method):
    p = np.array(prob).astype(float).flatten()
    if calibrator is None or method == "raw":
        return np.clip(p, 0.0, 1.0)
    if method == "platt":
        return np.clip(calibrator.predict_proba(p.reshape(-1, 1))[:, 1], 0.0, 1.0)
    return np.clip(calibrator.predict(p), 0.0, 1.0)


pred_clf_val_raw, pred_reg_val = model.predict(X_val, verbose=0)
pred_clf_val_raw = pred_clf_val_raw.flatten()
calibrator, calibration_method, calibration_scores = fit_calibrators(y_clf_val, pred_clf_val_raw)

threshold_table = sweep_thresholds(
    y_clf_val,
    apply_calibration(pred_clf_val_raw, calibrator, calibration_method)
)
best_row = threshold_table.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
BEST_THRESHOLD = float(best_row["threshold"])

threshold_table.to_csv(ARTIFACT_DIR / "threshold_metrics.csv", index=False)
with open(ARTIFACT_DIR / "calibration_summary.json", "w", encoding="utf-8") as f:
    json.dump({
        "method": calibration_method,
        "brier_scores": calibration_scores,
        "best_threshold": BEST_THRESHOLD,
    }, f, ensure_ascii=False, indent=2)

if calibrator is not None:
    with open(ARTIFACT_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump({"method": calibration_method, "model": calibrator}, f)

pred_clf_te_raw, pred_reg_te = evaluate_model(model, X_te, y_clf_te, y_reg_te, clf_threshold=BEST_THRESHOLD)
pred_clf_te = apply_calibration(pred_clf_te_raw, calibrator, calibration_method)
test_labels = (pred_clf_te >= BEST_THRESHOLD).astype(int)

print(f"\n[THRESHOLD] Umbral óptimo por F1 en validación: {BEST_THRESHOLD:.2f}")
print(f"[CALIBRATION] Método seleccionado: {calibration_method} | Brier: {calibration_scores}")
print(f"[TEST] Precision={precision_score(y_clf_te, test_labels, zero_division=0):.4f} "
      f"Recall={recall_score(y_clf_te, test_labels, zero_division=0):.4f} "
      f"F1={f1_score(y_clf_te, test_labels, zero_division=0):.4f}")


def croston_sba(series: np.ndarray,
                alpha: float = 0.1,
                use_sba: bool = True) -> dict:
    series = np.array(series, dtype=float)
    z = series[series > 0]
    if len(z) == 0:
        return {"demand_rate": 0.0, "avg_qty": 0.0, "avg_interval": float("inf"), "prob_sale": 0.0}

    z_hat = z[0]
    p_hat = 1.0
    non_zero_idx = np.where(series > 0)[0]
    for i in range(1, len(non_zero_idx)):
        q = non_zero_idx[i] - non_zero_idx[i - 1]
        z_actual = series[non_zero_idx[i]]
        z_hat = alpha * z_actual + (1 - alpha) * z_hat
        p_hat = alpha * q + (1 - alpha) * p_hat

    factor = (1 - alpha / 2) if use_sba else 1.0
    demand_rate = (z_hat / p_hat) * factor
    return {
        "demand_rate": demand_rate,
        "avg_qty": z_hat,
        "avg_interval": p_hat,
        "prob_sale": min(1.0, 1.0 / p_hat)
    }


def calculate_stockout_day(consumos: np.ndarray, stock_inicial: float, horizon: int):
    stock = float(stock_inicial)
    for day, c in enumerate(consumos[:horizon], start=1):
        stock -= float(c)
        if stock <= 0:
            return day, 1
    return horizon + 1, 0


def simulate_stockout_probability(prob_sale, qty_mu, stock_inicial, n_sim=300, seed=0):
    rng = np.random.default_rng(seed)
    horizon = len(prob_sale)
    stockout_days = []
    for _ in range(n_sim):
        stock = float(stock_inicial)
        stockout_day = None
        for day in range(horizon):
            if rng.random() < prob_sale[day] and stock > 0:
                safe_mu = max(qty_mu[day], EPSILON)
                q = max(1, rng.poisson(safe_mu))
                stock -= min(stock, q)
                if stock <= 0:
                    stockout_day = day + 1
                    break
        if stockout_day is not None:
            stockout_days.append(stockout_day)
    prob = len(stockout_days) / n_sim if n_sim > 0 else 0.0
    return prob, stockout_days


def evaluate_business_benchmark(df_feat, model, scaler, feature_cols, seq_len,
                                calibrator=None, calibration_method="raw",
                                clf_threshold=0.5, horizon=30, n_simulations=300,
                                max_skus=0):
    summary_rows, daily_rows = [], []
    sku_iter = list(df_feat.groupby("sku"))
    if max_skus > 0:
        sku_iter = sku_iter[:max_skus]

    for idx, (sku, grp) in enumerate(sku_iter):
        grp = grp.sort_values("fecha").reset_index(drop=True)
        n = len(grp)
        cutoff = int(n * TRAIN_CUTOFF_RATIO)
        if cutoff < seq_len or cutoff >= n:
            continue

        horizon_eff = min(horizon, n - cutoff)
        if horizon_eff <= 0:
            continue

        train_series = grp["consumo"].values[:cutoff]
        test_cons = grp["consumo"].values[cutoff:cutoff + horizon_eff]
        test_dates = grp["fecha"].iloc[cutoff:cutoff + horizon_eff].values
        stock_ini = float(grp["stock"].iloc[cutoff - 1])

        feats = grp[feature_cols].values
        X_sku = []
        for i in range(cutoff, cutoff + horizon_eff):
            X_sku.append(feats[i - seq_len:i])
        X_sku = np.array(X_sku, dtype=np.float32)
        X_sku_sc = scaler.transform(X_sku.reshape(-1, len(feature_cols))).reshape(-1, seq_len, len(feature_cols))

        p_raw, q_pred = model.predict(X_sku_sc, verbose=0)
        p_gru = apply_calibration(p_raw.flatten(), calibrator, calibration_method)
        q_gru = np.maximum(0.0, q_pred.flatten())
        exp_cons_gru = p_gru * q_gru
        labels_gru = (p_gru >= clf_threshold).astype(int)

        sba = croston_sba(train_series, use_sba=True)
        p_croston = np.full(horizon_eff, sba["prob_sale"], dtype=float)
        q_croston = np.full(horizon_eff, max(0.0, sba["avg_qty"]), dtype=float)
        exp_cons_croston = p_croston * q_croston

        actual_day, actual_stockout = calculate_stockout_day(test_cons, stock_ini, horizon_eff)
        pred_day_gru, _ = calculate_stockout_day(exp_cons_gru, stock_ini, horizon_eff)
        pred_day_croston, _ = calculate_stockout_day(exp_cons_croston, stock_ini, horizon_eff)

        sku_seed = int(hashlib.sha256(str(sku).encode("utf-8")).hexdigest()[:8], 16)
        prob_gru, dist_gru = simulate_stockout_probability(
            p_gru, np.maximum(q_gru, EPSILON), stock_ini, n_sim=n_simulations, seed=sku_seed
        )
        prob_croston, dist_croston = simulate_stockout_probability(
            p_croston, np.maximum(q_croston, EPSILON), stock_ini, n_sim=n_simulations,
            seed=CROSTON_SEED_OFFSET + sku_seed
        )

        y_true_sale = (test_cons > 0).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_sale, labels_gru, labels=[0, 1]).ravel()
        if fp > fn:
            bias_label = "sobrepredice"
        elif fn > fp:
            bias_label = "subpredice"
        else:
            bias_label = "balanceado"

        summary_rows.append({
            "sku": sku,
            "horizon": horizon_eff,
            "actual_cum_consumo_30d": float(test_cons.sum()),
            "pred_cum_consumo_gru_30d": float(exp_cons_gru.sum()),
            "pred_cum_consumo_croston_30d": float(exp_cons_croston.sum()),
            "mae_cum_consumo_gru": abs(float(test_cons.sum() - exp_cons_gru.sum())),
            "mae_cum_consumo_croston": abs(float(test_cons.sum() - exp_cons_croston.sum())),
            "actual_dia_agotamiento": actual_day,
            "pred_dia_agotamiento_gru": pred_day_gru,
            "pred_dia_agotamiento_croston": pred_day_croston,
            "err_dia_agotamiento_gru": abs(pred_day_gru - actual_day),
            "err_dia_agotamiento_croston": abs(pred_day_croston - actual_day),
            "actual_stockout_flag": actual_stockout,
            "prob_stockout_gru": prob_gru,
            "prob_stockout_croston": prob_croston,
            "err_prob_stockout_gru": abs(prob_gru - actual_stockout),
            "err_prob_stockout_croston": abs(prob_croston - actual_stockout),
            "precision_gru": precision_score(y_true_sale, labels_gru, zero_division=0),
            "recall_gru": recall_score(y_true_sale, labels_gru, zero_division=0),
            "f1_gru": f1_score(y_true_sale, labels_gru, zero_division=0),
            "fp_gru": fp,
            "fn_gru": fn,
            "sesgo_gru": bias_label,
            "stockout_days_gru": ";".join(map(str, dist_gru)),
            "stockout_days_croston": ";".join(map(str, dist_croston)),
        })

        for j in range(horizon_eff):
            daily_rows.append({
                "sku": sku,
                "fecha": pd.to_datetime(test_dates[j]),
                "consumo_real": float(test_cons[j]),
                "prob_venta_gru": float(p_gru[j]),
                "qty_gru": float(q_gru[j]),
                "consumo_esperado_gru": float(exp_cons_gru[j]),
                "prob_venta_croston": float(p_croston[j]),
                "qty_croston": float(q_croston[j]),
                "consumo_esperado_croston": float(exp_cons_croston[j]),
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(daily_rows)


print("\n[BENCHMARK] Comparando GRU vs Croston/SBA con métricas de negocio...")
df_compare, df_daily_pred = evaluate_business_benchmark(
    df_feat, model, scaler, FEATURE_COLS, SEQ_LEN,
    calibrator=calibrator, calibration_method=calibration_method,
    clf_threshold=BEST_THRESHOLD, horizon=30, n_simulations=MC_ALL_SIMS,
    max_skus=BENCHMARK_MAX_SKUS
)

if df_compare.empty:
    raise ValueError(
        "No se pudo construir benchmark de negocio: no hay SKUs evaluables "
        "(causas comunes: cutoff<seq_len, SKUs sin tramo de test o historial insuficiente)."
    )

df_compare.to_csv(ARTIFACT_DIR / "benchmark_business_metrics.csv", index=False)
df_daily_pred.to_csv(ARTIFACT_DIR / "predicciones_por_sku.csv", index=False)

print("\n" + "=" * 75)
print("COMPARACIÓN GRU vs CROSTON/SBA (métricas de negocio)")
print("=" * 75)
print(df_compare[[
    "sku", "mae_cum_consumo_gru", "mae_cum_consumo_croston",
    "err_dia_agotamiento_gru", "err_dia_agotamiento_croston",
    "err_prob_stockout_gru", "err_prob_stockout_croston"
]].round(4).to_string(index=False))
print(f"\nMAE consumo 30d      | GRU: {df_compare['mae_cum_consumo_gru'].mean():.4f} | "
      f"Croston: {df_compare['mae_cum_consumo_croston'].mean():.4f}")
print(f"Error día agotamiento| GRU: {df_compare['err_dia_agotamiento_gru'].mean():.4f} | "
      f"Croston: {df_compare['err_dia_agotamiento_croston'].mean():.4f}")
print(f"Error prob stockout  | GRU: {df_compare['err_prob_stockout_gru'].mean():.4f} | "
      f"Croston: {df_compare['err_prob_stockout_croston'].mean():.4f}")


def montecarlo_stockout(sku_id: str,
                        df_feat: pd.DataFrame,
                        model,
                        scaler,
                        feature_cols: list,
                        seq_len: int = 14,
                        horizon: int = 30,
                        n_simulations: int = 1000,
                        calibrator_obj=None,
                        calibration_method_name="raw",
                        seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    grp = df_feat[df_feat["sku"] == sku_id].sort_values("fecha").reset_index(drop=True)
    if len(grp) < seq_len:
        return {"error": "Datos insuficientes para la simulación"}

    stock_inicial = float(grp["stock"].iloc[-1])
    feats_hist = grp[feature_cols].values
    stockout_days, final_stocks = [], []

    for _ in range(n_simulations):
        stock = stock_inicial
        window = feats_hist[-seq_len:].copy().tolist()
        stockout_day = None
        for day in range(1, horizon + 1):
            w_arr = np.array(window, dtype=np.float32)
            w_sc = scaler.transform(w_arr).reshape(1, seq_len, len(feature_cols)).astype(np.float32)
            p_clf, qty_pred = model.predict(w_sc, verbose=0)
            p_venta = apply_calibration([float(p_clf[0, 0])], calibrator_obj, calibration_method_name)[0]
            qty_mu = max(0.0, float(qty_pred[0, 0]))

            hay_venta = rng.random() < p_venta
            consumo_sim = 0
            if hay_venta and stock > 0:
                consumo_sim = max(1, rng.poisson(qty_mu if qty_mu > 0 else 1))
                consumo_sim = min(consumo_sim, stock)
                stock -= consumo_sim
                if stock <= 0:
                    stockout_day = day
                    stock = 0
                    break

            new_row = window[-1].copy()
            new_row[0] = consumo_sim
            new_row[1] = sum([r[0] for r in window[-6:]]) + consumo_sim
            new_row[6] = stock
            window.pop(0)
            window.append(new_row)

        final_stocks.append(stock)
        if stockout_day is not None:
            stockout_days.append(stockout_day)

    prob_stockout = len(stockout_days) / n_simulations
    avg_stockout_day = np.mean(stockout_days) if stockout_days else None
    return {
        "sku": sku_id,
        "stock_inicial": stock_inicial,
        "prob_stockout_30d": round(prob_stockout, 4),
        "dia_esperado_agotamiento": round(avg_stockout_day, 1) if avg_stockout_day else "N/A",
        "stock_final_esperado": round(float(np.mean(final_stocks)), 1),
        "n_simulaciones": n_simulations,
        "n_stockouts": len(stockout_days),
        "stockout_day_distribution": stockout_days,
        "alerta": "🔴 CRÍTICO" if prob_stockout > 0.7 else ("🟡 MODERADO" if prob_stockout > 0.3 else "🟢 BAJO"),
    }


TARGET_SKU = str(df_feat["sku"].iloc[0])
target_df = df_feat[df_feat["sku"] == TARGET_SKU].copy()
resultado = montecarlo_stockout(
    sku_id=TARGET_SKU, df_feat=df_feat, model=model, scaler=scaler,
    feature_cols=FEATURE_COLS, seq_len=SEQ_LEN, horizon=30,
    n_simulations=MC_TARGET_SIMS, calibrator_obj=calibrator,
    calibration_method_name=calibration_method
)
if "error" in resultado:
    raise ValueError(f"No se pudo simular {TARGET_SKU}: {resultado['error']}")


def run_all_skus(df_feat, model, scaler, feature_cols, seq_len, n_simulations=300,
                 calibrator_obj=None, calibration_method_name="raw"):
    resultados = []
    skus = df_feat["sku"].unique()
    print(f"\n[MC] Simulando {len(skus)} SKUs con {n_simulations} corridas c/u...")
    for i, sku in enumerate(skus):
        r = montecarlo_stockout(
            sku_id=sku, df_feat=df_feat, model=model, scaler=scaler,
            feature_cols=feature_cols, seq_len=seq_len, horizon=30,
            n_simulations=n_simulations, calibrator_obj=calibrator_obj,
            calibration_method_name=calibration_method_name, seed=i
        )
        if "error" not in r:
            resultados.append(r)
    df_res = pd.DataFrame(resultados).sort_values("prob_stockout_30d", ascending=False)
    return df_res


df_resultados = run_all_skus(
    df_feat, model, scaler, FEATURE_COLS, SEQ_LEN,
    n_simulations=MC_ALL_SIMS, calibrator_obj=calibrator, calibration_method_name=calibration_method
)
df_resultados.to_csv(ARTIFACT_DIR / "stockout_riesgo_skus.csv", index=False)

sku_analysis = df_compare[["sku", "precision_gru", "recall_gru", "f1_gru", "fp_gru", "fn_gru", "sesgo_gru"]].copy()
sku_analysis.to_csv(ARTIFACT_DIR / "analisis_sesgo_por_sku.csv", index=False)

print("\nTop SKUs sobrepredicción/subpredicción:")
print(sku_analysis.sort_values(["fp_gru", "fn_gru"], ascending=False).head(10).to_string(index=False))

# Visualizaciones principales
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("GRU + Croston/SBA — Métricas de negocio para stockout", fontsize=14, fontweight="bold")

ax = axes[0, 0]
if history is not None:
    ax.plot(history.history["loss"], label="Train loss", color="#3b82f6")
    ax.plot(history.history["val_loss"], label="Val loss", color="#ef4444", linestyle="--")
else:
    ax.text(0.5, 0.5, "Modelo cargado desde artefactos\n(sin reentrenar)", ha="center", va="center")
ax.set_title("Pérdida de entrenamiento"); ax.set_xlabel("Época"); ax.set_ylabel("Loss"); ax.grid(alpha=0.3); ax.legend()

ax = axes[0, 1]
pr, rc, _ = precision_recall_curve(y_clf_te.astype(int), pred_clf_te)
ax.plot(rc, pr, color="#10b981", label="PR curve GRU")
ax.set_title("Curva Precision-Recall (test)"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.grid(alpha=0.3); ax.legend()

ax = axes[1, 0]
ax.hist(df_compare["err_dia_agotamiento_gru"], alpha=0.6, label="GRU", color="#22c55e")
ax.hist(df_compare["err_dia_agotamiento_croston"], alpha=0.6, label="Croston/SBA", color="#6366f1")
ax.set_title("Histograma error día agotamiento"); ax.set_xlabel("Error absoluto (días)"); ax.grid(alpha=0.3); ax.legend()

ax = axes[1, 1]
daily_target = df_daily_pred[df_daily_pred["sku"] == TARGET_SKU].copy()
if not daily_target.empty:
    daily_target = daily_target.sort_values("fecha")
    ax.plot(daily_target["fecha"], daily_target["consumo_real"].cumsum(), label="Consumo real acumulado", color="#111827")
    ax.plot(daily_target["fecha"], daily_target["consumo_esperado_gru"].cumsum(), label="GRU acumulado", color="#22c55e")
    ax.plot(daily_target["fecha"], daily_target["consumo_esperado_croston"].cumsum(), label="Croston acumulado", color="#6366f1")
ax.set_title(f"Consumo real vs simulado ({TARGET_SKU})"); ax.grid(alpha=0.3); ax.legend()

plt.tight_layout()
plt.savefig(ARTIFACT_DIR / "resultados_inventario.png", dpi=150, bbox_inches="tight")
plt.close()

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
if not daily_target.empty:
    stock_ini = float(target_df["stock"].iloc[int(len(target_df) * TRAIN_CUTOFF_RATIO) - 1])
    real_stock = stock_ini - daily_target["consumo_real"].cumsum()
    gru_stock = stock_ini - daily_target["consumo_esperado_gru"].cumsum()
    axes2[0].plot(daily_target["fecha"], real_stock, label="Stock real", color="#111827")
    axes2[0].plot(daily_target["fecha"], gru_stock, label="Stock simulado GRU", color="#22c55e")
axes2[0].set_title(f"Stock real vs simulado ({TARGET_SKU})"); axes2[0].grid(alpha=0.3); axes2[0].legend()

sesgo_counts = sku_analysis["sesgo_gru"].value_counts()
bias_color_map = {"sobrepredice": "#ef4444", "subpredice": "#f59e0b", "balanceado": "#22c55e"}
axes2[1].bar(
    sesgo_counts.index,
    sesgo_counts.values,
    color=[bias_color_map.get(cat, "#6b7280") for cat in sesgo_counts.index]
)
axes2[1].set_title("Sesgo de clasificación GRU por SKU"); axes2[1].grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(ARTIFACT_DIR / "analisis_por_sku.png", dpi=150, bbox_inches="tight")
plt.close()

summary = {
    "artifact_dir": str(ARTIFACT_DIR.resolve()),
    "best_threshold": BEST_THRESHOLD,
    "calibration_method": calibration_method,
    "mean_mae_cum_30d_gru": float(df_compare["mae_cum_consumo_gru"].mean()),
    "mean_mae_cum_30d_croston": float(df_compare["mae_cum_consumo_croston"].mean()),
    "mean_err_day_gru": float(df_compare["err_dia_agotamiento_gru"].mean()),
    "mean_err_day_croston": float(df_compare["err_dia_agotamiento_croston"].mean()),
    "mean_err_prob_gru": float(df_compare["err_prob_stockout_gru"].mean()),
    "mean_err_prob_croston": float(df_compare["err_prob_stockout_croston"].mean()),
}
with open(ARTIFACT_DIR / "run_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n[OK] Artefactos, métricas y gráficas guardados en: {ARTIFACT_DIR.resolve()}")

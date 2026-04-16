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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, mean_absolute_error)
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import Adam

# Loader imports are intentionally lazy in the data-source block so that
# Colab runs from /data do not require MySQL dependencies.

tf.random.set_seed(42)
np.random.seed(42)

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

X_tr  = scaler.fit_transform(X_tr.reshape(-1, n_feat)).reshape(-1, seq, n_feat).astype(np.float32)
X_val = scaler.transform(X_val.reshape(-1, n_feat)).reshape(-1, seq, n_feat).astype(np.float32)
X_te  = scaler.transform(X_te.reshape(-1, n_feat)).reshape(-1, seq, n_feat).astype(np.float32)


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
history = model.fit(
    X_tr,
    {"output_clf": y_clf_tr, "output_reg": y_reg_tr},
    validation_data=(X_val, {"output_clf": y_clf_val, "output_reg": y_reg_val}),
    epochs=60,
    batch_size=128,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)


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


pred_clf_te, pred_reg_te = evaluate_model(model, X_te, y_clf_te, y_reg_te)


# ─────────────────────────────────────────────────────────────────────────────
# 8. BENCHMARK: CROSTON / SBA
# ─────────────────────────────────────────────────────────────────────────────
# Croston descompone la demanda intermitente en:
#   - z_t: tamaño de la demanda cuando ocurre
#   - p_t: intervalo entre demandas
# SBA (Syntetos-Boylan Approximation) agrega un factor corrector de sesgo.
# ─────────────────────────────────────────────────────────────────────────────

def croston_sba(series: np.ndarray,
                alpha: float = 0.1,
                use_sba: bool = True) -> dict:
    """
    Ajusta Croston o SBA sobre una serie de consumo histórico.
    Devuelve la demanda estimada por período.
    """
    series = np.array(series, dtype=float)
    n = len(series)

    z = series[series > 0]
    if len(z) == 0:
        return {
            "demand_rate": 0.0,
            "avg_qty": 0.0,
            "avg_interval": float("inf"),
            "prob_sale": 0.0,
            "intervals": [],
        }

    # Inicialización
    z_hat = z[0]
    p_hat = 1.0

    non_zero_idx = np.where(series > 0)[0]
    intervals = []
    if len(non_zero_idx) > 1:
        intervals = np.diff(non_zero_idx).tolist()

    for i in range(1, len(non_zero_idx)):
        q = non_zero_idx[i] - non_zero_idx[i-1]
        z_actual = series[non_zero_idx[i]]
        z_hat = alpha * z_actual + (1 - alpha) * z_hat
        p_hat = alpha * q       + (1 - alpha) * p_hat

    # Factor SBA: corrige el sesgo de sobreestimación de Croston
    factor = (1 - alpha / 2) if use_sba else 1.0
    demand_rate = (z_hat / p_hat) * factor

    return {
        "demand_rate": demand_rate,   # unidades esperadas por período
        "avg_qty":     z_hat,
        "avg_interval": p_hat,
        "prob_sale":   min(1.0, 1.0 / p_hat)  # P(venta en un día)
    }


def evaluate_croston_vs_gru(df_feat, model, scaler, feature_cols,
                             seq_len, clf_threshold=0.5):
    """
    Compara Croston/SBA vs GRU en términos de accuracy de clasificación
    en el último 15% temporal de cada SKU.
    """
    resultados = []

    for sku, grp in df_feat.groupby("sku"):
        grp = grp.sort_values("fecha").reset_index(drop=True)
        n = len(grp)
        cutoff = int(n * 0.85)

        train_series = grp["consumo"].values[:cutoff]
        test_clf     = (grp["consumo"].values[cutoff:] > 0).astype(int)
        if len(test_clf) == 0:
            continue

        # Croston: predice la misma probabilidad para todo el período de test
        sba = croston_sba(train_series, use_sba=True)
        pred_croston = np.full(len(test_clf), sba.get("prob_sale", 0.0))
        labels_croston = (pred_croston >= clf_threshold).astype(int)
        acc_croston = accuracy_score(test_clf, labels_croston)

        # GRU: solo si hay suficientes datos
        if cutoff < seq_len:
            continue
        feats = grp[feature_cols].values
        X_sku = []
        for i in range(cutoff, min(cutoff + len(test_clf), n)):
            if i >= seq_len:
                X_sku.append(feats[i - seq_len:i])
        if not X_sku:
            continue
        X_sku_arr = np.array(X_sku, dtype=np.float32)
        X_sku_sc  = scaler.transform(X_sku_arr.reshape(-1, len(feature_cols)))
        X_sku_sc  = X_sku_sc.reshape(-1, seq_len, len(feature_cols)).astype(np.float32)

        p_clf, _ = model.predict(X_sku_sc, verbose=0)
        labels_gru = (p_clf.flatten() >= clf_threshold).astype(int)
        acc_gru = accuracy_score(test_clf[:len(labels_gru)], labels_gru)

        resultados.append({
            "sku": sku,
            "acc_croston_sba": acc_croston,
            "acc_gru":         acc_gru,
            "n_test":          len(test_clf),
            "prob_sale_sba":   sba["prob_sale"]
        })

    return pd.DataFrame(resultados)


print("\n[BENCHMARK] Comparando Croston/SBA vs GRU...")
df_compare = evaluate_croston_vs_gru(df_feat, model, scaler, FEATURE_COLS, SEQ_LEN)

print("\n" + "="*55)
print("COMPARACIÓN CROSTON/SBA vs GRU (accuracy por SKU)")
print("="*55)
print(df_compare.set_index("sku").round(4).to_string())
print(f"\nPromedio Croston/SBA accuracy: {df_compare['acc_croston_sba'].mean():.4f}")
print(f"Promedio GRU accuracy        : {df_compare['acc_gru'].mean():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. SIMULACIÓN MONTE CARLO A 30 DÍAS
# ─────────────────────────────────────────────────────────────────────────────

def montecarlo_stockout(sku_id: str,
                        df_feat: pd.DataFrame,
                        model,
                        scaler,
                        feature_cols: list,
                        seq_len: int = 14,
                        horizon: int = 30,
                        n_simulations: int = 1000,
                        clf_threshold: float = 0.4,
                        seed: int = 0) -> dict:
    """
    Simula 30 días hacia adelante para un SKU usando la GRU.

    Lógica de cada simulación:
      1. Tomar la ventana histórica del SKU (últimos seq_len días).
      2. Para cada día futuro d:
         a. Predecir P(venta) y cantidad esperada con la GRU.
         b. Samplear si hubo venta: Bernoulli(P(venta)).
         c. Si hubo venta, samplear cantidad: Poisson(qty_pred).
         d. Descontar del stock. Si stock <= 0 → stockout en día d.
      3. Repetir n_simulations veces.
      4. Calcular:
         - prob_stockout: fracción de simulaciones con stockout
         - día esperado de stockout (promedio condicional)
         - stock final esperado
    """
    rng = np.random.default_rng(seed)

    grp = df_feat[df_feat["sku"] == sku_id].sort_values("fecha").reset_index(drop=True)
    if len(grp) < seq_len:
        return {"error": "Datos insuficientes para la simulación"}

    stock_inicial = grp["stock"].iloc[-1]
    feats_hist = grp[feature_cols].values  # (n_dias, n_feat)

    stockout_days   = []
    final_stocks    = []
    n_stockouts     = 0

    for sim in range(n_simulations):
        stock = stock_inicial
        # Ventana deslizante de features (copia para no modificar original)
        window = feats_hist[-seq_len:].copy().tolist()
        stockout_day = None

        for day in range(1, horizon + 1):
            # Escalar ventana
            w_arr = np.array(window, dtype=np.float32)
            w_sc  = scaler.transform(w_arr).reshape(1, seq_len, len(feature_cols))
            w_sc  = w_sc.astype(np.float32)

            p_clf, qty_pred = model.predict(w_sc, verbose=0)
            p_venta = float(p_clf[0, 0])
            qty_mu  = max(0.0, float(qty_pred[0, 0]))

            # Samplear evento de venta
            hay_venta = rng.random() < p_venta

            consumo_sim = 0
            if hay_venta and stock > 0:
                # Poisson con media qty_mu, al menos 1
                consumo_sim = max(1, rng.poisson(qty_mu if qty_mu > 0 else 1))
                consumo_sim = min(consumo_sim, stock)
                stock = stock - consumo_sim
                if stock <= 0:
                    stockout_day = day
                    stock = 0
                    break  # agotado

            # Actualizar ventana: desplazar y agregar nuevo feature vector
            # (aproximación: reciclamos el último vector con consumo actualizado)
            new_row = window[-1].copy()
            # Índices de features: consumo_1d=0, consumo_7d=1, stock=6
            new_row[0] = consumo_sim                      # consumo_1d
            new_row[1] = sum([r[0] for r in window[-6:]]) + consumo_sim  # consumo_7d approx
            new_row[6] = stock                             # stock_actual
            window.pop(0)
            window.append(new_row)

        final_stocks.append(stock)
        if stockout_day is not None:
            stockout_days.append(stockout_day)
            n_stockouts += 1

    prob_stockout = n_stockouts / n_simulations
    avg_stockout_day = np.mean(stockout_days) if stockout_days else None
    avg_final_stock  = np.mean(final_stocks)

    return {
        "sku":              sku_id,
        "stock_inicial":    stock_inicial,
        "prob_stockout_30d": round(prob_stockout, 4),
        "dia_esperado_agotamiento": round(avg_stockout_day, 1) if avg_stockout_day else "N/A",
        "stock_final_esperado": round(avg_final_stock, 1),
        "n_simulaciones":   n_simulations,
        "n_stockouts":      n_stockouts,
        "alerta": (
            "🔴 CRÍTICO"  if prob_stockout > 0.7 else
            "🟡 MODERADO" if prob_stockout > 0.3 else
            "🟢 BAJO"
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# 10. DEMO: PROBAR CON UN SKU ESPECÍFICO
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SKU = str(df_feat["sku"].iloc[0])

print(f"\n{'='*55}")
print(f"SIMULACIÓN MONTE CARLO — {TARGET_SKU}")
print(f"{'='*55}")

resultado = montecarlo_stockout(
    sku_id        = TARGET_SKU,
    df_feat       = df_feat,
    model         = model,
    scaler        = scaler,
    feature_cols  = FEATURE_COLS,
    seq_len       = SEQ_LEN,
    horizon       = 30,
    n_simulations = 500,    # usa 1000 en producción; 500 para demo rápido
    clf_threshold = 0.4
)

if "error" in resultado:
    raise ValueError(f"No se pudo simular {TARGET_SKU}: {resultado['error']}")

print(f"\n  SKU                     : {resultado['sku']}")
print(f"  Stock inicial           : {resultado['stock_inicial']}")
print(f"  Prob. agotamiento 30d   : {resultado['prob_stockout_30d']:.1%}")
print(f"  Día esperado agotamiento: {resultado['dia_esperado_agotamiento']}")
print(f"  Stock final esperado    : {resultado['stock_final_esperado']:.1f} unidades")
print(f"  Nivel de alerta         : {resultado['alerta']}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. RESUMEN PARA TODOS LOS SKUs
# ─────────────────────────────────────────────────────────────────────────────

def run_all_skus(df_feat, model, scaler, feature_cols, seq_len,
                 n_simulations=300):
    resultados = []
    skus = df_feat["sku"].unique()
    print(f"\n[MC] Simulando {len(skus)} SKUs con {n_simulations} corridas c/u...")
    for sku in skus:
        r = montecarlo_stockout(
            sku_id=sku, df_feat=df_feat, model=model, scaler=scaler,
            feature_cols=feature_cols, seq_len=seq_len,
            horizon=30, n_simulations=n_simulations
        )
        if "error" not in r:
            resultados.append(r)

    df_res = pd.DataFrame(resultados)
    df_res = df_res.sort_values("prob_stockout_30d", ascending=False)
    return df_res


df_resultados = run_all_skus(df_feat, model, scaler, FEATURE_COLS, SEQ_LEN,
                              n_simulations=300)

print("\n" + "="*70)
print("RANKING DE RIESGO DE AGOTAMIENTO (todos los SKUs)")
print("="*70)
print(df_resultados[["sku", "stock_inicial", "prob_stockout_30d",
                      "dia_esperado_agotamiento", "stock_final_esperado",
                      "alerta"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 12. VISUALIZACIONES
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("GRU + Croston/SBA — Predicción de Agotamiento de Inventario",
             fontsize=14, fontweight="bold")

# 12.1 Curva de pérdida
ax = axes[0, 0]
ax.plot(history.history["loss"],     label="Train loss", color="#3b82f6")
ax.plot(history.history["val_loss"], label="Val loss",   color="#ef4444", linestyle="--")
ax.set_title("Pérdida durante entrenamiento"); ax.set_xlabel("Época"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(alpha=0.3)

# 12.2 Accuracy clasificación
ax = axes[0, 1]
ax.plot(history.history["output_clf_accuracy"],
        label="Train acc", color="#10b981")
ax.plot(history.history["val_output_clf_accuracy"],
        label="Val acc",   color="#f59e0b", linestyle="--")
ax.set_title("Accuracy clasificación"); ax.set_xlabel("Época")
ax.legend(); ax.grid(alpha=0.3)

# 12.3 Comparación Croston vs GRU
ax = axes[1, 0]
x = np.arange(len(df_compare))
width = 0.35
ax.bar(x - width/2, df_compare["acc_croston_sba"], width,
       label="Croston/SBA", color="#6366f1", alpha=0.8)
ax.bar(x + width/2, df_compare["acc_gru"],         width,
       label="GRU",         color="#22c55e", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(df_compare["sku"], rotation=45, ha="right", fontsize=7)
ax.set_title("Accuracy por SKU: Croston/SBA vs GRU"); ax.legend(); ax.grid(alpha=0.3, axis="y")

# 12.4 Distribución probabilidades de stockout
ax = axes[1, 1]
colors = df_resultados["alerta"].map({
    "🔴 CRÍTICO": "#ef4444", "🟡 MODERADO": "#f59e0b", "🟢 BAJO": "#22c55e"
})
ax.barh(df_resultados["sku"],
        df_resultados["prob_stockout_30d"],
        color=colors.values)
ax.axvline(0.3, color="#f59e0b", linestyle="--", alpha=0.7, label="Umbral moderado")
ax.axvline(0.7, color="#ef4444", linestyle="--", alpha=0.7, label="Umbral crítico")
ax.set_xlabel("Probabilidad de agotamiento (30d)")
ax.set_title("Riesgo de stockout por SKU"); ax.legend(); ax.grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("resultados_inventario.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n[OK] Gráfica guardada en resultados_inventario.png")

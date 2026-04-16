# Proyecto de forecasting de stockout (MySQL + GRU)

Este proyecto ahora tiene un flujo simple:

1. Subir Excels de `data/` a MySQL.
2. Leer esos snapshots diarios desde MySQL.
3. Entrenar y correr el modelo de `stockout_forecasting.py` con datos reales.

## Archivos principales

- `import_excel.py`: ingesta de archivos `dia_mes_anio.xlsx` a MySQL.
- `mysql_inventory_loader.py`: normaliza columnas, rellena dias faltantes e infiere `consumo`.
- `stockout_forecasting.py`: entrenamiento GRU, benchmark Croston/SBA y simulacion Monte Carlo.

## Dependencias

```powershell
python -m pip install -r requirements.txt
```

## 1) Cargar Excels a MySQL

Conexion usada (editable en `import_excel.py`):
- host: `localhost`
- puerto: `3306`
- db: `inventario_db`
- tabla: `inventario`

Subida por defecto en modo `append`:

```powershell
python import_excel.py
```

Si quieres reemplazar la tabla completa:

```powershell
$env:MYSQL_IMPORT_MODE = "replace"
python import_excel.py
```

## 2) Entrenar forecasting con datos reales

Ahora `stockout_forecasting.py` puede cargar de:

- `excel` (por defecto): lee `*.xlsx` desde carpeta `data/`.
- `mysql`: usa la tabla MySQL.
- `synthetic`: fuerza datos simulados.

### Fuente Excel (recomendado para Colab)

```powershell
$env:STOCKOUT_DATA_SOURCE = "excel"
$env:STOCKOUT_DATA_DIR = "data"
python stockout_forecasting.py
```

En Google Colab normalmente:

```python
import os
os.environ["STOCKOUT_DATA_SOURCE"] = "excel"
os.environ["STOCKOUT_DATA_DIR"] = "/content/data"
```

Luego ejecutas tu script.

### Fuente MySQL (opcional)

```powershell
$env:STOCKOUT_DATA_SOURCE = "mysql"
python stockout_forecasting.py
```

### Usar solo ultimos 30 dias

```powershell
$env:STOCKOUT_ALL_HISTORY = "0"
$env:STOCKOUT_DAYS_BACK = "30"
python stockout_forecasting.py
```

### Forzar datos sinteticos

```powershell
$env:STOCKOUT_DATA_SOURCE = "synthetic"
python stockout_forecasting.py
```

### Modo rapido (debug)

Para probar sin esperar tanto (menos epochs, benchmark y Monte Carlo reducidos):

```powershell
$env:STOCKOUT_QUICK_MODE = "1"
python stockout_forecasting.py
```

Ajustes opcionales:

```powershell
$env:STOCKOUT_QUICK_MODE = "1"
$env:STOCKOUT_TRAIN_EPOCHS = "6"
$env:STOCKOUT_BENCHMARK_MAX_SKUS = "120"
$env:STOCKOUT_MC_TARGET_SIMS = "80"
$env:STOCKOUT_MC_ALL_SIMS = "40"
python stockout_forecasting.py
```

## Artefactos reproducibles (modelo + métricas)

El pipeline guarda automáticamente artefactos en `artifacts/`:

- `gru_model.keras`, `scaler.pkl`, `metadata.json`
  - Modelo GRU entrenado, scaler de features y metadatos de configuración.
- `threshold_metrics.csv`, `calibration_summary.json`, `calibrator.pkl`
  - Barrido de umbrales, calidad de calibración (Brier) y calibrador seleccionado.
- `predicciones_por_sku.csv`, `benchmark_business_metrics.csv`
  - Predicciones/probabilidades por día y métricas de negocio GRU vs Croston.
- `stockout_riesgo_skus.csv`, `analisis_sesgo_por_sku.csv`
  - Ranking de riesgo de quiebre y diagnóstico de sobre/subpredicción por SKU.
- `resultados_inventario.png`, `analisis_por_sku.png`, `run_summary.json`
  - Visualizaciones clave y resumen agregado de métricas finales.

Para reutilizar modelo/scaler sin reentrenar:

```powershell
$env:STOCKOUT_LOAD_MODEL = "1"
python stockout_forecasting.py
```

## Limpieza de SKUs con muchos ceros (sin romper la serie temporal)

`mysql_inventory_loader.py` ahora mantiene los dias en cero, pero elimina SKUs no informativos:

- Prefiltro: minimo de puntos historicos por SKU.
- Postfiltro: elimina SKUs sin ventas o con consumo promedio extremadamente bajo.

Variables opcionales:

```powershell
$env:STOCKOUT_MIN_POINTS_PER_SKU = "10"
$env:STOCKOUT_MIN_CONSUMO_MEAN = "0.01"
python stockout_forecasting.py
```

## Notas de datos

- Si faltan dias en MySQL, `mysql_inventory_loader.py` completa la serie diaria por SKU.
- `consumo` se infiere como caida diaria de stock (`stock_t-1 - stock_t`, truncado en 0).
- Si MySQL falla o no hay datos utiles, `stockout_forecasting.py` cae a datos sinteticos para no bloquear el flujo.

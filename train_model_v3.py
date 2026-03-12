import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import pyodbc
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_training_v3.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# KONFIGURACJA
# ---------------------------------------------------------
class Config:
    BASE_DIR = Path.cwd()
    SQL_DIR = BASE_DIR / "SQL"
    SQL_PATH_PROGNOZA = SQL_DIR / "prognozapogody.sql"
    SQL_PATH_WYKONANIE = SQL_DIR / "wykonanie.sql"

    OUTPUT_DIR = BASE_DIR / "output_v3"
    MODEL_PATH = OUTPUT_DIR / "model_korekty_slonca_v3.json"
    META_PATH = OUTPUT_DIR / "model_meta_v3.json"
    IMPORTANCE_PLOT_PATH = OUTPUT_DIR / "feature_importance.png"

    CONN_STR_PROGNOZA = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "Server=MISDWPPRD.GKPGE.PL;"
        "DATABASE=PGESA_MarketAnalytics;"
        "Trusted_Connection=yes;"
    )

    CONN_STR_WYKONANIE = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "Server=MISDWPPRD.GKPGE.PL;"
        "DATABASE=PGEEO_DDS;"
        "Trusted_Connection=yes;"
    )

# ---------------------------------------------------------
# FUNKCJE POMOCNICZE
# ---------------------------------------------------------

def load_sql(path, conn):
    logger.info(f"Ładowanie SQL z: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku SQL: {path}")
    with open(path, "r", encoding="utf-8") as f:
        q = f.read()
    with pyodbc.connect(conn) as c:
        return pd.read_sql(q, c)

def ensure_tz(ts):
    ts = pd.to_datetime(ts, errors="coerce")
    try:
        if ts.dt.tz is not None:
            return ts.dt.tz_convert("Europe/Warsaw")
    except:
        pass
    return ts.dt.tz_localize("UTC").dt.tz_convert("Europe/Warsaw")

def floor_to_hour_warsaw(ts):
    ts = ts.dt.tz_convert("UTC")
    ts = ts.dt.floor("h")
    return ts.dt.tz_convert("Europe/Warsaw")

def nrmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    scale = np.percentile(y_true, 95)
    return rmse, rmse / max(scale, 1e-6) * 100

def time_split(df, frac=0.2):
    cut = int(len(df) * (1 - frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    logger.info("=== START TRAINING MODEL PRO 2026 V3 - ADVANCED FEATURES ===")
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. LOAD DATA
    try:
        df_prog = load_sql(Config.SQL_PATH_PROGNOZA, Config.CONN_STR_PROGNOZA)
        df_wyk = load_sql(Config.SQL_PATH_WYKONANIE, Config.CONN_STR_WYKONANIE)
    except Exception as e:
        logger.error(f"Błąd podczas ładowania danych: {e}")
        return

    # 2. PROGNOZA & WYKONANIE PREPROCESSING
    logger.info("Przetwarzanie danych...")
    df_prog["ts"] = ensure_tz(df_prog["dataGodzinaCET"])
    df_prog = df_prog[df_prog["ts"].dt.minute == 0]
    df_prog["dataGodzinaCET"] = floor_to_hour_warsaw(df_prog["ts"])
    # Normalizacja nazw kolumn do lowercase dla spójności z predict i backtest
    df_prog.columns = [c.lower() for c in df_prog.columns]
    col_rad = "calkowitepromieniowanieslonecznenettogodzinowe"
    df_prog["Prognoza_Wm2"] = pd.to_numeric(df_prog[col_rad], errors="coerce").fillna(0.0) / 3600
    df_prog["temperatura"] = pd.to_numeric(df_prog["temperatura"], errors="coerce").fillna(0)
    df_prog["punkt"] = df_prog["punkt"].astype("string")
    df_prog = df_prog[["punkt", "dataGodzinaCET", "Prognoza_Wm2", "temperatura"]]

    df_wyk["ts"] = ensure_tz(df_wyk["Data"].astype(str) + " " + df_wyk["Czas"].astype(str))
    df_wyk = df_wyk[df_wyk["ts"].dt.minute == 0]
    df_wyk["dataGodzinaCET"] = floor_to_hour_warsaw(df_wyk["ts"])
    df_wyk_hour = df_wyk.groupby(["NazwaFarmy", "dataGodzinaCET"])["NaslonecznienieHistoria"].mean().reset_index()
    df_wyk_hour = df_wyk_hour.rename(columns={"NazwaFarmy": "punkt", "NaslonecznienieHistoria": "Actual_Wm2"})
    df_wyk_hour["punkt"] = df_wyk_hour["punkt"].astype("string")

    # 3. MERGE & CLEANUP
    df = df_prog.merge(df_wyk_hour, on=["punkt", "dataGodzinaCET"], how="inner")
    df = df.sort_values(["punkt", "dataGodzinaCET"])
    df = df.dropna(subset=["Actual_Wm2"])
    
    # Filtrowanie nocy - tylko gdy słońce realnie świeci
    df = df[df["Prognoza_Wm2"] > 0]
    df["Target"] = df["Actual_Wm2"] - df["Prognoza_Wm2"]
    df["Error"] = df["Actual_Wm2"] - df["Prognoza_Wm2"] # Identical to target in this case

    # 4. ADVANCED FEATURE ENGINEERING
    logger.info("Generowanie zaawansowanych cech...")
    
    # Cykliczność
    doy = df["dataGodzinaCET"].dt.dayofyear
    is_leap = df["dataGodzinaCET"].dt.is_leap_year
    year_len = np.where(is_leap, 366, 365)
    df["hour_sin"] = np.sin(2*np.pi * df["dataGodzinaCET"].dt.hour / 24)
    df["hour_cos"] = np.cos(2*np.pi * df["dataGodzinaCET"].dt.hour / 24)
    df["day_sin"] = np.sin(2*np.pi * doy / year_len)
    df["day_cos"] = np.cos(2*np.pi * doy / year_len)
    df["temperatura_24"] = df.groupby("punkt")["temperatura"].shift(24)

    # Rozbudowane Lagi
    lags = [1,2,3,6,12,24,48]
    lag_cols = []
    for lag in lags:
        c_target = f"lag_target_{lag}h"
        c_error = f"lag_error_{lag}h"
        df[c_target] = df.groupby("punkt")["Target"].shift(lag)
        df[c_error] = df.groupby("punkt")["Error"].shift(lag)
        lag_cols.extend([c_target, c_error])

    # --- NOWOŚĆ: STATYSTYKI KROCZĄCE (ROLLING WINDOWS) ---
    logger.info("Obliczanie statystyk kroczących...")
    rolling_windows = [3, 6, 12]
    rolling_cols = []
    
    for w in rolling_windows:
        # Średnia i odchylenie z błędu (Error)
        c_mean = f"rolling_mean_err_{w}h"
        c_std = f"rolling_std_err_{w}h"
        # Używamy shift(1), aby uniknąć data leakage (nie znamy błędu z obecnej godziny)
        df[c_mean] = df.groupby("punkt")["Error"].transform(lambda x: x.shift(1).rolling(w).mean())
        df[c_std] = df.groupby("punkt")["Error"].transform(lambda x: x.shift(1).rolling(w).std())
        rolling_cols.extend([c_mean, c_std])

    # 5. FINALNA LISTA CECH
    features = (
        ["Prognoza_Wm2", "punkt", "hour_sin", "hour_cos", "day_sin", "day_cos", "temperatura", "temperatura_24"]
        + lag_cols
        + rolling_cols
    )

    df = df.dropna(subset=features)
    df["punkt"] = df["punkt"].astype("category")
    logger.info(f"Liczba rekordów po generowaniu cech: {len(df)}")

    # 6. TRAIN/TEST SPLIT
    train, test = time_split(df)

    # 7. MODEL TRAINING
    logger.info("Rozpoczynanie treningu XGBoost (v3)...")
    model = xgb.XGBRegressor(
        n_estimators=5000, # Zmniejszone dla demo, w prod można dać więcej + early stopping
        learning_rate=0.01, # Nieco wyższy dla szybszego zbiegania w v3
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
        early_stopping_rounds=100
    )

    model.fit(
        train[features], train["Target"],
        eval_set=[(test[features], test["Target"])],
        verbose=100
    )

    # 8. EVALUATION
    test = test.copy()
    test["Pred"] = test["Prognoza_Wm2"] + model.predict(test[features])
    test["Pred"] = test["Pred"].clip(0, 1500)
    rmse_b, nrmse_b = nrmse(test["Actual_Wm2"], test["Prognoza_Wm2"])
    rmse_m, nrmse_m = nrmse(test["Actual_Wm2"], test["Pred"])
    logger.info(f"Baseline nRMSE: {nrmse_b:.2f}%, Model V3 nRMSE: {nrmse_m:.2f}% (Gain: {nrmse_m - nrmse_b:.2f} pp)")

    # 9. FEATURE IMPORTANCE VISUALIZATION
    logger.info("Generowanie wykresu istotności cech...")
    plt.figure(figsize=(10, 15))
    xgb.plot_importance(model, max_num_features=30, height=0.5)
    plt.title("Top 30 Features - Model V3")
    plt.savefig(str(Config.IMPORTANCE_PLOT_PATH))
    plt.close()

    # 10. SAVE
    logger.info(f"Zapisywanie modelu i metadanych v3...")
    model.save_model(str(Config.MODEL_PATH))
    meta = {
        "features": features,
        "punkt_categories": list(df["punkt"].cat.categories),
        "windows": rolling_windows,
        "metrics": {"baseline_nrmse": float(nrmse_b), "model_nrmse": float(nrmse_m)}
    }
    with open(Config.META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info("=== MODEL V3 READY ===")

if __name__ == "__main__":
    main()

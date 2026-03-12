import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import pyodbc
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_training.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# KONFIGURACJA (PROD-READY)
# ---------------------------------------------------------
class Config:
    # Używamy pathlib dla przenośności między Windows/Linux
    BASE_DIR = Path.cwd()
    
    # Ścieżki do skryptów SQL (można zmienić na relatywne)
    SQL_DIR = BASE_DIR / "SQL"
    SQL_PATH_PROGNOZA = SQL_DIR / "prognozapogody.sql"
    SQL_PATH_WYKONANIE = SQL_DIR / "wykonanie.sql"

    # Ścieżki wyjściowe modelu
    OUTPUT_DIR = BASE_DIR / "output"
    MODEL_PATH = OUTPUT_DIR / "model_korekty_slonca.json"
    META_PATH = OUTPUT_DIR / "model_meta.json"

    # Connection Strings (Warto przenieść do zmiennych środowiskowych w pełnym PROD)
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
    logger.info("=== START TRAINING MODEL PRO 2026 - IRRADIANCE CORRECTION ===")

    # Tworzenie folderu na wynik jeśli nie istnieje
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. LOAD DATA
    try:
        df_prog = load_sql(Config.SQL_PATH_PROGNOZA, Config.CONN_STR_PROGNOZA)
        df_wyk = load_sql(Config.SQL_PATH_WYKONANIE, Config.CONN_STR_WYKONANIE)
    except Exception as e:
        logger.error(f"Błąd podczas ładowania danych: {e}")
        return

    # 2. PROGNOZA
    logger.info("Przetwarzanie danych prognozy...")
    df_prog["ts"] = ensure_tz(df_prog["dataGodzinaCET"])
    df_prog = df_prog[df_prog["ts"].dt.minute == 0]
    df_prog["dataGodzinaCET"] = floor_to_hour_warsaw(df_prog["ts"])

    df_prog["Prognoza_Wm2"] = (
        df_prog["CalkowitePromieniowanieSloneczneNettoGodzinowe"] / 3600
    )

    df_prog["temperatura"] = pd.to_numeric(df_prog["temperatura"], errors="coerce").fillna(0)
    df_prog["punkt"] = df_prog["punkt"].astype("string")

    df_prog = df_prog[["punkt", "dataGodzinaCET", "Prognoza_Wm2", "temperatura"]]

    # 3. WYKONANIE
    logger.info("Przetwarzanie danych wykonania...")
    df_wyk["Data"] = df_wyk["Data"].astype(str)
    df_wyk["Czas"] = df_wyk["Czas"].astype(str)

    combined = df_wyk["Data"] + " " + df_wyk["Czas"]
    df_wyk["ts"] = ensure_tz(combined)

    df_wyk = df_wyk[df_wyk["ts"].dt.minute == 0]
    df_wyk["dataGodzinaCET"] = floor_to_hour_warsaw(df_wyk["ts"])

    df_wyk_hour = (
        df_wyk.groupby(["NazwaFarmy", "dataGodzinaCET"])["NaslonecznienieHistoria"]
        .mean()
        .reset_index()
        .rename(columns={
            "NazwaFarmy": "punkt",
            "NaslonecznienieHistoria": "Actual_Wm2"
        })
    )
    df_wyk_hour["punkt"] = df_wyk_hour["punkt"].astype("string")

    # 3B. ELIGIBILITY CHECK (3 MONTH HISTORY)
    hist_hours = df_wyk_hour.groupby("punkt")["dataGodzinaCET"].count()
    eligible_points = set(hist_hours[hist_hours >= 90*24].index)
    
    logger.info(f"Liczba punktów z odpowiednią historią: {len(eligible_points)}")
    df_prog = df_prog[df_prog["punkt"].isin(eligible_points)]
    df_wyk_hour = df_wyk_hour[df_wyk_hour["punkt"].isin(eligible_points)]

    # 4. MERGE
    df = df_prog.merge(df_wyk_hour, on=["punkt", "dataGodzinaCET"], how="inner")
    df = df.sort_values(["punkt", "dataGodzinaCET"])
    df = df.dropna(subset=["Actual_Wm2"])

    # --- OPTYMALIZACJA: FILTROWANIE NOCY ---
    # Model uczy się tylko wtedy, gdy prognozowane jest słońce (> 0 Wm2)
    morning_records = len(df)
    df = df[df["Prognoza_Wm2"] > 0]
    logger.info(f"Filtrowanie nocy: usunięto {morning_records - len(df)} rekordów nocnych.")

    df["Target"] = df["Actual_Wm2"] - df["Prognoza_Wm2"]

    # 5. FEATURES
    logger.info("Generowanie cech...")
    doy = df["dataGodzinaCET"].dt.dayofyear
    is_leap = df["dataGodzinaCET"].dt.is_leap_year
    year_len = np.where(is_leap, 366, 365)

    df["hour_sin"] = np.sin(2*np.pi * df["dataGodzinaCET"].dt.hour / 24)
    df["hour_cos"] = np.cos(2*np.pi * df["dataGodzinaCET"].dt.hour / 24)
    df["day_sin"] = np.sin(2*np.pi * doy / year_len)
    df["day_cos"] = np.cos(2*np.pi * doy / year_len)

    df["temperatura_24"] = df.groupby("punkt")["temperatura"].shift(24)

    # Parametry lagów - Uwaga: należy sprawdzić opóźnienie danych SCADA w prod!
    lags = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,24,30,36,48]
    
    # Lag TARGET (poprawki)
    for lag in lags:
        df[f"lag_{lag}h"] = df.groupby("punkt")["Target"].shift(lag)
    lag_cols = [f"lag_{l}h" for l in lags]

    # Lag ERROR (odchylenia)
    df["Error"] = df["Actual_Wm2"] - df["Prognoza_Wm2"]
    for lag in lags:
        df[f"error_lag_{lag}h"] = df.groupby("punkt")["Error"].shift(lag)
    error_lag_cols = [f"error_lag_{l}h" for l in lags]

    # Lag FORECAST (historia prognozy)
    for lag in lags:
        df[f"forecast_lag_{lag}h"] = df.groupby("punkt")["Prognoza_Wm2"].shift(lag)
    forecast_lag_cols = [f"forecast_lag_{l}h" for l in lags]

    # Usuwanie NaN powstałych przez lagi
    df = df.dropna(subset=lag_cols + error_lag_cols + forecast_lag_cols + ["temperatura_24"])
    df = df.groupby("punkt").apply(lambda g: g.iloc[48:]).reset_index(drop=True)
    df["punkt"] = df["punkt"].astype("category")

    logger.info(f"Final training records: {len(df)}")

    # 6. TRAIN/TEST SPLIT
    train, test = time_split(df)

    features = (
        ["Prognoza_Wm2", "punkt",
         "hour_sin", "hour_cos",
         "day_sin", "day_cos",
         "temperatura", "temperatura_24"]
        + lag_cols
        + error_lag_cols
        + forecast_lag_cols
    )

    # 7. MODEL
    logger.info("Rozpoczynanie treningu XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=100000,
        learning_rate=0.001,
        max_depth=8,
        reg_alpha=2.0,
        reg_lambda=8.0,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        tree_method="hist",
        eval_metric="rmse",
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=1500
    )

    model.fit(
        train[features],
        train["Target"],
        eval_set=[(test[features], test["Target"])],
        verbose=200
    )

    # 8. EVALUATION
    logger.info("Ewaluacja modelu na zbiorze testowym...")
    test = test.copy()
    test["Pred"] = test["Prognoza_Wm2"] + model.predict(test[features])
    test["Pred"] = test["Pred"].clip(0, 1500)

    rmse_b, nrmse_b = nrmse(test["Actual_Wm2"], test["Prognoza_Wm2"])
    rmse_m, nrmse_m = nrmse(test["Actual_Wm2"], test["Pred"])

    logger.info(f"Baseline nRMSE: {nrmse_b:.2f}%")
    logger.info(f"Model    nRMSE: {nrmse_m:.2f}%")
    logger.info(f"Gain: {(nrmse_m - nrmse_b):+.2f} pp")

    # 9. SAVE MODEL + META
    logger.info(f"Zapisywanie modelu do: {Config.MODEL_PATH}")
    model.save_model(str(Config.MODEL_PATH))

    meta = {
        "features": features,
        "punkt_categories": list(df["punkt"].cat.categories),
        "lags": lags,
        "metrics": {
            "baseline_nrmse": float(nrmse_b),
            "model_nrmse": float(nrmse_m)
        }
    }

    with open(Config.META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info("=== MODEL PRO 2026 - READY FOR PRODUCTION ===")

if __name__ == "__main__":
    main()

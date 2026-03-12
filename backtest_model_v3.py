import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pyodbc
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_backtest.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# KONFIGURACJA
# ---------------------------------------------------------
class Config:
    BASE_DIR = Path.cwd()

    # ===== ZMIEŃ TĆ ŚCIŹKĘ NA SWOJĄ =====
    SQL_DIR = Path(r"C:\Users\10200871\Desktop\PGEEO\PV1\SQL")
    # =============================================
    SQL_PATH_PROGNOZA = SQL_DIR / "prognozapogody.sql"  # Backtest używa historycznej bazy prognoz
    SQL_PATH_WYKONANIE = SQL_DIR / "wykonanie.sql"

    OUTPUT_DIR = BASE_DIR / "BACKTEST_RESULTS"
    MODEL_INPUT_DIR = Path(r"C:\Users\10200871\Desktop\PGEEO\PV1")
    MODEL_PATH = MODEL_INPUT_DIR / "model_korekty_slonca_v3.json"
    META_PATH = MODEL_INPUT_DIR / "model_meta_v3.json"

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
    if not Path(path).exists():
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

def add_features_v3(df, meta):
    logger.info("Generowanie cech dla backtestu...")
    
    hour = df["dataGodzinaCET"].dt.hour
    doy = df["dataGodzinaCET"].dt.dayofyear
    is_leap = df["dataGodzinaCET"].dt.is_leap_year
    year_len = np.where(is_leap, 366, 365)
    
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_sin"] = np.sin(2 * np.pi * doy / year_len)
    df["day_cos"] = np.cos(2 * np.pi * doy / year_len)
    
    df["temperatura"] = pd.to_numeric(df["temperatura"], errors="coerce").fillna(0.0)
    df["temperatura_24"] = df.groupby("punkt")["temperatura"].shift(24).fillna(0.0)
    
    # Backtest ma dostęp do Actual_Wm2 (wykonania historycznego)
    df["Actual_Wm2"] = pd.to_numeric(df["Actual_Wm2"], errors="coerce")
    df["Prognoza_Wm2"] = pd.to_numeric(df["Prognoza_Wm2"], errors="coerce")
    
    df["Error"] = df["Actual_Wm2"] - df["Prognoza_Wm2"]
    df["Target"] = df["Error"]
    
    # Lagi
    lags = [1,2,3,6,12,24,48]
    for lag in lags:
        df[f"lag_target_{lag}h"] = df.groupby("punkt")["Target"].shift(lag)
        df[f"lag_error_{lag}h"] = df.groupby("punkt")["Error"].shift(lag)

    # Rolling Stats
    rolling_windows = meta.get("windows", [3, 6, 12])
    for w in rolling_windows:
        c_mean = f"rolling_mean_err_{w}h"
        c_std = f"rolling_std_err_{w}h"
        df[c_mean] = df.groupby("punkt")["Error"].transform(lambda x: x.shift(1).rolling(w).mean())
        df[c_std] = df.groupby("punkt")["Error"].transform(lambda x: x.shift(1).rolling(w).std())
        
    return df

# ---------------------------------------------------------
# MAIN BACKTEST
# ---------------------------------------------------------

def main():
    logger.info("=== START BACKTEST PV MODEL PRO 2026 ===")
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 0. LOAD META
    if not Config.META_PATH.exists():
        logger.error(f"Nie znaleziono pliku META: {Config.META_PATH}. Uruchom najpierw trening V3!")
        return
    with open(Config.META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    features = meta["features"]

    # 1. LOAD DATA (Używa prognozapogody.sql z szerokim oknem)
    try:
        df_prog = load_sql(Config.SQL_PATH_PROGNOZA, Config.CONN_STR_PROGNOZA)
        df_wyk = load_sql(Config.SQL_PATH_WYKONANIE, Config.CONN_STR_WYKONANIE)
    except Exception as e:
        logger.error(f"Błąd ładowania danych: {e}")
        return

    # 2. PREPROCESSING
    logger.info("Przetwarzanie danych...")
    df_prog["ts"] = ensure_tz(df_prog["dataGodzinaCET"])
    df_prog = df_prog[df_prog["ts"].dt.minute == 0]
    df_prog["dataGodzinaCET"] = floor_to_hour_warsaw(df_prog["ts"])
    
    # Obsługa nazwy kolumny – normalizujemy do lowercase dla spójności
    df_prog.columns = [c.lower() for c in df_prog.columns]
    col_rad = "calkowitepromieniowanieslonecznenettogodzinowe"
    if col_rad not in df_prog.columns:
        raise KeyError(f"Nie znaleziono kolumny promieniowania. Dostępne: {list(df_prog.columns)}")
    df_prog["Prognoza_Wm2"] = pd.to_numeric(df_prog[col_rad], errors="coerce").fillna(0.0) / 3600
    df_prog["temperatura"] = pd.to_numeric(df_prog["temperatura"], errors="coerce").fillna(0.0)
    df_prog["punkt"] = df_prog["punkt"].astype("string")
    df_prog = df_prog[["punkt", "dataGodzinaCET", "Prognoza_Wm2", "temperatura"]]

    df_wyk["ts"] = ensure_tz(df_wyk["Data"].astype(str) + " " + df_wyk["Czas"].astype(str))
    df_wyk = df_wyk[df_wyk["ts"].dt.minute == 0]
    df_wyk["dataGodzinaCET"] = floor_to_hour_warsaw(df_wyk["ts"])
    
    # Filtr jakości wykonania (taki sam jak w predict) 
    df_wyk["NaslonecznienieHistoria"] = pd.to_numeric(df_wyk["NaslonecznienieHistoria"], errors="coerce")
    df_wyk = df_wyk[(df_wyk["NaslonecznienieHistoria"] >= 0) & (df_wyk["NaslonecznienieHistoria"] < 2500)]
    
    df_wyk_h = df_wyk.groupby(["NazwaFarmy", "dataGodzinaCET"])["NaslonecznienieHistoria"].mean().reset_index()
    df_wyk_h = df_wyk_h.rename(columns={"NazwaFarmy": "punkt", "NaslonecznienieHistoria": "Actual_Wm2"})
    df_wyk_h["punkt"] = df_wyk_h["punkt"].astype("string")

    # 3. MERGE + FEATURES
    df = df_prog.merge(df_wyk_h, on=["punkt", "dataGodzinaCET"], how="inner").sort_values(["punkt", "dataGodzinaCET"])
    df = add_features_v3(df, meta)
    
    # Filtrowanie nocy (aby metryki były realne dla dnia)
    df = df[df["Prognoza_Wm2"] > 0]
    df = df.dropna(subset=features)

    # 4. PREDICTION
    logger.info("Uruchamianie symulacji modelu na danych historycznych...")
    model = xgb.XGBRegressor()
    model.load_model(str(Config.MODEL_PATH))
    
    df["punkt"] = df["punkt"].astype("category").cat.set_categories(meta["punkt_categories"])
    df["Korekta_ML"] = model.predict(df[features])
    df["Prognoza_Finalna_ML"] = (df["Prognoza_Wm2"] + df["Korekta_ML"]).clip(lower=0)

    # 5. METRYKI
    rmse_b, nrmse_b = nrmse(df["Actual_Wm2"], df["Prognoza_Wm2"])
    rmse_m, nrmse_m = nrmse(df["Actual_Wm2"], df["Prognoza_Finalna_ML"])
    
    logger.info(f"BACKTEST RESULTS:")
    logger.info(f"Baseline nRMSE: {nrmse_b:.2f}%")
    logger.info(f"Model V3 nRMSE: {nrmse_m:.2f}%")
    logger.info(f"Historical Gain: {nrmse_b - nrmse_m:+.2f} pp")

    # 6. ZAPIS RAPORTU
    report_path = Config.OUTPUT_DIR / "backtest_detailed_report.csv"
    cols_to_save = ["punkt", "dataGodzinaCET", "Prognoza_Wm2", "Actual_Wm2", "Prognoza_Finalna_ML", "Korekta_ML"]
    df[cols_to_save].to_csv(report_path, sep=";", decimal=",", index=False)
    
    # Zapis statystyk per punkt
    stats_per_punkt = []
    for pt, group in df.groupby("punkt"):
        rb, nb = nrmse(group["Actual_Wm2"], group["Prognoza_Wm2"])
        rm, nm = nrmse(group["Actual_Wm2"], group["Prognoza_Finalna_ML"])
        stats_per_punkt.append({
            "punkt": pt,
            "baseline_nrmse": nb,
            "model_nrmse": nm,
            "gain": nb - nm
        })
    
    stats_df = pd.DataFrame(stats_per_punkt)
    stats_path = Config.OUTPUT_DIR / "backtest_summary_per_punkt.csv"
    stats_df.to_csv(stats_path, sep=";", decimal=",", index=False)

    logger.info(f"Raporty zapisane w: {Config.OUTPUT_DIR}")
    logger.info("=== BACKTEST COMPLETED SUCCESS ===")

if __name__ == "__main__":
    main()

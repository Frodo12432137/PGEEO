import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pyodbc
import xgboost as xgb

# ---------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_prediction_v3.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# KONFIGURACJA (PROD-READY)
# ---------------------------------------------------------
class Config:
    BASE_DIR = Path.cwd()

    # ===== ZMIEŃ TĆ ŚCIŹKĘ NA SWOJĄ =====
    SQL_DIR = Path(r"C:\Users\10200871\Desktop\PGEEO\PV1\SQL")
    # =============================================
    SQL_PATH_PROGNOZA = SQL_DIR / "pogodajankins.sql"  # PRODUKCJA: używa najnowszej prognozy
    SQL_PATH_WYKONANIE = SQL_DIR / "wykonanie.sql"

    # Ścieżki do modelu i metadanych z wersji V3
    INPUT_DIR = Path(r"C:\Users\10200871\Desktop\PGEEO\PV1")
    MODEL_PATH = INPUT_DIR / "model_korekty_slonca_v3.json"
    META_PATH = INPUT_DIR / "model_meta_v3.json"

    # Katalog wyjściowy
    OUTPUT_DIR = Path(r"C:\Users\10200871\Desktop\PGEEO\PV1\OUT")
    
    # Połączenia (oryginalny serwer z Twojego skryptu)
    CONN_STR_BASE = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "Server=MISDWHPRD.GKPGE.PL;"
        "Trusted_Connection=yes;"
    )
    
    MIN_HISTORY_HOURS = 90 * 24

# ---------------------------------------------------------
# FUNKCJE POMOCNICZE
# ---------------------------------------------------------

def load_sql(path, db):
    logger.info(f"Ładowanie SQL z: {path} (DB: {db})")
    if not path.exists():
        raise FileNotFoundError(f"Brak pliku SQL: {path}")
    with open(path, "r", encoding="utf-8") as f:
        q = f.read()
    conn_str = Config.CONN_STR_BASE + f"DATABASE={db};"
    with pyodbc.connect(conn_str) as c:
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

# ---------------------------------------------------------
# FEATURE ENGINEERING (DOPASOWANY DO V3)
# ---------------------------------------------------------

def add_features_v3(df, meta):
    logger.info("Generowanie cech V3 (w tym rolling stats)...")
    
    # Cykliczność
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
    
    # W predykcji Actual może być NaN dla przyszłych godzin
    df["Actual_Wm2"] = pd.to_numeric(df["Actual_Wm2"], errors="coerce")
    df["Prognoza_Wm2"] = pd.to_numeric(df["Prognoza_Wm2"], errors="coerce")
    
    # Target / Error do wyliczania lagów (jeśli mamy dane historyczne w ramce)
    df["Error"] = df["Actual_Wm2"] - df["Prognoza_Wm2"]
    df["Target"] = df["Error"] # W V3 Target i Error są tożsame
    
    # Przygotowanie lagów i rolling stats wg metadanych
    # Ważne: shift(1) i rolling muszą być na poziomie pojedynczego punktu!
    
    # 1. Lagi punktowe
    lags = [1,2,3,6,12,24,48] # Laga 4h, 5h itd z Twojego kodu v1 zostały uproszczone w v3 treningu
    for lag in lags:
        df[f"lag_target_{lag}h"] = df.groupby("punkt")["Target"].shift(lag).fillna(0.0)
        df[f"lag_error_{lag}h"] = df.groupby("punkt")["Error"].shift(lag).fillna(0.0)

    # 2. Statystyki kroczące (Rolling Windows)
    rolling_windows = meta.get("windows", [3, 6, 12])
    for w in rolling_windows:
        c_mean = f"rolling_mean_err_{w}h"
        c_std = f"rolling_std_err_{w}h"
        
        # shift(1) jest kluczowy - używamy historii do przewidywania teraźniejszości/przyszłości
        df[c_mean] = df.groupby("punkt")["Error"].transform(lambda x: x.shift(1).rolling(w).mean()).fillna(0.0)
        df[c_std] = df.groupby("punkt")["Error"].transform(lambda x: x.shift(1).rolling(w).std()).fillna(0.0)
        
    return df

def filter_all_with_max_exec(df):
    if "execId" not in df.columns or "punkt" not in df.columns:
        return df
    df2 = df.copy()
    df2["execId_num"] = pd.to_numeric(df2["execId"], errors="coerce")
    df2 = df2[df2["execId_num"].notna()].copy()
    if df2.empty: return df.head(0)
    df2["max_exec_per_punkt"] = df2.groupby("punkt")["execId_num"].transform("max")
    return df2[df2["execId_num"] == df2["max_exec_per_punkt"]].drop(columns=["execId_num", "max_exec_per_punkt"])

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    logger.info("=== START PREDICTION IRRADIANCE - PRO 2026 V3 ===")
    
    # 0. LOAD META
    if not Config.META_PATH.exists():
        logger.error(f"Nie znaleziono pliku META: {Config.META_PATH}. Uruchom najpierw trening V3!")
        return
        
    with open(Config.META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    features = meta["features"]
    trained_pts = set(meta["punkt_categories"])
    
    RUN_AT = pd.Timestamp.now(tz="Europe/Warsaw").floor("s").tz_localize(None)
    run_ts_str = RUN_AT.strftime("%Y%m%d_%H%M%S")
    
    # 1. LOAD DATA
    try:
        df_prog = load_sql(Config.SQL_PATH_PROGNOZA, "PGESA_MarketAnalytics")
        df_wyk = load_sql(Config.SQL_PATH_WYKONANIE, "PGEEO_DDS")
    except Exception as e:
        logger.error(f"Błąd krytyczny przy ładowaniu danych: {e}")
        return

    # 2. PREPROCESSING
    logger.info("Przetwarzanie danych wejściowych...")
    df_prog["ts"] = ensure_tz(df_prog["dataGodzinaCET"])
    df_prog = df_prog[df_prog["ts"].dt.minute == 0]
    df_prog["dataGodzinaCET"] = floor_to_hour_warsaw(df_prog["ts"])
    
    # Obsługa kolumny rad-netto – normalizujemy do lowercase dla bezpieczeństwa
    df_prog.columns = [c.lower() for c in df_prog.columns]
    col_rad = "calkowitepromieniowanieslonecznenettogodzinowe"
    if col_rad not in df_prog.columns:
        raise KeyError(f"Nie znaleziono kolumny promieniowania w danych prognozy. Dostępne: {list(df_prog.columns)}")
    df_prog["Prognoza_Wm2"] = pd.to_numeric(df_prog[col_rad], errors="coerce").fillna(0.0) / 3600
    df_prog["temperatura"] = pd.to_numeric(df_prog["temperatura"], errors="coerce").fillna(0.0)
    df_prog["punkt"] = df_prog["punkt"].astype("string")
    
    if "execid" in df_prog.columns: df_prog = df_prog.rename(columns={"execid": "execId"})
    if "execId" not in df_prog.columns: df_prog["execId"] = "0"
    df_prog["execId"] = df_prog["execId"].astype("string")

    # Wykonanie
    df_wyk["ts"] = ensure_tz(df_wyk["Data"].astype(str) + " " + df_wyk["Czas"].astype(str))
    df_wyk = df_wyk[df_wyk["ts"].dt.minute == 0]
    df_wyk["dataGodzinaCET"] = floor_to_hour_warsaw(df_wyk["ts"])
    
    # Filtr jakości wykonania
    df_wyk["NaslonecznienieHistoria"] = pd.to_numeric(df_wyk["NaslonecznienieHistoria"], errors="coerce")
    df_wyk = df_wyk[(df_wyk["NaslonecznienieHistoria"] >= 0) & (df_wyk["NaslonecznienieHistoria"] < 2500)]
    
    df_wyk_h = df_wyk.groupby(["NazwaFarmy", "dataGodzinaCET"])["NaslonecznienieHistoria"].mean().reset_index()
    df_wyk_h = df_wyk_h.rename(columns={"NazwaFarmy": "punkt", "NaslonecznienieHistoria": "Actual_Wm2"})
    df_wyk_h["punkt"] = df_wyk_h["punkt"].astype("string")

    # 3. MERGE + FEATURES
    df_hour = df_prog.merge(df_wyk_h, on=["punkt", "dataGodzinaCET"], how="left").sort_values(["punkt", "dataGodzinaCET"])
    df_hour = add_features_v3(df_hour, meta)

    # 4. DECISION ML
    hist_hours = df_wyk_h.groupby("punkt")["dataGodzinaCET"].count()
    eligible_points = set(hist_hours[hist_hours >= Config.MIN_HISTORY_HOURS].index)
    
    df_hour["use_ml"] = df_hour["punkt"].isin(eligible_points) & df_hour["punkt"].isin(trained_pts)
    df_hour["why_no_ml"] = "OK"
    df_hour.loc[~df_hour["punkt"].isin(eligible_points), "why_no_ml"] = "no_3m_history"
    df_hour.loc[~df_hour["punkt"].isin(trained_pts), "why_no_ml"] = "unknown_to_model"

    # 5. PREDICTION
    df_hour["Korekta_ML_hour"] = 0.0
    
    df_ml = df_hour[df_hour["use_ml"]].copy()
    if not df_ml.empty:
        logger.info(f"Uruchamianie modelu dla {len(df_ml)} rekordów...")
        df_ml["punkt"] = df_ml["punkt"].astype("category").cat.set_categories(meta["punkt_categories"])
        
        # XGBoost Prediction
        model = xgb.XGBRegressor()
        model.load_model(str(Config.MODEL_PATH))
        
        # Upewnienie się, że mamy wszystkie kolumny (nawet jeśli są puste)
        for c in features:
            if c not in df_ml.columns: df_ml[c] = 0.0
            
        df_ml["Korekta_ML_hour"] = model.predict(df_ml[features])
        
        # Złącz z powrotem
        df_hour.loc[df_ml.index, "Korekta_ML_hour"] = df_ml["Korekta_ML_hour"]

    df_hour["Final_Wm2_hour"] = (df_hour["Prognoza_Wm2"] + df_hour["Korekta_ML_hour"]).clip(lower=0)
    df_hour["modelRunAt"] = RUN_AT

    # 6. OUTPUT GODZINOWY
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"irradiancja_hour_PROD_{run_ts_str}.csv"
    path_hour = Config.OUTPUT_DIR / fname
    df_hour.to_csv(path_hour, sep=";", decimal=",", index=False, float_format="%.3f")
    logger.info(f"Zapisano wynik godzinowy: {path_hour}")

    # 7. 15-MIN INTERPOLATION
    logger.info("Generowanie interpolacji 15-min...")
    
    def upsample_to_15min_v3(g):
        g = g.sort_values("dataGodzinaCET")
        start, end = g["dataGodzinaCET"].min(), g["dataGodzinaCET"].max()
        idx_15 = pd.date_range(start=start, end=end, freq="15T")
        
        g = g.set_index("dataGodzinaCET").reindex(idx_15)
        g["punkt"] = g["punkt"].ffill().bfill().astype("string")
        g["execId"] = g["execId"].ffill().bfill()
        
        # Interpolacja liniowa korekty
        g["Korekta_ML_hour_15min"] = pd.to_numeric(g["Korekta_ML_hour"], errors="coerce").interpolate(method="time", limit_direction="both")
        g["Prognoza_Wm2_15min"] = pd.to_numeric(g["Prognoza_Wm2"], errors="coerce").ffill()
        
        g["Final_Wm2_15min"] = (g["Prognoza_Wm2_15min"] + g["Korekta_ML_hour_15min"]).clip(lower=0)
        
        g = g.reset_index().rename(columns={"index": "dataCET_15min"})
        g["dataCET_15min_naive"] = g["dataCET_15min"].dt.tz_localize(None)
        g["dataUTC_naive"] = g["dataCET_15min"].dt.tz_localize("Europe/Warsaw").dt.tz_convert("UTC").dt.tz_localize(None)
        
        return g

    # Ustawiamy strefę przed grupowaniem dla poprawności daty
    df_hour["dataGodzinaCET"] = df_hour["dataGodzinaCET"].dt.tz_convert("Europe/Warsaw")
    df_out_15min = df_hour.groupby("punkt", group_keys=False).apply(upsample_to_15min_v3).reset_index(drop=True)
    df_out_15min["data_wykonania"] = RUN_AT

    # Zmiana nazw na format PGE
    rename_map = {
        "dataCET_15min_naive": "dataGodzinaCET",
        "dataUTC_naive": "dataGodzinaUTC",
        "Final_Wm2_15min": "Prognoza_Finalna_ML",
    }
    
    path_15 = Config.OUTPUT_DIR / f"irradiancja_15min_PROD_{run_ts_str}.csv"
    df_out_15min.rename(columns=rename_map).to_csv(path_15, sep=";", decimal=",", index=False, float_format="%.3f")
    logger.info(f"Zapisano wynik 15-min: {path_15}")

    # LATEST EXEC PREZENTACJA
    df_latest = filter_all_with_max_exec(df_out_15min)
    path_latest = Config.OUTPUT_DIR / f"irradiancja_15min_latest_execId_{run_ts_str}.csv"
    df_latest.rename(columns=rename_map).to_csv(path_latest, sep=";", decimal=",", index=False, float_format="%.3f")
    
    logger.info("=== PRODUKCJA V3 ZAKOŃCZONA SUCCESS ===")

if __name__ == "__main__":
    main()

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import pyodbc
from sklearn.metrics import mean_squared_error

# ==============================================================================
# KONFIG
# ==============================================================================

SQL_PATH_PROGNOZA = r"C:\Users\10200871\Desktop\PGEEO\PV1\SQL\prognozapogody.sql"
SQL_PATH_WYKONANIE = r"C:\Users\10200871\Desktop\PGEEO\PV1\SQL\wykonanie.sql"

MODEL_PATH = r"C:\Users\10200871\Desktop\PGEEO\PV1\model_korekty_slonca.json"
META_PATH  = r"C:\Users\10200871\Desktop\PGEEO\PV1\model_meta.json"

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

# ==============================================================================
# FUNKCJE
# ==============================================================================

def load_sql(path, conn):
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

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n=== TRAINING MODEL PRO 2026 V3 - IRRADIANCE CORRECTION ===\n")

    # 1. LOAD DATA
    df_prog = load_sql(SQL_PATH_PROGNOZA, CONN_STR_PROGNOZA)
    df_wyk  = load_sql(SQL_PATH_WYKONANIE, CONN_STR_WYKONANIE)

    # Normalize column names to lowercase
    df_prog.columns = [c.lower() for c in df_prog.columns]
    df_wyk.columns = [c.lower() for c in df_wyk.columns]

    # 2. PROGNOZA
    df_prog["ts"] = ensure_tz(df_prog["datagodzinacet"])
    df_prog = df_prog[df_prog["ts"].dt.minute == 0]
    df_prog["dataGodzinaCET"] = floor_to_hour_warsaw(df_prog["ts"])

    df_prog["Prognoza_Wm2"] = (
        df_prog["calkowitepromieniowanieslonecznenettogodzinowe"] / 3600
    )
    df_prog["temperatura"] = pd.to_numeric(df_prog["temperatura"], errors="coerce").fillna(0)
    df_prog["punkt"] = df_prog["punkt"].astype("string")
    df_prog = df_prog[["punkt", "dataGodzinaCET", "Prognoza_Wm2", "temperatura"]]

    # 3. WYKONANIE
    df_wyk["data"] = df_wyk["data"].astype(str)
    df_wyk["czas"] = df_wyk["czas"].astype(str)
    df_wyk["ts"] = ensure_tz(df_wyk["data"] + " " + df_wyk["czas"])
    df_wyk = df_wyk[df_wyk["ts"].dt.minute == 0]
    df_wyk["dataGodzinaCET"] = floor_to_hour_warsaw(df_wyk["ts"])

    df_wyk_hour = (
        df_wyk.groupby(["nazwafarmy", "dataGodzinaCET"])["naslonecznieniehistoria"]
        .mean().reset_index()
        .rename(columns={"nazwafarmy": "punkt", "naslonecznieniehistoria": "Actual_Wm2"})
    )
    df_wyk_hour["punkt"] = df_wyk_hour["punkt"].astype("string")

    # 3B. 3 MONTH HISTORY FILTER
    hist_hours = df_wyk_hour.groupby("punkt")["dataGodzinaCET"].count()
    eligible_points = set(hist_hours[hist_hours >= 90*24].index)
    df_prog     = df_prog[df_prog["punkt"].isin(eligible_points)]
    df_wyk_hour = df_wyk_hour[df_wyk_hour["punkt"].isin(eligible_points)]

    # 4. MERGE
    df = df_prog.merge(df_wyk_hour, on=["punkt", "dataGodzinaCET"], how="inner")
    df = df.sort_values(["punkt", "dataGodzinaCET"])
    df = df.dropna(subset=["Actual_Wm2"])

    # --- NOWOŚĆ V3: Filtrowanie nocy ---
    df = df[df["Prognoza_Wm2"] > 0]

    df["Target"] = df["Actual_Wm2"] - df["Prognoza_Wm2"]
    df["Error"]  = df["Actual_Wm2"] - df["Prognoza_Wm2"]

    # 5. FEATURES
    doy = df["dataGodzinaCET"].dt.dayofyear
    is_leap = df["dataGodzinaCET"].dt.is_leap_year
    year_len = np.where(is_leap, 366, 365)

    df["hour_sin"] = np.sin(2*np.pi * df["dataGodzinaCET"].dt.hour / 24)
    df["hour_cos"] = np.cos(2*np.pi * df["dataGodzinaCET"].dt.hour / 24)
    df["day_sin"]  = np.sin(2*np.pi * doy / year_len)
    df["day_cos"]  = np.cos(2*np.pi * doy / year_len)
    df["temperatura_24"] = df.groupby("punkt")["temperatura"].shift(24)

    # Lagi targetu
    lags = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,24,30,36,48]
    for lag in lags:
        df[f"lag_{lag}h"] = df.groupby("punkt")["Target"].shift(lag)
    lag_cols = [f"lag_{l}h" for l in lags]

    # Lagi błędu
    error_lags = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,24,30,36,48]
    for lag in error_lags:
        df[f"error_lag_{lag}h"] = df.groupby("punkt")["Error"].shift(lag)
    error_lag_cols = [f"error_lag_{l}h" for l in error_lags]

    # Lagi prognozy
    forecast_lags = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,24,30,36,48]
    for lag in forecast_lags:
        df[f"forecast_lag_{lag}h"] = df.groupby("punkt")["Prognoza_Wm2"].shift(lag)
    forecast_lag_cols = [f"forecast_lag_{l}h" for l in forecast_lags]

    # --- NOWOŚĆ V3: Statystyki Kroczące ---
    rolling_windows = [3, 6, 12]
    rolling_cols = []
    for w in rolling_windows:
        c_mean = f"rolling_mean_err_{w}h"
        c_std  = f"rolling_std_err_{w}h"
        df[c_mean] = df.groupby("punkt")["Error"].transform(lambda x: x.shift(1).rolling(w).mean())
        df[c_std]  = df.groupby("punkt")["Error"].transform(lambda x: x.shift(1).rolling(w).std())
        rolling_cols.extend([c_mean, c_std])

    df = df.dropna(subset=lag_cols + error_lag_cols + forecast_lag_cols + rolling_cols + ["temperatura_24"])
    df = df.groupby("punkt").apply(lambda g: g.iloc[48:]).reset_index(drop=True)
    df["punkt"] = df["punkt"].astype("category")

    print(f"\nFinal training records: {len(df)}")

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
        + rolling_cols
    )

    # 7. MODEL
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
        train[features], train["Target"],
        eval_set=[(test[features], test["Target"])],
        verbose=200
    )

    # 8. EVALUATION
    test = test.copy()
    test["Pred"] = test["Prognoza_Wm2"] + model.predict(test[features])
    test["Pred"] = test["Pred"].clip(0, 1500)

    rmse_b, nrmse_b = nrmse(test["Actual_Wm2"], test["Prognoza_Wm2"])
    rmse_m, nrmse_m = nrmse(test["Actual_Wm2"], test["Pred"])

    print(f"\nBaseline nRMSE: {nrmse_b:.2f}%")
    print(f"Model V3 nRMSE: {nrmse_m:.2f}%")
    print(f"Gain: {(nrmse_m - nrmse_b):+.2f} pp")

    # 9. SAVE MODEL + META
    model.save_model(MODEL_PATH)

    meta = {
        "features":          features,
        "punkt_categories":  list(df["punkt"].cat.categories),
        "lags":              lags,
        "error_lags":        error_lags,
        "forecast_lags":     forecast_lags,
        "windows":           rolling_windows,
        "metrics": {
            "baseline_nrmse": float(nrmse_b),
            "model_nrmse":    float(nrmse_m)
        }
    }

    Path(os.path.dirname(META_PATH)).mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n=== MODEL PRO 2026 V3 - READY FOR PRODUCTION ===\n")


if __name__ == "__main__":
    main()

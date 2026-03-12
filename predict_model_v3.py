import os
import json
import numpy as np
import pandas as pd
import pyodbc
import xgboost as xgb
from pathlib import Path

# ==============================================================================
# KONFIG
# ==============================================================================

# ===== ZMIEŃ TĘ ŚCIEŻKĘ NA SWOJĄ =====
SQL_DIR = Path(r"C:\Users\10200871\Desktop\PGEEO\PV1\SQL")
# =============================================

SQL_PATH_PROGNOZA  = SQL_DIR / "pogodajankins.sql"  # PRODUKCJA: używa najnowszej prognozy
SQL_PATH_WYKONANIE = SQL_DIR / "wykonanie.sql"

MODEL_PATH = r"C:\Users\10200871\Desktop\PGEEO\PV1\model_korekty_slonca.json"
META_PATH  = r"C:\Users\10200871\Desktop\PGEEO\PV1\model_meta.json"

OUTPUT_DIR = r"C:\Users\10200871\Desktop\PGEEO\PV1\OUT"

CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "Server=MISDWPPRD.GKPGE.PL;"
    "Trusted_Connection=yes;"
)

MIN_HISTORY_HOURS = 90 * 24

# ==============================================================================
# FUNKCJE PODSTAWOWE
# ==============================================================================

def load_sql(path, db):
    with open(path, "r", encoding="utf-8") as f:
        q = f.read()
    with pyodbc.connect(CONN_STR + f"DATABASE={db};") as c:
        return pd.read_sql(q, c)

def ensure_tz(ts):
    ts = pd.to_datetime(ts, errors="coerce")
    try:
        if ts.dt.tz is not None:
            return ts.dt.tz_convert("Europe/Warsaw")
    except Exception:
        pass
    return ts.dt.tz_localize("UTC").dt.tz_convert("Europe/Warsaw")

def floor_to_hour_warsaw(ts):
    ts = ts.dt.tz_convert("UTC")
    ts = ts.dt.floor("h")
    return ts.dt.tz_convert("Europe/Warsaw")

# ==============================================================================
# CECHY (V3 – z Rolling Stats)
# ==============================================================================

def add_features(df, lags, error_lags, forecast_lags, rolling_windows=None):
    if rolling_windows is None:
        rolling_windows = [3, 6, 12]

    hour = df["dataGodzinaCET"].dt.hour
    doy = df["dataGodzinaCET"].dt.dayofyear
    is_leap = df["dataGodzinaCET"].dt.is_leap_year
    year_len = np.where(is_leap, 366, 365)

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_sin"]   = np.sin(2 * np.pi * doy / year_len)
    df["day_cos"]   = np.cos(2 * np.pi * doy / year_len)

    df["temperatura"] = pd.to_numeric(df["temperatura"], errors="coerce").fillna(0.0)
    df["temperatura_24"] = df.groupby("punkt")["temperatura"].shift(24).fillna(0.0)

    df["Target"] = df["Actual_Wm2"] - df["Prognoza_Wm2"]
    df["Error"]  = df["Actual_Wm2"] - df["Prognoza_Wm2"]

    for lag in lags:
        df[f"lag_{lag}h"] = df.groupby("punkt")["Target"].shift(lag).fillna(0.0)

    for lag in error_lags:
        df[f"error_lag_{lag}h"] = df.groupby("punkt")["Error"].shift(lag).fillna(0.0)

    for lag in forecast_lags:
        df[f"forecast_lag_{lag}h"] = df.groupby("punkt")["Prognoza_Wm2"].shift(lag).fillna(0.0)

    # --- NOWOŚĆ V3: Statystyki Kroczące ---
    for w in rolling_windows:
        df[f"rolling_mean_err_{w}h"] = df.groupby("punkt")["Error"].transform(
            lambda x: x.shift(1).rolling(w).mean()
        ).fillna(0.0)
        df[f"rolling_std_err_{w}h"] = df.groupby("punkt")["Error"].transform(
            lambda x: x.shift(1).rolling(w).std()
        ).fillna(0.0)

    return df


def filter_all_with_max_exec(df: pd.DataFrame) -> pd.DataFrame:
    if "execId" not in df.columns or "punkt" not in df.columns:
        return df

    df2 = df.copy()
    df2["punkt"] = df2["punkt"].astype("string")

    exec_str = df2["execId"].astype("string")
    mask_nonempty = exec_str.notna() & (exec_str != "")
    df2 = df2[mask_nonempty].copy()
    if df2.empty:
        return df.head(0)

    df2["execId_num"] = pd.to_numeric(df2["execId"], errors="coerce")
    df2 = df2[df2["execId_num"].notna()].copy()
    if df2.empty:
        return df.head(0)

    df2["max_exec_per_punkt"] = df2.groupby("punkt")["execId_num"].transform("max")
    out = df2[df2["execId_num"] == df2["max_exec_per_punkt"]].drop(
        columns=["execId_num", "max_exec_per_punkt"]
    )
    return out


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n=== PRODUKCJA IRRADIACJI - WERSJA PRO 2026 V3 ===\n")

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    features       = meta["features"]
    trained_pts    = set(meta["punkt_categories"])
    lags           = meta.get("lags", [])
    error_lags     = meta.get("error_lags", [])
    forecast_lags  = meta.get("forecast_lags", [])
    rolling_windows = meta.get("windows", [3, 6, 12])

    RUN_AT = (
        pd.Timestamp.now(tz="Europe/Warsaw")
        .floor("s")
        .tz_localize(None)
    )
    run_ts_str = RUN_AT.strftime("%Y%m%d_%H%M%S")

    # --------------------------------------------------------------------------
    # PROGNOZA
    # --------------------------------------------------------------------------
    df_prog = load_sql(SQL_PATH_PROGNOZA, "PGESA_MarketAnalytics")

    # Normalizacja nazw kolumn do lowercase dla spójności i uniknięcia KeyErrors
    df_prog.columns = [c.lower() for c in df_prog.columns]
    
    df_prog["ts"] = ensure_tz(df_prog["datagodzinacet"])
    df_prog = df_prog[df_prog["ts"].dt.minute == 0]
    df_prog["datagodzinacet"] = floor_to_hour_warsaw(df_prog["ts"])

    col_rad = "calkowitepromieniowanieslonecznenettogodzinowe"
    df_prog["Prognoza_Wm2"] = (
        pd.to_numeric(df_prog[col_rad], errors="coerce").fillna(0.0) / 3600
    )
    df_prog["temperatura"] = pd.to_numeric(df_prog["temperatura"], errors="coerce").fillna(0.0)
    df_prog["punkt"] = df_prog["punkt"].astype("string")

    if "execid" not in df_prog.columns:
        df_prog["execid"] = pd.Series([None] * len(df_prog), dtype="string")
    df_prog["execid"] = df_prog["execid"].astype("string")

    df_prog = df_prog[["punkt", "datagodzinacet", "Prognoza_Wm2", "temperatura", "execid"]]
    df_prog = df_prog.rename(columns={"datagodzinacet": "dataGodzinaCET", "execid": "execId"})

    # WYKONANIE
    df_wyk = load_sql(SQL_PATH_WYKONANIE, "PGEEO_DDS")
    df_wyk.columns = [c.lower() for c in df_wyk.columns]

    df_wyk["data"] = df_wyk["data"].astype(str)
    df_wyk["czas"] = df_wyk["czas"].astype(str)
    df_wyk["ts"] = ensure_tz(df_wyk["data"] + " " + df_wyk["czas"])

    df_wyk = df_wyk[df_wyk["ts"].dt.minute == 0]
    df_wyk["dataGodzinaCET"] = floor_to_hour_warsaw(df_wyk["ts"])

    # Filtr jakości
    df_wyk = df_wyk[
        (pd.to_numeric(df_wyk["naslonecznieniehistoria"], errors="coerce") >= 0) &
        (pd.to_numeric(df_wyk["naslonecznieniehistoria"], errors="coerce") < 2500)
    ]

    df_wyk_h = (
        df_wyk.groupby(["nazwafarmy", "dataGodzinaCET"])["naslonecznieniehistoria"]
        .mean().reset_index()
        .rename(columns={"nazwafarmy": "punkt", "naslonecznieniehistoria": "Actual_Wm2"})
    )
    df_wyk_h["punkt"] = df_wyk_h["punkt"].astype("string")

    hist_hours = df_wyk_h.groupby("punkt")["dataGodzinaCET"].count()
    eligible_points = set(hist_hours[hist_hours >= MIN_HISTORY_HOURS].index)

    # --------------------------------------------------------------------------
    # MERGE + CECHY
    # --------------------------------------------------------------------------
    df_hour = (
        df_prog.merge(df_wyk_h, on=["punkt", "dataGodzinaCET"], how="left")
        .sort_values(["punkt", "dataGodzinaCET"])
        .copy()
    )
    df_hour["execId"] = df_hour["execId"].astype("string")
    df_hour = add_features(df_hour, lags, error_lags, forecast_lags, rolling_windows)

    # --------------------------------------------------------------------------
    # DECYZJA ML / FALLBACK
    # --------------------------------------------------------------------------
    df_hour["use_ml"] = df_hour["punkt"].isin(eligible_points) & df_hour["punkt"].isin(trained_pts)

    df_hour["why_no_ml"] = ""
    df_hour.loc[~df_hour["punkt"].isin(eligible_points), "why_no_ml"] += "no_3m_history;"
    df_hour.loc[~df_hour["punkt"].isin(trained_pts),     "why_no_ml"] += "unknown_to_model;"
    df_hour.loc[df_hour["why_no_ml"] == "",              "why_no_ml"] = "OK"

    df_ml    = df_hour[df_hour["use_ml"]].copy()
    df_naive = df_hour[~df_hour["use_ml"]].copy()

    # --------------------------------------------------------------------------
    # PREDYKCJA ML
    # --------------------------------------------------------------------------
    if not df_ml.empty:
        df_ml["punkt"] = df_ml["punkt"].astype("category").cat.set_categories(meta["punkt_categories"])

        missing = [c for c in features if c not in df_ml.columns]
        for col in missing:
            if col != "punkt":
                df_ml[col] = 0.0

        num_features = [c for c in features if c != "punkt"]
        df_ml[num_features] = df_ml[num_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        model = xgb.XGBRegressor(enable_categorical=True, tree_method="hist")
        model.load_model(MODEL_PATH)

        df_ml["Korekta_ML_hour"] = model.predict(df_ml[features])
        df_ml["Final_Wm2_hour"] = (df_ml["Prognoza_Wm2"] + df_ml["Korekta_ML_hour"]).clip(lower=0)

    # --------------------------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------------------------
    if not df_naive.empty:
        df_naive["Korekta_ML_hour"] = 0.0
        df_naive["Final_Wm2_hour"] = df_naive["Prognoza_Wm2"].clip(lower=0)

    # --------------------------------------------------------------------------
    # OUTPUT GODZINOWY
    # --------------------------------------------------------------------------
    df_out = pd.concat([df_ml, df_naive], ignore_index=True)
    df_out = df_out.sort_values(["punkt", "dataGodzinaCET"]).reset_index(drop=True)
    df_out["punkt"] = df_out["punkt"].astype("string")
    df_out["modelRunAt"] = RUN_AT

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = f"irradiancja_hour_PROD_{run_ts_str}.csv"
    path = os.path.join(OUTPUT_DIR, fname)

    cols_hour = [
        "punkt", "dataGodzinaCET", "execId",
        "Prognoza_Wm2", "Korekta_ML_hour", "Final_Wm2_hour",
        "Actual_Wm2", "use_ml", "why_no_ml", "modelRunAt"
    ]
    cols_hour = [c for c in cols_hour if c in df_out.columns]
    df_out[cols_hour].to_csv(path, sep=";", decimal=",", index=False, float_format="%.3f")
    print(f"\n[OK] Wynik godzinowy zapisany: {path}")

    # ==========================================================================
    # 15-MIN INTERPOLACJA
    # ==========================================================================
    df_15 = df_out.sort_values(["punkt", "dataGodzinaCET"]).copy()
    df_15["dataGodzinaCET"] = df_15["dataGodzinaCET"].dt.tz_convert("Europe/Warsaw")

    def upsample_to_15min(g: pd.DataFrame) -> pd.DataFrame:
        tz = g["dataGodzinaCET"].dt.tz
        start = g["dataGodzinaCET"].min()
        end   = g["dataGodzinaCET"].max()
        idx_15 = pd.date_range(start=start, end=end, freq="15T", tz=tz)

        g = g.set_index("dataGodzinaCET").reindex(idx_15)
        g["punkt"] = g["punkt"].ffill().bfill().astype("string")

        if "execId" in g.columns:
            g["execId"] = g["execId"].astype("string").ffill().bfill()

        g["Korekta_ML_hour_15min"] = (
            pd.to_numeric(g["Korekta_ML_hour"], errors="coerce")
            .interpolate(method="time", limit_direction="both")
        )
        g["Prognoza_Wm2_15min"] = pd.to_numeric(g["Prognoza_Wm2"], errors="coerce").ffill()
        g["Final_Wm2_15min"] = (g["Prognoza_Wm2_15min"] + g["Korekta_ML_hour_15min"]).clip(lower=0)

        g["use_ml"]     = g["use_ml"].ffill().bfill()
        g["why_no_ml"]  = g["why_no_ml"].ffill().bfill()

        g = g.reset_index().rename(columns={"index": "dataCET_15min"})
        g["dataCET_15min"] = g["dataCET_15min"].dt.tz_convert("Europe/Warsaw").dt.tz_localize(None)
        g["dataUTC"] = (
            g["dataCET_15min"]
            .dt.tz_localize("Europe/Warsaw")
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )
        return g

    df_out_15min = (
        df_15.groupby("punkt", group_keys=False)
        .apply(upsample_to_15min)
        .sort_values(["punkt", "dataCET_15min"])
        .reset_index(drop=True)
    )
    df_out_15min["modelRunAt"] = RUN_AT

    fname15 = f"irradiancja_15min_PROD_{run_ts_str}.csv"
    path15  = os.path.join(OUTPUT_DIR, fname15)

    final_cols = [
        "dataCET_15min", "dataUTC", "punkt", "execId",
        "Prognoza_Wm2", "use_ml",
        "Korekta_ML_hour", "Final_Wm2_hour",
        "Prognoza_Wm2_15min", "Final_Wm2_15min",
        "Actual_Wm2", "modelRunAt"
    ]
    final_cols = [c for c in final_cols if c in df_out_15min.columns]

    rename_map = {
        "dataCET_15min":  "dataGodzinaCET",
        "dataUTC":        "dataGodzinaUTC",
        "modelRunAt":     "data_wykonania",
        "Final_Wm2_15min": "Prognoza_Finalna_ML",
    }

    df_out_15min[final_cols].rename(columns=rename_map).to_csv(
        path15, sep=";", decimal=",", index=False, float_format="%.3f"
    )
    print("\n[OK] Wynik 15-min zapisany:", path15)

    # Prezentacja – tylko najwyższy execId per punkt
    df_15_latest = filter_all_with_max_exec(df_out_15min)
    cols_prez = [c for c in final_cols if c in df_15_latest.columns]
    path_15_latest = os.path.join(OUTPUT_DIR, f"irradiancja_15min_latest_execId_{run_ts_str}.csv")
    df_15_latest[cols_prez].rename(columns=rename_map).to_csv(
        path_15_latest, sep=";", decimal=",", index=False, float_format="%.3f"
    )
    print("[OK] Prezentacja (najwyższy execId):", path_15_latest)

    print("\n-------------------------------------------")
    print("PRODUKCJA ZAKOŃCZONA - HOUR + 15-MIN GOTOWE")
    print("===========================================\n")


if __name__ == "__main__":
    main()

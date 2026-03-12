# PGE PV SOLAR FORECAST CORRECTION – ML MODEL

System uczenia maszynowego (XGBoost) do korekty prognoz nasłonecznienia dla instalacji PV.

## Architektura

```
┌─────────────────────────────────────────────────┐
│               prognozapogody.sql                │
│    (Historyczne dane – Trening & Backtest)      │
└──────────────┬──────────────────────────────────┘
               │
    ┌──────────▼──────────┐   ┌───────────────────────┐
    │  train_model_v3.py  │   │  backtest_model_v3.py │
    │  (Trening XGBoost)  │   │  (Symulacja historycz.)│
    └──────────┬──────────┘   └───────────────────────┘
               │ model_meta_v3.json
               │ model_korekty_slonca_v3.json
    ┌──────────▼──────────┐
    │  predict_model_v3.py│  ← pogodajankins.sql (PROD)
    │  (Predykcja LIVE)   │
    └─────────────────────┘
```

## Pliki

| Plik | Opis |
|---|---|
| `train_model_v3.py` | Trening modelu (Rolling Stats, Feature Importance) |
| `train_model_v2.py` | Starsza wersja treningu (więcej lagów) |
| `predict_model_v3.py` | Produkcja – predykcja godzinowa + 15-min |
| `backtest_model_v3.py` | Backtest – symulacja na danych historycznych |

## Wymagania

```bash
pip install xgboost pandas numpy pyodbc scikit-learn matplotlib
```

## Użycie

### 1. Trening
```bash
python train_model_v3.py
```
Wynik: `output_v3/model_korekty_slonca_v3.json` + `model_meta_v3.json`

### 2. Predykcja (Produkcja)
```bash
python predict_model_v3.py
```
Wynik: `OUT/irradiancja_15min_PROD_<timestamp>.csv`

### 3. Backtest
```bash
python backtest_model_v3.py
```
Wynik: `BACKTEST_RESULTS/backtest_summary_per_punkt.csv`

## Konfiguracja

Ustaw zakres dat bezpośrednio w plikach SQL:
- `SQL/prognozapogody.sql` – zakres historyczny (trening + backtest)
- `SQL/pogodajankins.sql` – najnowsza prognoza (predykcja)
- `SQL/wykonanie.sql` – dane historyczne z SCADA

## Feature Engineering (V3)

- Funkcje cykliczne (sin/cos godziny i dnia roku)
- Lagi błędu i targetu: 1, 2, 3, 6, 12, 24, 48h
- **Rolling Statistics**: Średnia i odchylenie standardowe błędu z okien 3h, 6h, 12h
- Temperatura aktualna i 24h wstecz
- Filtrowanie godzin nocnych (`Prognoza_Wm2 > 0`)

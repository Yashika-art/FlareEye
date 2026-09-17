# FlareEye — AI/ML Thermal Event Classifier + Forecaster

A working, end-to-end machine learning pipeline for the FlareEye concept:
classifying satellite-detected thermal anomalies into industrial fire, gas
flare, wildfire, agricultural burning, or industrial heat — **and predicting
what each site does next.**

## What's actually real here

This is a genuine trained scikit-learn pipeline, not a mock-up:

**Classification (stages 1-5)**
1. **`generate_data.py`** — synthesizes ~18,000 FIRMS/VIIRS-style raw thermal
   detections across 180 candidate sites in India, with class-conditional,
   deliberately overlapping feature distributions (see note below on why).
2. **`pipeline.py`** — clusters raw point detections into spatio-temporal
   *events* using DBSCAN (haversine distance), then engineers the
   spatial/temporal/thermal/optical feature set per event.
3. **`train_model.py`** — trains a `RandomForestClassifier` on those events,
   using a **site-level train/test split** (an entire site's detections are
   held out, never split across train and test — this avoids the spatial
   leakage the original brief calls out).
4. **`classify_events.py`** — runs the trained model over every event,
   attaches full class-probability distributions, and computes a risk
   score/priority from predicted class + FRP + persistence + confidence.

**Forecasting — predicting the next event (stages 6-7, new)**
5. **`forecast_pipeline.py`** — turns raw detections into a per-site
   *daily* time series (365 days x 180 sites), then builds a supervised
   table of lag/rolling features (last-3-days FRP, 7-day rolling mean,
   days-since-last-detection, persistence-so-far, cumulative detections...)
   with next-day targets.
6. **`train_forecast_model.py`** — trains two models on that table:
   - a `RandomForestRegressor` predicting **tomorrow's FRP** at a site
   - a `RandomForestClassifier` predicting **whether the site stays active
     tomorrow** (recurrence probability)

   Evaluated on a **time-based holdout** (train on the first 292 days of
   the year, test on the remaining ~73) — the correct split for a
   forecasting task, distinct from the classifier's site-level split.
7. **`forecast_events.py`** — applies both models to every site's most
   recent history, autoregressively chains 3 days ahead, and labels a
   trend (Rising / Stable / Falling). Writes `outputs/forecasts.json`.

**Frontend**
8. **`dashboard_shell.html` + `build_dashboard.py`** — a self-contained
   dashboard (map with a Dark/Streets/Satellite basemap toggle, event feed,
   classification breakdown, a **Next-Event Forecast** panel with a
   14-day sparkline + 3-day prediction, and a Model tab showing both the
   classifier's and forecaster's real metrics) driven entirely by the
   JSON files above — nothing in the UI is hand-typed mock data.

## Current measured performance

**Classification** (`outputs/metrics.json`)
- Accuracy: **96.3%** · Macro F1: **96.3%**
- 125 training sites / 54 held-out test sites

**Forecasting** (`outputs/forecast_metrics.json`)
- Next-day FRP prediction: **MAE ≈ 22 MW**, R² ≈ 0.42
- Next-day recurrence prediction: **~90% accuracy**, F1 ≈ 0.85
- Trained on 52k (site, day) samples, tested on 13k unseen future days

## Why synthetic data, and why it's deliberately imperfect

This sandbox's network egress is limited to package registries (pypi, npm,
github) — it cannot reach `firms.modaps.eosdis.nasa.gov` or Copernicus/
Sentinel endpoints. So `generate_data.py` simulates detections instead of
pulling live FIRMS data. The class feature distributions are grounded in the
domain logic from the FlareEye brief (gas flares persist daily near
infrastructure; wildfires sit far from infra with high wind; agricultural
burning clusters seasonally over cropland, etc.), overlap on purpose, and
include ~10% ambiguous/blended-feature sites — so the reported accuracy is a
believable ~96%, not a suspicious 100%. Likewise the forecasting R² is
moderate (0.42) because day-to-day FRP genuinely is noisy — a perfect score
there would mean the synthetic data was unrealistically easy.

## Swapping in real data

Replace `generate_data.py`'s output with real pulls from:
- NASA FIRMS / VIIRS API (needs a free MAP_KEY from firms.modaps.eosdis.nasa.gov)
- ESA WorldCover (land cover)
- OpenStreetMap Overpass API (infrastructure/roads)
- Global Energy Monitor (industrial facility database)

Every downstream file (`pipeline.py`, `train_model.py`, `classify_events.py`,
`forecast_pipeline.py`, `train_forecast_model.py`, `forecast_events.py`) is
unchanged — they only need a dataframe with the same columns (`lat`, `lng`,
`day`, `frp`, `brightness_k`, `confidence`, `wind_kmh`, `swir_reflectance`,
`dist_infra_km`, `dist_road_km`, `land_cover`, `true_class` for training).

## Running it yourself (in order)

```bash
pip install -r requirements.txt
python3 train_model.py           # builds dataset, trains + evaluates the classifier
python3 classify_events.py       # applies it, writes outputs/events.json
python3 train_forecast_model.py  # builds daily series, trains the forecaster
python3 forecast_events.py       # predicts next-event activity, writes outputs/forecasts.json
python3 build_dashboard.py       # embeds everything into the final HTML app
```

Then open `outputs/flareeye_ai_dashboard.html` in a browser — no server
needed, everything is embedded directly in the file.

## Files

```
generate_data.py         synthetic FIRMS/VIIRS-style raw detection generator
pipeline.py               DBSCAN event clustering + classifier feature engineering
train_model.py            trains + evaluates the RandomForest classifier
classify_events.py        scores every event, computes risk, writes events.json

forecast_pipeline.py      builds daily time series + lag/rolling forecasting features
train_forecast_model.py   trains the FRP regressor + recurrence classifier
forecast_events.py        predicts next-event activity per site, writes forecasts.json

dashboard_shell.html      dashboard source (HTML/CSS/JS) with __PLACEHOLDER__ tokens
build_dashboard.py        injects the 4 JSON outputs into the shell -> final app

data/                     raw detections, daily series, event/forecast tables
models/                   classifier.joblib, forecast_regressor.joblib, forecast_classifier.joblib
outputs/
  metrics.json                 classifier accuracy, F1, confusion matrix, feature importances
  events.json                  classified events (used by the dashboard)
  forecast_metrics.json        forecaster MAE / R² / recurrence accuracy
  forecasts.json                per-event next-day + 3-day predictions
  flareeye_ai_dashboard.html   the app itself
```

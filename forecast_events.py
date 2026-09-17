"""
forecast_events.py
---------------------
Uses the trained forecasting models to predict, for every event, what
happens NEXT: tomorrow's expected FRP, the probability the site stays
active, and a 3-day-ahead autoregressive trend. Writes
outputs/forecasts.json, keyed by the same event ids as outputs/events.json
so the dashboard can join classification + prediction per event.
"""

import json
import joblib
import numpy as np
import pandas as pd

from forecast_pipeline import NUMERIC_FORECAST_FEATURES, CATEGORICAL_FORECAST_FEATURES


def featurize_row(row: dict, columns: list) -> pd.DataFrame:
    df = pd.DataFrame([row])
    X = pd.get_dummies(df[NUMERIC_FORECAST_FEATURES + CATEGORICAL_FORECAST_FEATURES],
                        columns=CATEGORICAL_FORECAST_FEATURES)
    for col in columns:
        if col not in X.columns:
            X[col] = 0
    return X[columns]


def latest_feature_row(daily_site: pd.DataFrame, predicted_class: str) -> dict:
    """Recreate the same lag/rolling features train_forecast_model.py
    used, but for the single most recent day of a site -- this is the
    live 'as of today, what happens next' feature vector."""
    g = daily_site.sort_values("day").reset_index(drop=True)
    t = len(g) - 1
    frp_series = g["frp"].to_numpy()
    active_series = g["active"].to_numpy()
    first_active_day = g.loc[g.active == 1, "day"].min()
    first_active_day = 0 if pd.isna(first_active_day) else first_active_day

    window = frp_series[max(0, t - 6): t + 1]
    active_window = active_series[max(0, t - 6): t + 1]
    days_since_last = 0
    for back in range(t, -1, -1):
        if active_series[back] == 1:
            days_since_last = t - back
            break
    else:
        days_since_last = t

    day = int(g.loc[t, "day"])
    return dict(
        frp_today=frp_series[t], frp_lag1=frp_series[t - 1] if t >= 1 else 0.0,
        frp_lag2=frp_series[t - 2] if t >= 2 else 0.0, frp_lag3=frp_series[t - 3] if t >= 3 else 0.0,
        rolling_mean_7=float(np.mean(window)), rolling_max_7=float(np.max(window)),
        rolling_active_rate_7=float(np.mean(active_window)),
        days_since_last_detection=int(days_since_last),
        persistence_so_far=int(day - first_active_day) if day >= first_active_day else 0,
        cumulative_detections=int(np.sum(active_series[:t + 1])),
        dist_infra_km=g.loc[t, "dist_infra_km"], land_cover=g.loc[t, "land_cover"],
        predicted_class=predicted_class,
    ), frp_series[max(0, t - 13):t + 1].tolist()  # last-14-day history for sparklines


def trend_label(frp_recent_mean: float, frp_next: float) -> str:
    if frp_recent_mean <= 1e-6:
        return "Rising" if frp_next > 5 else "Stable"
    ratio = frp_next / frp_recent_mean
    if ratio >= 1.15:
        return "Rising"
    if ratio <= 0.85:
        return "Falling"
    return "Stable"


def main():
    daily = pd.read_csv("data/daily_series.csv")
    events = json.load(open("outputs/events.json"))

    reg_bundle = joblib.load("models/forecast_regressor.joblib")
    clf_bundle = joblib.load("models/forecast_classifier.joblib")
    reg_model, reg_cols = reg_bundle["model"], reg_bundle["columns"]
    clf_model, clf_cols = clf_bundle["model"], clf_bundle["columns"]

    # map event -> site_id via matching name (events.json doesn't carry
    # site_id directly, so recover it from the daily series' site_name)
    name_to_site = daily.drop_duplicates("site_id").set_index("site_name")["site_id"].to_dict()

    forecasts = {}
    for ev in events:
        site_id = name_to_site.get(ev["name"])
        if site_id is None:
            continue
        site_daily = daily[daily.site_id == site_id]

        feat_row, history_14d = latest_feature_row(site_daily, ev["predicted_class"])

        Xr = featurize_row(feat_row, reg_cols)
        Xc = featurize_row(feat_row, clf_cols)

        frp_next = max(0.0, float(reg_model.predict(Xr)[0]))
        active_prob = float(clf_model.predict_proba(Xc)[0][list(clf_model.classes_).index(1)])

        # simple 3-day-ahead autoregressive chain, reusing the same
        # trained models by rolling the predicted value into the next
        # day's lag features
        chain_row = dict(feat_row)
        day3 = []
        for step in range(3):
            Xr_step = featurize_row(chain_row, reg_cols)
            Xc_step = featurize_row(chain_row, clf_cols)
            f_next = max(0.0, float(reg_model.predict(Xr_step)[0]))
            a_next = float(clf_model.predict_proba(Xc_step)[0][list(clf_model.classes_).index(1)])
            day3.append(dict(frp=round(f_next, 1), active_probability=round(a_next * 100, 1)))
            chain_row["frp_lag3"] = chain_row["frp_lag2"]
            chain_row["frp_lag2"] = chain_row["frp_lag1"]
            chain_row["frp_lag1"] = chain_row["frp_today"]
            chain_row["frp_today"] = f_next
            chain_row["rolling_mean_7"] = np.mean([chain_row["frp_lag1"], chain_row["frp_lag2"], chain_row["frp_lag3"], f_next])

        trend = trend_label(feat_row["rolling_mean_7"], frp_next)

        forecasts[ev["id"]] = dict(
            predicted_frp_next=round(frp_next, 1),
            recurrence_probability=round(active_prob * 100, 1),
            trend=trend,
            history_14d=[round(v, 1) for v in history_14d],
            forecast_3day=day3,
        )

    with open("outputs/forecasts.json", "w") as f:
        json.dump(forecasts, f, indent=2)
    print(f"Forecasted next-event activity for {len(forecasts)} sites -> outputs/forecasts.json")


if __name__ == "__main__":
    main()

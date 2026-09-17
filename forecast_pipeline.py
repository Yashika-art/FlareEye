"""
forecast_pipeline.py
----------------------
Turns raw detections into a per-site DAILY time series, then builds a
supervised table (lag/rolling features -> next-day target) used to train
a forecasting model. This is what lets FlareEye predict tomorrow's
thermal activity at a site, not just classify what's already been seen.
"""

import numpy as np
import pandas as pd

PERIOD_DAYS = 365
LOOKBACK_FOR_FEATURES = 7   # rolling window size
TRAIN_TEST_DAY_CUTOFF = 292  # ~80% of the year -> train, last ~20% -> test


def build_daily_series(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (site, day) for every day in the observation period,
    with frp=0 / active=0 on days that had no detection."""
    site_meta = raw.drop_duplicates("site_id")[
        ["site_id", "site_name", "state", "true_class", "lat", "lng",
         "dist_infra_km", "dist_road_km", "land_cover"]
    ].set_index("site_id")

    daily_frp = raw.groupby(["site_id", "day"])["frp"].max().reset_index()

    rows = []
    for site_id, meta in site_meta.iterrows():
        frp_by_day = dict(zip(daily_frp.loc[daily_frp.site_id == site_id, "day"],
                               daily_frp.loc[daily_frp.site_id == site_id, "frp"]))
        for day in range(PERIOD_DAYS):
            frp = frp_by_day.get(day, 0.0)
            rows.append(dict(
                site_id=site_id, day=day, frp=frp, active=int(frp > 0),
                site_name=meta.site_name, state=meta.state, true_class=meta.true_class,
                lat=meta.lat, lng=meta.lng, dist_infra_km=meta.dist_infra_km,
                dist_road_km=meta.dist_road_km, land_cover=meta.land_cover,
            ))
    return pd.DataFrame(rows)


def build_supervised_table(daily: pd.DataFrame, predicted_class_by_site: dict) -> pd.DataFrame:
    """For every (site, day) with day+1 still inside the period, build
    lag/rolling features from history up to `day`, and next-day targets."""
    samples = []
    for site_id, g in daily.groupby("site_id"):
        g = g.sort_values("day").reset_index(drop=True)
        frp_series = g["frp"].to_numpy()
        active_series = g["active"].to_numpy()
        first_active_day = g.loc[g.active == 1, "day"].min()
        first_active_day = 0 if pd.isna(first_active_day) else first_active_day

        for t in range(1, len(g) - 1):  # need at least 1 day of history + 1 future day
            day = g.loc[t, "day"]
            window = frp_series[max(0, t - LOOKBACK_FOR_FEATURES + 1): t + 1]
            active_window = active_series[max(0, t - LOOKBACK_FOR_FEATURES + 1): t + 1]

            days_since_last = 0
            for back in range(t, -1, -1):
                if active_series[back] == 1:
                    days_since_last = t - back
                    break
            else:
                days_since_last = t

            samples.append(dict(
                site_id=site_id, day=int(day),
                frp_today=frp_series[t], frp_lag1=frp_series[t - 1],
                frp_lag2=frp_series[t - 2] if t >= 2 else 0.0,
                frp_lag3=frp_series[t - 3] if t >= 3 else 0.0,
                rolling_mean_7=float(np.mean(window)),
                rolling_max_7=float(np.max(window)),
                rolling_active_rate_7=float(np.mean(active_window)),
                days_since_last_detection=int(days_since_last),
                persistence_so_far=int(day - first_active_day) if day >= first_active_day else 0,
                cumulative_detections=int(np.sum(active_series[:t + 1])),
                dist_infra_km=g.loc[t, "dist_infra_km"],
                land_cover=g.loc[t, "land_cover"],
                predicted_class=predicted_class_by_site.get(site_id, g.loc[t, "true_class"]),
                target_frp_next=frp_series[t + 1],
                target_active_next=int(active_series[t + 1]),
            ))
    return pd.DataFrame(samples)


NUMERIC_FORECAST_FEATURES = [
    "frp_today", "frp_lag1", "frp_lag2", "frp_lag3",
    "rolling_mean_7", "rolling_max_7", "rolling_active_rate_7",
    "days_since_last_detection", "persistence_so_far", "cumulative_detections",
    "dist_infra_km",
]
CATEGORICAL_FORECAST_FEATURES = ["land_cover", "predicted_class"]

"""
pipeline.py
------------
Stage 2-3 of the FlareEye pipeline: turn raw point detections into
spatio-temporal "events" (DBSCAN) and engineer the multi-source feature
set (spatial + temporal + thermal + optical/environmental) used by the
classifier.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

EARTH_RADIUS_KM = 6371.0


def haversine_matrix_radians(coords_deg: np.ndarray) -> np.ndarray:
    return np.radians(coords_deg)


def cluster_events(df: pd.DataFrame, eps_km: float = 3.0, min_samples: int = 1) -> pd.DataFrame:
    """Cluster raw detections into events using DBSCAN with a haversine metric.

    Detections that fall within eps_km of each other are treated as the
    same recurring thermal event/site, mirroring FlareEye's step 3
    ('spatio-temporal clustering using DBSCAN').
    """
    coords = haversine_matrix_radians(df[["lat", "lng"]].to_numpy())
    eps_rad = eps_km / EARTH_RADIUS_KM
    labels = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine").fit_predict(coords)
    df = df.copy()
    df["event_cluster"] = labels
    return df


def _mode(series: pd.Series):
    return series.mode().iloc[0] if not series.mode().empty else series.iloc[0]


def build_event_features(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw detections within each DBSCAN cluster into one
    feature row per event, the unit the classifier operates on."""
    groups = []
    for cluster_id, g in df_clustered.groupby("event_cluster"):
        if cluster_id == -1:
            # treat unclustered noise points as singleton events too
            for _, row in g.iterrows():
                groups.append(_aggregate_group(pd.DataFrame([row])))
            continue
        groups.append(_aggregate_group(g))
    return pd.DataFrame(groups)


def _aggregate_group(g: pd.DataFrame) -> dict:
    persistence_days = int(g["day"].max() - g["day"].min() + 1)
    recurrence_count = int(len(g))
    return dict(
        site_name=_mode(g["site_name"]),
        state=_mode(g["state"]),
        true_class=_mode(g["true_class"]),
        lat=g["lat"].mean(),
        lng=g["lng"].mean(),
        frp_mean=g["frp"].mean(),
        frp_max=g["frp"].max(),
        brightness_mean=g["brightness_k"].mean(),
        brightness_max=g["brightness_k"].max(),
        confidence_mean=g["confidence"].mean(),
        wind_mean=g["wind_kmh"].mean(),
        swir_mean=g["swir_reflectance"].mean(),
        dist_infra_km=g["dist_infra_km"].mean(),
        dist_road_km=g["dist_road_km"].mean(),
        land_cover=_mode(g["land_cover"]),
        persistence_days=persistence_days,
        recurrence_count=recurrence_count,
        activity_rate=round(recurrence_count / max(1, persistence_days), 3),
        source=_mode(g["source"]),
    )


NUMERIC_FEATURES = [
    "frp_mean", "frp_max", "brightness_mean", "brightness_max", "confidence_mean",
    "wind_mean", "swir_mean", "dist_infra_km", "dist_road_km",
    "persistence_days", "recurrence_count", "activity_rate",
]
CATEGORICAL_FEATURES = ["land_cover"]

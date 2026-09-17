"""
classify_events.py
--------------------
Stage 5 of the FlareEye pipeline: run the trained classifier over every
event, attach real class-probability distributions, and compute a risk
score/priority. Emits outputs/events.json in the shape the dashboard
frontend expects, so the UI is driven by actual model output.
"""

import json
import joblib
import numpy as np
import pandas as pd

from pipeline import NUMERIC_FEATURES, CATEGORICAL_FEATURES

CLASSES = ["Industrial Fire", "Gas Flare", "Wildfire", "Agricultural Burning", "Industrial Heat"]

# base risk weight per predicted class -- an industrial fire is
# inherently higher priority than a routine, permitted gas flare even
# at equal confidence.
BASE_RISK = {
    "Industrial Fire": 78,
    "Wildfire": 68,
    "Agricultural Burning": 38,
    "Gas Flare": 22,
    "Industrial Heat": 12,
}


def load_model():
    bundle = joblib.load("models/classifier.joblib")
    return bundle["model"], bundle["columns"]


def featurize(events: pd.DataFrame, columns: list) -> pd.DataFrame:
    X = pd.get_dummies(events[NUMERIC_FEATURES + CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES)
    # align to the columns the model was trained on
    for col in columns:
        if col not in X.columns:
            X[col] = 0
    return X[columns]


def risk_from(pred_class: str, proba: dict, frp_mean: float, persistence_days: int, recurrence_count: int) -> tuple:
    base = BASE_RISK[pred_class]
    confidence = proba[pred_class]

    # sudden, high-power, low-persistence events are riskier than the
    # same class showing up as a long, stable baseline
    novelty_bonus = 18 if persistence_days <= 7 else (8 if persistence_days <= 30 else 0)
    frp_bonus = min(15, frp_mean / 25)
    uncertainty_penalty = (1 - confidence / 100) * 10  # low-confidence calls get flagged, not buried

    score = base + novelty_bonus + frp_bonus + uncertainty_penalty
    score = int(np.clip(score, 1, 99))
    level = "High" if score >= 65 else ("Medium" if score >= 35 else "Low")
    return score, level


def confidence_tier(pct: float) -> str:
    return "A" if pct >= 85 else ("B" if pct >= 65 else "C")


def main():
    events = pd.read_csv("data/events_with_split.csv")
    model, columns = load_model()
    X = featurize(events, columns)

    proba_matrix = model.predict_proba(X)
    class_order = list(model.classes_)
    preds = model.predict(X)

    out_events = []
    for i, row in events.reset_index(drop=True).iterrows():
        proba = {c: round(float(proba_matrix[i][class_order.index(c)]) * 100, 1) for c in CLASSES}
        pred_class = preds[i]
        conf_pct = proba[pred_class]
        score, level = risk_from(
            pred_class, proba, row["frp_mean"], int(row["persistence_days"]), int(row["recurrence_count"])
        )
        out_events.append(dict(
            id=f"FE-{1000 + i}",
            name=row["site_name"],
            state=row["state"],
            lat=round(float(row["lat"]), 4),
            lng=round(float(row["lng"]), 4),
            true_class=row["true_class"],
            predicted_class=pred_class,
            correct=bool(pred_class == row["true_class"]),
            split=row["split"],
            probabilities=proba,
            confidence=conf_pct,
            confidence_tier=confidence_tier(conf_pct),
            frp=round(float(row["frp_mean"]), 1),
            brightness_k=round(float(row["brightness_mean"]), 1),
            wind_kmh=round(float(row["wind_mean"]), 1),
            dist_infra_km=round(float(row["dist_infra_km"]), 2),
            land_cover=row["land_cover"],
            persistence_days=int(row["persistence_days"]),
            recurrence_count=int(row["recurrence_count"]),
            source=row["source"],
            risk_score=score,
            risk_level=level,
        ))

    with open("outputs/events.json", "w") as f:
        json.dump(out_events, f, indent=2)

    n_correct = sum(e["correct"] for e in out_events)
    print(f"Classified {len(out_events)} events. Model agreed with ground truth on {n_correct}/{len(out_events)}.")
    print("Saved -> outputs/events.json")


if __name__ == "__main__":
    main()

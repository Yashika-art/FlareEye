"""
train_forecast_model.py
-------------------------
Trains the predictive layer on top of the classifier:
  1. A RandomForestRegressor that predicts tomorrow's radiative power (FRP)
     at a site from its recent history.
  2. A RandomForestClassifier that predicts whether the site will still be
     active (produce a detection) tomorrow.

Evaluated on a TIME-based holdout (train on days 0-291, test on days
292-364) -- the correct split for a forecasting task, as opposed to the
site-level split used for the classifier.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score

from forecast_pipeline import (
    build_daily_series, build_supervised_table, TRAIN_TEST_DAY_CUTOFF,
    NUMERIC_FORECAST_FEATURES, CATEGORICAL_FORECAST_FEATURES,
)


def featurize(table: pd.DataFrame):
    X = pd.get_dummies(table[NUMERIC_FORECAST_FEATURES + CATEGORICAL_FORECAST_FEATURES],
                        columns=CATEGORICAL_FORECAST_FEATURES)
    return X


def main():
    raw = pd.read_csv("data/raw_detections.csv")

    # reuse the classifier's predictions as an input feature to the
    # forecaster, exactly like a real deployment: classify first, then
    # forecast what a site of that predicted type will do next.
    events = pd.read_json("outputs/events.json")
    predicted_class_by_site = dict(zip(
        events["name"].map(lambda n: raw.loc[raw.site_name == n, "site_id"].iloc[0]
                            if (raw.site_name == n).any() else None),
        events["predicted_class"],
    ))
    predicted_class_by_site = {k: v for k, v in predicted_class_by_site.items() if k is not None}

    print("Building daily time series...")
    daily = build_daily_series(raw)
    daily.to_csv("data/daily_series.csv", index=False)

    print("Building supervised forecasting table...")
    table = build_supervised_table(daily, predicted_class_by_site)
    table.to_csv("data/forecast_table.csv", index=False)
    print(f"{len(table)} (site, day) training samples")

    X = featurize(table)
    y_reg = table["target_frp_next"]
    y_clf = table["target_active_next"]

    train_mask = table["day"] < TRAIN_TEST_DAY_CUTOFF
    test_mask = ~train_mask

    X_train, X_test = X[train_mask], X[test_mask]
    yreg_train, yreg_test = y_reg[train_mask], y_reg[test_mask]
    yclf_train, yclf_test = y_clf[train_mask], y_clf[test_mask]

    print(f"train samples: {train_mask.sum()}   test samples: {test_mask.sum()}")

    reg = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=7)
    reg.fit(X_train, yreg_train)
    pred_reg = reg.predict(X_test)
    mae = mean_absolute_error(yreg_test, pred_reg)
    r2 = r2_score(yreg_test, pred_reg)

    clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3,
                                  class_weight="balanced", random_state=7)
    clf.fit(X_train, yclf_train)
    pred_clf = clf.predict(X_test)
    acc = accuracy_score(yclf_test, pred_clf)
    f1 = f1_score(yclf_test, pred_clf)

    print(f"\nFRP regressor  -> MAE {mae:.2f} MW   R2 {r2:.3f}")
    print(f"Recurrence clf -> accuracy {acc:.3f}   F1 {f1:.3f}")

    joblib.dump({"model": reg, "columns": list(X.columns)}, "models/forecast_regressor.joblib")
    joblib.dump({"model": clf, "columns": list(X.columns)}, "models/forecast_classifier.joblib")

    metrics = dict(
        frp_mae=round(float(mae), 2),
        frp_r2=round(float(r2), 4),
        recurrence_accuracy=round(float(acc), 4),
        recurrence_f1=round(float(f1), 4),
        n_train_samples=int(train_mask.sum()),
        n_test_samples=int(test_mask.sum()),
        train_test_day_cutoff=TRAIN_TEST_DAY_CUTOFF,
    )
    with open("outputs/forecast_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved models -> models/forecast_regressor.joblib, models/forecast_classifier.joblib")
    print("Saved metrics -> outputs/forecast_metrics.json")


if __name__ == "__main__":
    main()

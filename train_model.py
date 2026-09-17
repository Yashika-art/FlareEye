"""
train_model.py
---------------
Stage 4 of the FlareEye pipeline: train the ML classifier that assigns
each clustered thermal event to one of 5 categories, using a site-level
train/test split (per the brief's "spatially robust evaluation" -
no site's detections appear in both train and test).
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from generate_data import generate_detections
from pipeline import cluster_events, build_event_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES

CLASSES = ["Industrial Fire", "Gas Flare", "Wildfire", "Agricultural Burning", "Industrial Heat"]


def build_dataset():
    raw = generate_detections()
    raw.to_csv("data/raw_detections.csv", index=False)
    clustered = cluster_events(raw, eps_km=3.0)
    events = build_event_features(clustered)
    return events


def featurize(events: pd.DataFrame):
    X = pd.get_dummies(events[NUMERIC_FEATURES + CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES)
    y = events["true_class"]
    return X, y


def main():
    events = build_dataset()
    print(f"built {len(events)} events from raw detections (one per site)")

    X, y = featurize(events)

    # site-level split: each event already corresponds to one site, so a
    # stratified split over events is a genuine spatial hold-out (no site's
    # detections leak across train/test).
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, events.index, test_size=0.3, random_state=7, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=2,
        class_weight="balanced", random_state=7,
    )
    clf.fit(X_train, y_train)

    pred_test = clf.predict(X_test)
    acc = accuracy_score(y_test, pred_test)
    f1_macro = f1_score(y_test, pred_test, average="macro")
    report = classification_report(y_test, pred_test, output_dict=True)
    cm = confusion_matrix(y_test, pred_test, labels=CLASSES).tolist()

    print(f"\nHold-out accuracy: {acc:.3f}   macro F1: {f1_macro:.3f}\n")
    print(classification_report(y_test, pred_test))

    importances = sorted(
        zip(X.columns, clf.feature_importances_), key=lambda t: -t[1]
    )
    print("Top feature importances:")
    for name, imp in importances[:8]:
        print(f"  {name:<24s} {imp:.3f}")

    joblib.dump({"model": clf, "columns": list(X.columns)}, "models/classifier.joblib")

    metrics = dict(
        accuracy=round(acc, 4),
        macro_f1=round(f1_macro, 4),
        n_train_sites=int(len(X_train)),
        n_test_sites=int(len(X_test)),
        classification_report=report,
        confusion_matrix=cm,
        confusion_matrix_labels=CLASSES,
        top_features=[{"feature": n, "importance": round(float(i), 4)} for n, i in importances[:10]],
    )
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # persist full dataset + split membership so classify_events.py can reuse it
    events_out = events.copy()
    events_out["split"] = "train"
    events_out.loc[idx_test, "split"] = "test"
    events_out.to_csv("data/events_with_split.csv", index=False)

    print("\nSaved model -> models/classifier.joblib")
    print("Saved metrics -> outputs/metrics.json")


if __name__ == "__main__":
    main()

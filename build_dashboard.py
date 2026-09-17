"""
build_dashboard.py
--------------------
Stage 6: inject the real classifier output (outputs/events.json,
outputs/metrics.json) into the dashboard HTML shell, so the frontend is
driven entirely by the trained model -- no server required.
"""
import json

events = json.load(open("outputs/events.json"))
metrics = json.load(open("outputs/metrics.json"))
forecasts = json.load(open("outputs/forecasts.json"))
forecast_metrics = json.load(open("outputs/forecast_metrics.json"))

html = open("dashboard_shell.html").read()
html = html.replace("__EVENTS_JSON__", json.dumps(events))
html = html.replace("__METRICS_JSON__", json.dumps(metrics))
html = html.replace("__FORECASTS_JSON__", json.dumps(forecasts))
html = html.replace("__FORECAST_METRICS_JSON__", json.dumps(forecast_metrics))

with open("outputs/flareeye_ai_dashboard.html", "w") as f:
    f.write(html)

print(f"built outputs/flareeye_ai_dashboard.html ({len(html):,} chars, {len(events)} events embedded)")

"""
Entry point for the Weather Forecasting & Anomaly Detection app.

Run with:
    python app.py

All logic lives in the ``webapp/`` package:
    webapp/config.py       – constants & paths
    webapp/models.py       – MLPStudentModel + model loaders
    webapp/weather_api.py  – Open-Meteo API helpers
    webapp/inference.py    – predict_forecast() & detect_anomalies()
    webapp/routes.py       – Flask route handlers
"""

import logging

from flask import Flask

from webapp.models import load_anomaly_model, load_forecast_model
from webapp.routes import api_bp

# ── Logging ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── App Factory ──────────────────────────────────────────────────────

app = Flask(__name__)
app.register_blueprint(api_bp)


if __name__ == "__main__":
    load_forecast_model()      # pre-load on startup
    load_anomaly_model()       # pre-load on startup
    app.run(debug=True, host="0.0.0.0", port=5000)

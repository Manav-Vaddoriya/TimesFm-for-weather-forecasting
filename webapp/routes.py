"""
Flask route handlers.

Blueprint: ``api_bp`` — mounted at ``/`` in the main app.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from flask import Blueprint, jsonify, render_template, request

from webapp.config import HISTORY_HOURS, PATCH_LEN, TOTAL_HOURS
from webapp.inference import detect_anomalies, predict_forecast
from webapp.models import load_anomaly_model, load_forecast_model
from webapp.weather_api import fetch_current_weather, fetch_historical_temps, geocode_city

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


# ── Pages ────────────────────────────────────────────────────────────

@api_bp.route("/")
def index():
    return render_template("index.html")


# ── API ──────────────────────────────────────────────────────────────

@api_bp.route("/geocode", methods=["GET"])
def geocode():
    """Geocode a city name to coordinates."""
    city = request.args.get("city", "").strip()
    if not city:
        return jsonify({"error": "City name required"}), 400
    try:
        return jsonify(geocode_city(city))
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@api_bp.route("/predict", methods=["POST"])
def predict():
    """
    Fetch the last 72 hours of weather and predict the next 32 hours.

    Request JSON:
        { "latitude": float, "longitude": float }

    Response JSON:
        {
            "history":  { "times": [...], "temps": [...] },
            "prediction": { "times": [...], "temps": [...] },
            "anomaly_probabilities": [...],
            "current": { ... },
            "model_info": { ... }
        }
    """
    data = request.get_json()
    lat = data.get("latitude")
    lon = data.get("longitude")

    if lat is None or lon is None:
        return jsonify({"error": "latitude and longitude required"}), 400

    try:
        logger.info("Fetching weather data for (%.2f, %.2f)", lat, lon)

        # 1. Fetch the last 72 hours of temperature
        history = fetch_historical_temps(lat, lon, hours=HISTORY_HOURS)

        if len(history) < PATCH_LEN:
            return jsonify({"error": "Insufficient historical data available"}), 422

        temps = [h["temp"] for h in history]
        times = [h["time"] for h in history]

        # 2. Run forecast (uses 72 h, internally pads to 1024)
        predicted = predict_forecast(temps)

        # 3. Run anomaly detection
        anomaly_probs = detect_anomalies(temps, predicted)

        # 4. Generate future timestamps
        last_time = datetime.fromisoformat(times[-1])
        future_times = [
            (last_time + timedelta(hours=i + 1)).strftime("%Y-%m-%dT%H:%M")
            for i in range(len(predicted))
        ]

        # 5. Current weather
        current = fetch_current_weather(lat, lon)

        # 6. Model metadata
        forecast_m = load_forecast_model()
        anomaly_m = load_anomaly_model()

        return jsonify({
            "history": {
                "times": times,
                "temps": temps,
            },
            "prediction": {
                "times": future_times,
                "temps": [round(t, 2) for t in predicted],
            },
            "anomaly_probabilities": anomaly_probs,
            "current": current,
            "model_info": {
                "forecast_parameters": f"{forecast_m.count_parameters():,}",
                "forecast_architecture": "TimesFMLiteGPT (Decoder-only Transformer)",
                "anomaly_parameters": f"{sum(p.numel() for p in anomaly_m.parameters()):,}",
                "anomaly_architecture": "MLPStudentModel (Knowledge Distillation)",
                "context_window": f"{HISTORY_HOURS} hours ({HISTORY_HOURS // 24} days)",
                "forecast_horizon": f"{PATCH_LEN} hours",
            },
        })

    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500

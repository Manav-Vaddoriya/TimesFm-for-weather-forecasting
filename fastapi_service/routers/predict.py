"""
Predict router — POST /predict

Matches the Flask webapp contract exactly.  PyTorch inference is
blocking/sync, so we delegate to ``run_in_threadpool`` to avoid
stalling the async event loop.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from webapp.config import HISTORY_HOURS, PATCH_LEN
from webapp.inference import detect_anomalies, predict_forecast
from webapp.models import load_anomaly_model, load_forecast_model

from fastapi_service.schemas import (
    CurrentWeather,
    ErrorResponse,
    HistoryData,
    ModelInfo,
    PredictionData,
    PredictRequest,
    PredictResponse,
)
from fastapi_service.weather_client import (
    fetch_current_weather,
    fetch_historical_temps,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Missing coordinates"},
        422: {"model": ErrorResponse, "description": "Insufficient data"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
    },
)
async def predict(body: PredictRequest):
    """
    Fetch the last 72 hours of weather and predict the next 32 hours.

    Request JSON::

        { "latitude": float, "longitude": float }

    Response JSON::

        {
            "history":  { "times": [...], "temps": [...] },
            "prediction": { "times": [...], "temps": [...] },
            "anomaly_probabilities": [...],
            "current": { ... },
            "model_info": { ... }
        }
    """
    try:
        logger.info("Fetching weather data for (%.2f, %.2f)", body.latitude, body.longitude)

        # 1. Fetch the last 72 hours of temperature (async)
        history = await fetch_historical_temps(
            body.latitude, body.longitude, hours=HISTORY_HOURS
        )

        if len(history) < PATCH_LEN:
            raise HTTPException(
                status_code=422, detail="Insufficient historical data available"
            )

        temps = [h["temp"] for h in history]
        times = [h["time"] for h in history]

        # 2. Run forecast — blocking PyTorch, offload to threadpool
        predicted = await run_in_threadpool(predict_forecast, temps)

        # 3. Run anomaly detection — blocking PyTorch, offload to threadpool
        anomaly_probs = await run_in_threadpool(detect_anomalies, temps, predicted)

        # 4. Generate future timestamps
        last_time = datetime.fromisoformat(times[-1])
        future_times = [
            (last_time + timedelta(hours=i + 1)).strftime("%Y-%m-%dT%H:%M")
            for i in range(len(predicted))
        ]

        # 5. Current weather (async)
        current = await fetch_current_weather(body.latitude, body.longitude)

        # 6. Model metadata
        forecast_m = load_forecast_model()
        anomaly_m = load_anomaly_model()

        return PredictResponse(
            history=HistoryData(times=times, temps=temps),
            prediction=PredictionData(
                times=future_times,
                temps=[round(t, 2) for t in predicted],
            ),
            anomaly_probabilities=anomaly_probs,
            current=CurrentWeather(**current),
            model_info=ModelInfo(
                forecast_parameters=f"{forecast_m.count_parameters():,}",
                forecast_architecture="TimesFMLiteGPT (Decoder-only Transformer)",
                anomaly_parameters=f"{sum(p.numel() for p in anomaly_m.parameters()):,}",
                anomaly_architecture="MLPStudentModel (Knowledge Distillation)",
                context_window=f"{HISTORY_HOURS} hours ({HISTORY_HOURS // 24} days)",
                forecast_horizon=f"{PATCH_LEN} hours",
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

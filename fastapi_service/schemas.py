"""
Pydantic models for request/response validation and OpenAPI docs.

Every field name matches the Flask webapp's JSON contract exactly so the
existing frontend can point at either backend without changes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Geocode ──────────────────────────────────────────────────────────

class GeocodeResponse(BaseModel):
    name: str
    country: str
    latitude: float
    longitude: float


# ── Predict request ──────────────────────────────────────────────────

class PredictRequest(BaseModel):
    latitude: float
    longitude: float


# ── Predict response (nested models) ────────────────────────────────

class HistoryData(BaseModel):
    times: list[str]
    temps: list[float]


class PredictionData(BaseModel):
    times: list[str]
    temps: list[float]


class CurrentWeather(BaseModel):
    temperature: float | None = None
    humidity: float | None = None
    wind_speed: float | None = None
    weather_code: int = 0


class ModelInfo(BaseModel):
    forecast_parameters: str
    forecast_architecture: str
    anomaly_parameters: str
    anomaly_architecture: str
    context_window: str
    forecast_horizon: str


class PredictResponse(BaseModel):
    history: HistoryData
    prediction: PredictionData
    anomaly_probabilities: list[float]
    current: CurrentWeather
    model_info: ModelInfo


# ── Error ────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str

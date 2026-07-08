"""
Async Open-Meteo API helpers using httpx.

Drop-in replacements for ``webapp.weather_api`` functions, but fully
async and non-blocking.  Same parameters and return shapes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import httpx

from webapp.config import ARCHIVE_URL, FORECAST_URL, GEOCODING_URL, HISTORY_HOURS

logger = logging.getLogger(__name__)


async def geocode_city(city: str) -> dict:
    """Look up city coordinates via Open-Meteo Geocoding API."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            GEOCODING_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

    if "results" not in data or not data["results"]:
        raise ValueError(f"City '{city}' not found")

    r = data["results"][0]
    return {
        "name": r.get("name", city),
        "country": r.get("country", ""),
        "latitude": r["latitude"],
        "longitude": r["longitude"],
    }


async def fetch_historical_temps(
    lat: float, lon: float, hours: int = HISTORY_HOURS
) -> list[dict]:
    """
    Fetch the last ``hours`` of hourly temperature_2m from Open-Meteo.

    Default is 72 hours (3 days).  Uses the forecast API with ``past_days``
    so we always get the most-recent observations without needing the
    archive endpoint for such a short window.

    Returns a list of ``{"time": ..., "temp": ...}`` dicts sorted by time.
    """
    past_days = -(-hours // 24)  # ceiling division

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": "temperature_2m",
                    "timezone": "UTC",
                    "past_days": past_days,
                    "forecast_days": 1,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.error("Failed to fetch historical temps: %s", e)
        raise RuntimeError("Failed to fetch temperature data from Open-Meteo") from e

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])

    if not temps:
        raise RuntimeError("No temperature data returned from Open-Meteo")

    # Filter out nulls and pair up
    entries = [
        {"time": t, "temp": v}
        for t, v in zip(times, temps)
        if v is not None
    ]

    # Keep only entries up to "now" (drop future forecast hours)
    now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M")
    entries = [e for e in entries if e["time"] <= now_str]

    # Take the last `hours` entries
    return entries[-hours:]


async def fetch_current_weather(lat: float, lon: float) -> dict:
    """Fetch current weather conditions (temp, humidity, wind)."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                    "timezone": "UTC",
                },
                timeout=10,
            )
            resp.raise_for_status()
            c = resp.json().get("current", {})
            return {
                "temperature": c.get("temperature_2m"),
                "humidity": c.get("relative_humidity_2m"),
                "wind_speed": c.get("wind_speed_10m"),
                "weather_code": c.get("weather_code", 0),
            }
    except Exception as e:
        logger.warning("Current weather fetch failed: %s", e)
        return {"temperature": None, "humidity": None, "wind_speed": None, "weather_code": 0}

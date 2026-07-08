"""
Geocode router — GET /geocode?city=...

Matches the Flask webapp contract exactly.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from fastapi_service.schemas import ErrorResponse, GeocodeResponse
from fastapi_service.weather_client import geocode_city

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/geocode",
    response_model=GeocodeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Missing city parameter"},
        404: {"model": ErrorResponse, "description": "City not found"},
    },
)
async def geocode(city: str = Query("", description="City name to geocode")):
    """Geocode a city name to coordinates."""
    city = city.strip()
    if not city:
        raise HTTPException(status_code=400, detail="City name required")
    try:
        result = await geocode_city(city)
        return result
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

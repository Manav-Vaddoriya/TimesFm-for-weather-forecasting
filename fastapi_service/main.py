"""
FastAPI application entry point.

Lifespan context manager pre-loads both PyTorch models once at startup.
CORS is open so the existing frontend can target this service on :8000
just as easily as Flask on :5000.

Run with:
    uvicorn fastapi_service.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from webapp.models import load_anomaly_model, load_forecast_model

from fastapi_service.routers import geocode, predict

# ── Logging ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load models once at startup, not per-request."""
    logger.info("FastAPI startup — loading models …")
    load_forecast_model()
    load_anomaly_model()
    logger.info("Models ready.")
    yield
    logger.info("FastAPI shutdown.")


# ── App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Weather Forecasting & Anomaly Detection API",
    description=(
        "Async API layer backed by a TimesFMLiteGPT forecasting model "
        "and an MLPStudentModel anomaly detector.  Same contract as the "
        "Flask UI on :5000."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────

app.include_router(geocode.router)
app.include_router(predict.router)


# ── Health ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Simple liveness probe."""
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to the interactive API docs."""
    return RedirectResponse(url="/docs")


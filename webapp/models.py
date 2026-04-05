"""
Model definitions and lazy-loading helpers.

Contains:
    - MLPStudentModel   : distilled anomaly-detection MLP
    - load_forecast_model()   : lazy-loads TimesFMLiteGPT
    - load_anomaly_model()    : lazy-loads MLPStudentModel
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from src.pretrained_model.timesfm_lite import TimesFMLiteGPT
from webapp.config import (
    ANOMALY_MODEL_CONFIG,
    ANOMALY_MODEL_PATH,
    FORECAST_MODEL_CONFIG,
    FORECAST_MODEL_PATH,
)

logger = logging.getLogger(__name__)

# ── Device ───────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"


# ── MLPStudentModel ──────────────────────────────────────────────────

class MLPStudentModel(nn.Module):
    """Distilled MLP for anomaly detection via knowledge distillation."""

    def __init__(
        self,
        window_size: int = 32,
        hidden_dims: list | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layer_sizes = [window_size] + hidden_dims
        layers: list[nn.Module] = []

        for i in range(len(layer_sizes) - 1):
            layers += [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                nn.BatchNorm1d(layer_sizes[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]

        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


# ── Lazy Singletons ─────────────────────────────────────────────────

_forecast_model: TimesFMLiteGPT | None = None
_anomaly_model: MLPStudentModel | None = None


def load_forecast_model() -> TimesFMLiteGPT:
    """Load the pretrained TimesFMLiteGPT forecasting model (lazy)."""
    global _forecast_model
    if _forecast_model is not None:
        return _forecast_model

    logger.info("Loading forecasting model from %s on %s", FORECAST_MODEL_PATH, device)
    m = TimesFMLiteGPT(FORECAST_MODEL_CONFIG)

    if FORECAST_MODEL_PATH.exists():
        state = torch.load(FORECAST_MODEL_PATH, map_location=device, weights_only=True)
        m.load_state_dict(state)
        logger.info("Forecasting model loaded (%s params)", f"{m.count_parameters():,}")
    else:
        logger.warning("Model not found at %s — using random weights!", FORECAST_MODEL_PATH)

    m.to(device).eval()
    _forecast_model = m
    return m


def load_anomaly_model() -> MLPStudentModel:
    """Load the distilled MLP anomaly-detection model (lazy)."""
    global _anomaly_model
    if _anomaly_model is not None:
        return _anomaly_model

    logger.info("Loading anomaly model from %s", ANOMALY_MODEL_PATH)
    m = MLPStudentModel(
        window_size=ANOMALY_MODEL_CONFIG["window_size"],
        hidden_dims=ANOMALY_MODEL_CONFIG["hidden_dims"],
        dropout=ANOMALY_MODEL_CONFIG["dropout"],
    )

    if ANOMALY_MODEL_PATH.exists():
        state = torch.load(ANOMALY_MODEL_PATH, map_location=device, weights_only=True)
        m.load_state_dict(state)
        params = sum(p.numel() for p in m.parameters())
        logger.info("Anomaly model loaded (%s params)", f"{params:,}")
    else:
        logger.warning("Anomaly model not found at %s!", ANOMALY_MODEL_PATH)

    m.to(device).eval()
    _anomaly_model = m
    return m

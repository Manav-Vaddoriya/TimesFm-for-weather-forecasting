"""
Application-wide configuration constants.

All model paths, hyperparameters, and API URLs live here so every
other module can simply ``from webapp.config import ...``.
"""

from pathlib import Path

# ── Model Paths ──────────────────────────────────────────────────────
FORECAST_MODEL_PATH = Path("models/pretrained model/best_model.pth")
ANOMALY_MODEL_PATH = Path("models/distilled model/best_student_mlp.pt")

# ── Model Hyperparameters ────────────────────────────────────────────
PATCH_LEN = 32
CONTEXT_LEN = 32
TOTAL_HOURS = PATCH_LEN * CONTEXT_LEN  # 1024 hours ≈ 42 days

# The model expects (1, CONTEXT_LEN, PATCH_LEN) = (1, 32, 32) = 1024 values
# but we only fetch 72 hours of real data and pad the rest.
HISTORY_HOURS = 72  # 3 days of context shown to the user / fed to model

FORECAST_MODEL_CONFIG = {
    "patch_len": PATCH_LEN,
    "d_model": 128,
    "n_layers": 6,
    "n_heads": 4,
    "d_ff": 512,
    "dropout": 0.0,
    "context_len": CONTEXT_LEN,
}

ANOMALY_MODEL_CONFIG = {
    "window_size": 32,
    "hidden_dims": [512, 128, 64, 32],
    "dropout": 0.0,
}

# ── Open-Meteo API URLs ─────────────────────────────────────────────
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

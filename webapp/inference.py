"""
Inference helpers for forecasting and anomaly detection.

Functions:
    predict_forecast()   – run TimesFMLiteGPT on temperature history
    detect_anomalies()   – run MLPStudentModel on predicted temps
"""

from __future__ import annotations

import numpy as np
import torch

from webapp.config import ANOMALY_MODEL_CONFIG, CONTEXT_LEN, PATCH_LEN, TOTAL_HOURS
from webapp.models import device, load_anomaly_model, load_forecast_model


def predict_forecast(temperatures: list[float]) -> list[float]:
    """
    Run forecasting inference.

    The model expects (1, CONTEXT_LEN, PATCH_LEN) = (1, 32, 32) = 1024 values.
    If fewer than 1024 are supplied (e.g. 72), the sequence is padded on the
    left by repeating the earliest value.

    Returns: list of PATCH_LEN (32) predicted future temperatures.
    """
    m = load_forecast_model()

    arr = np.array(temperatures, dtype=np.float32)

    # Pad to TOTAL_HOURS if needed
    if len(arr) < TOTAL_HOURS:
        pad_len = TOTAL_HOURS - len(arr)
        arr = np.pad(arr, (pad_len, 0), mode="edge")

    patches = arr[-TOTAL_HOURS:].reshape(1, CONTEXT_LEN, PATCH_LEN)
    x = torch.from_numpy(patches).to(device)

    with torch.no_grad():
        output = m(x)  # (1, CONTEXT_LEN, PATCH_LEN)

    return output[0, -1, :].cpu().numpy().tolist()


def detect_anomalies(
    history_temps: list[float],
    predicted_temps: list[float],
) -> list[float]:
    """
    For each predicted hour, take a sliding window of the preceding 32
    temperature values and run it through the distilled MLP.

    Returns a list of raw sigmoid probabilities ∈ [0, 1].
    """
    mlp = load_anomaly_model()
    window_size = ANOMALY_MODEL_CONFIG["window_size"]

    combined = history_temps[-window_size:] + predicted_temps
    probabilities: list[float] = []

    with torch.no_grad():
        for i in range(len(predicted_temps)):
            window = combined[i : i + window_size]
            if len(window) < window_size:
                probabilities.append(0.5)
                continue

            x = torch.tensor([window], dtype=torch.float32).to(device)
            prob = mlp(x).item()
            probabilities.append(round(prob, 4))

    return probabilities

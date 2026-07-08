"""
fastapi_service — async FastAPI microservice for Weather Forecasting & Anomaly Detection.

Mirrors the Flask webapp's /geocode and /predict API contract using
async httpx for weather data and run_in_threadpool for blocking PyTorch inference.
"""

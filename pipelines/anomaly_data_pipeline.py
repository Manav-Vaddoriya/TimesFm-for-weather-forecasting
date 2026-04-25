"""
Anomaly-detection data preparation pipeline (knowledge distillation).

Generates the distillation dataset from existing patch files:
  Stage 1 — Load the pre-trained teacher model (TimesFMLiteGPT).
  Stage 2 — For each patch file, generate soft labels per series via teacher
             inference (sliding-window approach from the notebook).
  Stage 3 — Assign hard labels using per-series percentile thresholds.
  Stage 4 — Save partitioned Parquet files for downstream training.

Run with:
    python -m pipelines.anomaly_data_pipeline
"""

from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from config.settings import settings
from src.pretrained_model.timesfm_lite import TimesFMLiteGPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _discover_patch_metadata_pairs(output_dir: Path) -> List[Tuple[Path, Path]]:
    """Find matching (patches.npy, metadata.parquet) pairs under output_dir."""
    patch_files = sorted(output_dir.rglob("weather_part_*_patches.npy"))
    pairs = []
    for pf in patch_files:
        mf = pf.parent / pf.name.replace("_patches.npy", "_metadata.parquet")
        if mf.exists():
            pairs.append((pf, mf))
        else:
            logger.warning("No metadata found for %s — skipping.", pf.name)
    if not pairs:
        raise FileNotFoundError(
            f"No (patches, metadata) pairs found under {output_dir}. "
            "Run the data pipeline first: python -m pipelines.data_pipeline"
        )
    logger.info("Discovered %d patch/metadata pair(s) under %s", len(pairs), output_dir)
    return pairs


def _load_teacher_model(device: str) -> TimesFMLiteGPT:
    """Load the pre-trained TimesFMLiteGPT teacher model."""
    cfg = settings.training
    model_path = cfg.model_output_dir / "best_model.pth"

    model_config = cfg.to_model_config(settings.patches.patch_size)
    model = TimesFMLiteGPT(model_config)

    if model_path.exists():
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        logger.info("Teacher model loaded from %s (%s params)",
                     model_path, f"{model.count_parameters():,}")
    else:
        raise FileNotFoundError(
            f"Teacher model not found at {model_path}. "
            "Train the forecasting model first: python -m pipelines.train_pipeline"
        )

    model.to(device).eval()
    return model


# ── Core: soft-label generation for one series ──────────────────────────────

def _create_dataset_for_series(
    model: TimesFMLiteGPT,
    patches: np.ndarray,
    series_id: str,
    config: dict,
    device: str,
) -> List[dict]:
    """
    Port of notebook's `create_dataset_for_series`:
    Sliding-window teacher inference → soft labels via sigmoid(z-score).
    """
    context_len = config["context_len"]
    patch_len = config["patch_len"]

    concat_ts = patches.flatten()
    total_ts = len(concat_ts)

    n_patches = context_len // patch_len
    usable_ts = n_patches * patch_len

    rows: List[dict] = []
    for sample_idx in range(0, total_ts - context_len, 32):
        # 1. Raw context window and actual 33rd value
        context_window = concat_ts[sample_idx: sample_idx + context_len]
        actual_33rd = concat_ts[sample_idx + context_len]

        # 2. Mean/std of raw context
        actual_mean = float(np.mean(context_window))
        actual_std = float(np.std(context_window)) + 1e-8

        # 3. Normalized context
        normalized_context = (context_window - actual_mean) / actual_std

        # 4. Reshape for model (uses raw context, not normalized)
        model_input = context_window[:usable_ts].reshape(n_patches, patch_len)

        # 5. Teacher inference
        with torch.no_grad():
            x = torch.from_numpy(model_input.astype(np.float32)).unsqueeze(0).to(device)
            out = model(x)
            predicted_patches = out[0].cpu().numpy()

        # 6. Predicted mean from last patch
        predicted_mean = float(np.mean(predicted_patches[-1]))

        # 7. Soft label = sigmoid((actual_33rd - predicted_mean) / actual_std)
        score = (float(actual_33rd) - predicted_mean) / actual_std
        soft_label = float(torch.sigmoid(torch.tensor(score)).item())

        window_id = f"{series_id}_w{sample_idx:05d}"

        rows.append({
            "window_id": window_id,
            "series_id": series_id,
            "window_start": sample_idx,
            "window_end": sample_idx + context_len - 1,
            "context_window": [round(v, 4) for v in context_window.tolist()],
            "normalized_context": [round(v, 4) for v in normalized_context.tolist()],
            "actual_mean": round(actual_mean, 4),
            "actual_std": round(actual_std, 4),
            "actual_33rd": round(float(actual_33rd), 4),
            "predicted_patches": [[round(v, 4) for v in p] for p in predicted_patches.tolist()],
            "predicted_mean": round(predicted_mean, 4),
            "soft_label": round(soft_label, 4),
        })

    return rows


# ── Hard labels via percentile thresholds ────────────────────────────────────

def _assign_hard_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Port of notebook's hard-label logic:
    For each series_id, compute 99th/1st percentile of all predicted patches;
    mark windows where actual_33rd falls outside this range as anomalies.
    """
    df = df.copy()
    df["hard_label"] = 0

    for series_id in df["series_id"].unique():
        mask = df["series_id"] == series_id
        subset = df.loc[mask]

        # Flatten all predicted patches for this series
        combined = []
        for row in subset["predicted_patches"].values:
            for patch in row:
                combined.extend(patch)

        if not combined:
            continue

        q99 = np.quantile(combined, 0.99)
        q01 = np.quantile(combined, 0.01)

        # Anomaly: actual_33rd outside [q01, q99]
        anomaly_mask = mask & (
            (df["actual_33rd"] > q99) | (df["actual_33rd"] < q01)
        )
        df.loc[anomaly_mask, "hard_label"] = 1

    n_pos = int((df["hard_label"] == 1).sum())
    n_neg = int((df["hard_label"] == 0).sum())
    logger.info(
        "Hard labels assigned — Anomaly: %d (%.3f%%), Normal: %d (%.3f%%)",
        n_pos, 100 * n_pos / len(df), n_neg, 100 * n_neg / len(df),
    )
    return df


# ── Process a single patch file ─────────────────────────────────────────────

def _process_patch_file(
    model: TimesFMLiteGPT,
    patch_path: Path,
    metadata_path: Path,
    config: dict,
    device: str,
) -> pd.DataFrame:
    """Process one (patches.npy, metadata.parquet) pair into a distillation DataFrame."""
    logger.info("Processing %s ...", patch_path.name)

    metadata = pd.read_parquet(metadata_path)
    patches = np.load(patch_path)
    logger.info("  Loaded %d patches, %d metadata rows", patches.shape[0], len(metadata))

    all_series_ids = metadata["series_id"].unique()
    logger.info("  Processing %d unique series ...", len(all_series_ids))

    all_rows: List[dict] = []
    for i, sid in enumerate(all_series_ids):
        indices = metadata[metadata["series_id"] == sid]["patch_idx"].values
        series_patches = patches[indices]
        rows = _create_dataset_for_series(model, series_patches, sid, config, device)
        all_rows.extend(rows)

        if (i + 1) % 500 == 0:
            logger.info("    Processed %d/%d series (%d samples so far)",
                         i + 1, len(all_series_ids), len(all_rows))

    logger.info("  Finished — %d total samples from %s", len(all_rows), patch_path.name)
    df = pd.DataFrame(all_rows)
    return df


# ── Entry point ──────────────────────────────────────────────────────────────

def run_anomaly_data_pipeline() -> None:
    """End-to-end distillation data preparation pipeline."""
    pipeline_start = time.time()
    dcfg = settings.distillation
    output_dir = settings.storage.output_dir
    data_dir = dcfg.data_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 60)
    logger.info("Anomaly Detection — Data Preparation Pipeline")
    logger.info("Device: %s", device)
    logger.info("=" * 60)

    # Create output directory
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("Cannot create output directory '%s': %s", data_dir, e)
        return

    # Stage 1: Load teacher model
    logger.info("\n=== Stage 1: Loading teacher model ===")
    try:
        teacher_config = {
            "context_len": settings.training.context_len,
            "patch_len": settings.patches.patch_size,
        }
        model = _load_teacher_model(device)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # Discover patch files
    try:
        pairs = _discover_patch_metadata_pairs(output_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # Stage 2 & 3: Process each file, generate soft labels, assign hard labels
    for part_idx, (patch_path, meta_path) in enumerate(pairs, start=1):
        part_start = time.time()
        logger.info("\n=== Processing part %d/%d ===", part_idx, len(pairs))

        try:
            df = _process_patch_file(model, patch_path, meta_path, teacher_config, device)

            if df.empty:
                logger.warning("Part %d produced no samples — skipping.", part_idx)
                continue

            # Assign hard labels
            df = _assign_hard_labels(df)

            # Save as parquet
            out_path = data_dir / f"distillation_part_{part_idx}.parquet"
            df.to_parquet(out_path, index=False)
            logger.info("Saved %d rows → %s (%.1fs)",
                         len(df), out_path, time.time() - part_start)

        except Exception as e:
            logger.error("Failed to process part %d: %s — skipping.", part_idx, e, exc_info=True)
            continue

        del df
        gc.collect()

    elapsed = time.time() - pipeline_start
    logger.info("\n" + "=" * 60)
    logger.info("Data preparation complete (%.1fs). Outputs in: %s", elapsed, data_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    run_anomaly_data_pipeline()

"""
Model training and evaluation pipeline.

  Stage 1 — Data loading:   Discover .npy patch files, split train/val/test
  Stage 2 — Training:       AdamW + CosineAnnealingLR + AMP, per-epoch metrics
  Stage 3 — Evaluation:     Load best model, evaluate on test set, plot results

Run with:
    python -m pipelines.train_pipeline
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.settings import settings
from src.data.weather_dataset import WeatherPatchDataset
from src.pretrained_model.timesfm_lite import TimesFMLiteGPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _discover_patch_files(output_dir: Path) -> List[Path]:
    """Find all weather_part_*_patches.npy files across all year directories."""
    files = sorted(output_dir.rglob("weather_part_*_patches.npy"))
    if not files:
        raise FileNotFoundError(
            f"No patch .npy files found under {output_dir}. "
            "Run the data pipeline first: python -m pipelines.data_pipeline"
        )
    logger.info("Discovered %d patch file(s) under %s", len(files), output_dir)
    return files


def _split_files(
    all_files: List[Path],
    n_train: int,
    n_val: int,
    n_test: int,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split file list into train / val / test."""
    total_needed = n_train + n_val + n_test
    if len(all_files) < total_needed:
        raise ValueError(
            f"Need at least {total_needed} patch files for the configured split "
            f"(train={n_train}, val={n_val}, test={n_test}), "
            f"but only found {len(all_files)}."
        )

    train = all_files[:n_train]
    val = all_files[n_train : n_train + n_val]
    test = all_files[n_train + n_val : n_train + n_val + n_test]

    logger.info("Train files: %s", [f.name for f in train])
    logger.info("Val files:   %s", [f.name for f in val])
    logger.info("Test files:  %s", [f.name for f in test])
    return train, val, test


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Run evaluation and return averaged metrics."""
    model.eval()

    criterion_mse = nn.MSELoss(reduction="sum")
    criterion_mae = nn.L1Loss(reduction="sum")
    criterion_huber = nn.HuberLoss(reduction="sum")

    mse_sum = mae_sum = huber_sum = 0.0
    total_elements = 0

    try:
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    preds = model(x)

                mse_sum += criterion_mse(preds, y).item()
                mae_sum += criterion_mae(preds, y).item()
                huber_sum += criterion_huber(preds, y).item()
                total_elements += y.numel()
    except Exception as e:
        logger.error("Error during evaluation: %s", e, exc_info=True)
        raise

    if total_elements == 0:
        logger.warning("No elements evaluated — returning zero metrics.")
        return {"MSE": 0.0, "RMSE": 0.0, "MAE": 0.0, "Huber": 0.0}

    mse = mse_sum / total_elements
    mae = mae_sum / total_elements
    huber = huber_sum / total_elements
    rmse = math.sqrt(mse)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "Huber": huber}


def plot_predictions(
    preds: np.ndarray,
    targets: np.ndarray,
    filepath: Path,
) -> None:
    """Save a prediction-vs-actual comparison plot."""
    try:
        pred_seq = preds[0].flatten()
        target_seq = targets[0].flatten()

        plt.figure(figsize=(12, 6))
        plt.plot(target_seq, label="Actual", alpha=0.7)
        plt.plot(pred_seq, label="Predicted", alpha=0.7, linestyle="--")
        plt.title("TimesFM Weather Prediction (Sample Test Sequence)")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Prediction plot saved → %s", filepath)
    except Exception as e:
        logger.error("Failed to save prediction plot: %s", e, exc_info=True)


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    save_dir: Path,
) -> nn.Module:
    """
    Full training loop with:
      - AdamW optimizer + CosineAnnealingLR
      - Automatic Mixed Precision (AMP) on CUDA
      - Per-epoch train/val metrics (MSE, RMSE, MAE, Huber)
      - Best model saving + epoch checkpoints
      - Training metrics JSON + MSE curve plot
    """
    cfg = settings.training
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    criterion_huber = nn.HuberLoss()

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {
        "epochs": [],
        "train": {"MSE": [], "RMSE": [], "MAE": [], "Huber": []},
        "val": {"MSE": [], "RMSE": [], "MAE": [], "Huber": []},
    }

    best_val_loss = float("inf")
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting training — %d epoch(s), device=%s, AMP=%s", cfg.epochs, device, use_amp)
    logger.info("Model parameters: %s", f"{model.count_parameters():,}")
    logger.info("=" * 60)

    for epoch in range(cfg.epochs):
        epoch_start = time.time()

        # ── Train phase ──────────────────────────────────────────
        model.train()
        total_mse = total_mae = total_huber = 0.0
        total_elements = 0

        try:
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds = model(x)
                    mse = criterion_mse(preds, y)
                    mae = criterion_mae(preds, y)
                    huber = criterion_huber(preds, y)
                    loss = mse

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                n = y.numel()
                total_mse += mse.item() * n
                total_mae += mae.item() * n
                total_huber += huber.item() * n
                total_elements += n

                if (batch_idx + 1) % 500 == 0:
                    logger.info(
                        "  Epoch %d — batch %d/%d, loss=%.6f",
                        epoch + 1, batch_idx + 1, len(train_loader), mse.item(),
                    )

        except Exception as e:
            logger.error("Error in training epoch %d: %s", epoch + 1, e, exc_info=True)
            raise

        # ── Compute train metrics ────────────────────────────────
        train_mse = total_mse / total_elements
        train_mae = total_mae / total_elements
        train_huber = total_huber / total_elements
        train_rmse = math.sqrt(train_mse)

        # ── Validation phase ─────────────────────────────────────
        val_metrics = evaluate_model(model, val_loader, device)
        scheduler.step()

        # ── Logging ──────────────────────────────────────────────
        elapsed = time.time() - epoch_start
        logger.info("")
        logger.info("Epoch %d/%d  (%.1fs)", epoch + 1, cfg.epochs, elapsed)
        logger.info(
            "  Train → MSE: %.6f | RMSE: %.6f | MAE: %.6f | Huber: %.6f",
            train_mse, train_rmse, train_mae, train_huber,
        )
        logger.info(
            "  Val   → MSE: %.6f | RMSE: %.6f | MAE: %.6f | Huber: %.6f",
            val_metrics["MSE"], val_metrics["RMSE"],
            val_metrics["MAE"], val_metrics["Huber"],
        )

        # ── Record history ───────────────────────────────────────
        history["epochs"].append(epoch + 1)
        history["train"]["MSE"].append(train_mse)
        history["train"]["RMSE"].append(train_rmse)
        history["train"]["MAE"].append(train_mae)
        history["train"]["Huber"].append(train_huber)
        history["val"]["MSE"].append(val_metrics["MSE"])
        history["val"]["RMSE"].append(val_metrics["RMSE"])
        history["val"]["MAE"].append(val_metrics["MAE"])
        history["val"]["Huber"].append(val_metrics["Huber"])

        # ── Save best model ──────────────────────────────────────
        if val_metrics["MSE"] < best_val_loss:
            best_val_loss = val_metrics["MSE"]
            best_path = save_dir / "best_model.pth"
            torch.save(model.state_dict(), best_path)
            logger.info("  ★ Best model saved → %s (val MSE=%.6f)", best_path, best_val_loss)

        # ── Save epoch checkpoint ────────────────────────────────
        try:
            ckpt_path = save_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )
            logger.info("  Checkpoint saved → %s", ckpt_path)
        except Exception as e:
            logger.error("  Failed to save checkpoint: %s", e, exc_info=True)

        logger.info("-" * 60)

    # ── Save training metrics ────────────────────────────────────
    try:
        metrics_path = save_dir / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=4)
        logger.info("Training metrics saved → %s", metrics_path)
    except Exception as e:
        logger.error("Failed to save training metrics: %s", e, exc_info=True)

    # ── Plot MSE curve ───────────────────────────────────────────
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(history["epochs"], history["train"]["MSE"], label="Train MSE", marker="o")
        plt.plot(history["epochs"], history["val"]["MSE"], label="Val MSE", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Training & Validation MSE")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_dir / "mse_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("MSE curve saved → %s", save_dir / "mse_curve.png")
    except Exception as e:
        logger.error("Failed to save MSE curve: %s", e, exc_info=True)

    return model


# ── Test-set evaluation ──────────────────────────────────────────────────────

def run_test_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    save_dir: Path,
) -> None:
    """Evaluate the best model on the test set and save results."""
    logger.info("=== Test-set evaluation ===")

    try:
        metrics = evaluate_model(model, test_loader, device)
    except Exception as e:
        logger.error("Test evaluation failed: %s", e, exc_info=True)
        return

    logger.info("TEST RESULTS:")
    for k, v in metrics.items():
        logger.info("  %s: %.6f", k, v)

    # Save metrics
    try:
        results_path = save_dir / "evaluation_metrics.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("Evaluation metrics saved → %s", results_path)
    except Exception as e:
        logger.error("Failed to save evaluation metrics: %s", e, exc_info=True)

    # Generate prediction plot from first batch
    try:
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    preds = model(x)
                plot_predictions(
                    preds.cpu().numpy(),
                    y.cpu().numpy(),
                    save_dir / "prediction_comparison.png",
                )
                break  # only need first batch
    except Exception as e:
        logger.error("Failed to generate prediction plot: %s", e, exc_info=True)


# ── Entry point ──────────────────────────────────────────────────────────────

def run_training_pipeline() -> None:
    """End-to-end training pipeline: data loading → training → evaluation."""
    cfg = settings.training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    output_dir = settings.storage.output_dir
    save_dir = cfg.model_output_dir

    # ── Stage 1: Discover and split data ─────────────────────────
    try:
        all_files = _discover_patch_files(output_dir)
        train_files, val_files, test_files = _split_files(
            all_files, cfg.train_files, cfg.val_files, cfg.test_files,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("Data loading failed: %s", e)
        return

    # ── Build datasets & loaders ─────────────────────────────────
    try:
        train_ds = WeatherPatchDataset(train_files, context_len=cfg.context_len)
        val_ds = WeatherPatchDataset(val_files, context_len=cfg.context_len)
        test_ds = WeatherPatchDataset(test_files, context_len=cfg.context_len)

        if len(train_ds) == 0:
            logger.error("Training dataset is empty. Check your patch files.")
            return

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True,
        )

        logger.info(
            "Datasets ready — train=%d, val=%d, test=%d samples",
            len(train_ds), len(val_ds), len(test_ds),
        )
    except Exception as e:
        logger.error("Failed to create datasets: %s", e, exc_info=True)
        return

    # ── Stage 2: Train ───────────────────────────────────────────
    try:
        model_config = cfg.to_model_config(settings.patches.patch_size)
        model = TimesFMLiteGPT(model_config)
        model = train_model(model, train_loader, val_loader, device, save_dir)
    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        return

    # ── Stage 3: Evaluate on test set ────────────────────────────
    try:
        best_path = save_dir / "best_model.pth"
        if best_path.exists():
            model.load_state_dict(
                torch.load(best_path, map_location=device, weights_only=True)
            )
            logger.info("Loaded best model for test evaluation.")
        else:
            logger.warning("No best_model.pth found — evaluating last epoch model.")

        run_test_evaluation(model, test_loader, device, save_dir)
    except Exception as e:
        logger.error("Test evaluation failed: %s", e, exc_info=True)

    logger.info("Training pipeline complete. Outputs in: %s", save_dir)


if __name__ == "__main__":
    run_training_pipeline()

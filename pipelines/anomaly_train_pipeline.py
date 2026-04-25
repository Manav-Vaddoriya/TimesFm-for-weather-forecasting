"""
Anomaly-detection MLP training pipeline (knowledge distillation).

Trains the student MLPStudentModel on distillation parquet data
produced by anomaly_data_pipeline.py.

  Stage 1 — Data loading:  Discover distillation parquet files, split by part index
  Stage 2 — Training:      AdamW + CosineAnnealingLR, Huber loss on soft labels
  Stage 3 — Evaluation:    Load best model, evaluate on test set, save results

Run with:
    python -m pipelines.anomaly_train_pipeline
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config.settings import settings
from webapp.models import MLPStudentModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Dataset ──────────────────────────────────────────────────────────────────

class DistillationDataset(Dataset):
    """
    PyTorch Dataset for the distillation parquet files.
    Each sample: (context_window, soft_label, hard_label).
    """

    def __init__(self, df: pd.DataFrame):
        self.X = torch.tensor(
            np.stack(df["context_window"].values),
            dtype=torch.float32,
        )
        self.y_soft = torch.tensor(df["soft_label"].values, dtype=torch.float32)
        self.y_hard = torch.tensor(df["hard_label"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y_soft[idx], self.y_hard[idx]


def _load_parquet_files(data_dir: Path, part_indices: List[int]) -> pd.DataFrame:
    """Load and concatenate distillation parquet files by part index."""
    dfs = []
    for i in part_indices:
        path = data_dir / f"distillation_part_{i}.parquet"
        if not path.exists():
            logger.warning("  File %s not found — skipping.", path)
            continue
        df = pd.read_parquet(path)
        logger.info("  Loaded part_%d: %s rows", i, f"{len(df):>10,}")
        dfs.append(df)

    if not dfs:
        raise ValueError("No distillation parquet files loaded. Check data directory.")

    combined = pd.concat(dfs, ignore_index=True)
    n_pos = int((combined["hard_label"] == 1).sum())
    n_neg = int((combined["hard_label"] == 0).sum())
    logger.info("  %s", "─" * 38)
    logger.info("  Total    : %s rows", f"{len(combined):>10,}")
    logger.info("  Anomaly  : %s  (%.3f%%)", f"{n_pos:>10,}", 100 * n_pos / len(combined))
    logger.info("  Normal   : %s  (%.3f%%)", f"{n_neg:>10,}", 100 * n_neg / len(combined))
    return combined


# ── Loss ─────────────────────────────────────────────────────────────────────

class CombinedDistillationLoss(nn.Module):
    """
    L_total = Huber(student_prob, soft_label)

    Only soft distillation loss — hard label component removed
    (matching the notebook's final version).
    """

    def __init__(self, huber_delta: float = 1.0):
        super().__init__()
        self.huber_delta = huber_delta

    def forward(self, probs, y_soft, y_hard=None):
        l_soft = F.huber_loss(probs, y_soft, delta=self.huber_delta)
        return l_soft, l_soft.item(), 0.0  # placeholder for hard loss


# ── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: CombinedDistillationLoss,
    device: str,
) -> Dict[str, float]:
    """Evaluate model, return {loss: ...}."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y_soft, y_hard in loader:
            X, y_soft, y_hard = X.to(device), y_soft.to(device), y_hard.to(device)
            probs = model(X)
            loss, _, _ = criterion(probs, y_soft, y_hard)
            total_loss += loss.item()

    return {"loss": round(total_loss / max(len(loader), 1), 5)}


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    save_dir: Path,
) -> nn.Module:
    """
    Full training loop:
      - AdamW optimizer + CosineAnnealingLR
      - Huber loss on soft labels
      - Gradient clipping (max_norm=1.0)
      - Best model saving + training history CSV + loss curve plot
    """
    cfg = settings.distillation
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min,
    )
    criterion = CombinedDistillationLoss(huber_delta=cfg.huber_delta)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Student MLP parameters : %s", f"{total_params:,}")
    logger.info(
        "Architecture           : %d → %s → 1",
        cfg.window_size, " → ".join(str(d) for d in cfg.hidden_dims),
    )

    best_val_loss = float("inf")
    save_dir.mkdir(parents=True, exist_ok=True)
    history = []

    logger.info("\n" + "=" * 55)
    logger.info("TRAINING")
    logger.info("=" * 55)

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = train_soft = train_hard = 0.0

        for X, y_soft, y_hard in train_loader:
            X, y_soft, y_hard = X.to(device), y_soft.to(device), y_hard.to(device)
            optimizer.zero_grad()
            probs = model(X)
            loss, ls, lh = criterion(probs, y_soft, y_hard)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_soft += ls
            train_hard += lh

        scheduler.step()

        nb = len(train_loader)
        avg_loss = train_loss / nb
        avg_soft = train_soft / nb
        avg_hard = train_hard / nb

        val_metrics = evaluate(model, val_loader, criterion, device)

        # Save best checkpoint
        improved = ""
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), save_dir / "best_student_mlp.pt")
            improved = "  ✓ saved"

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 5),
            "soft_loss": round(avg_soft, 5),
            "hard_loss": round(avg_hard, 5),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        elapsed = time.time() - epoch_start
        logger.info(
            "Epoch [%02d/%d] (%.1fs) train_loss=%.4f "
            "(soft=%.4f, hard=%.4f) | val_loss=%.4f %s",
            epoch, cfg.epochs, elapsed, avg_loss,
            avg_soft, avg_hard, val_metrics["loss"], improved,
        )

    # Save training history
    history_df = pd.DataFrame(history)
    history_path = save_dir / "training_history_mlp.csv"
    history_df.to_csv(history_path, index=False)
    logger.info("Training history saved → %s", history_path)

    # Plot loss curve
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", marker="o")
        plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Distillation Training — Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_dir / "distillation_loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Loss curve saved → %s", save_dir / "distillation_loss_curve.png")
    except Exception as e:
        logger.error("Failed to save loss curve: %s", e, exc_info=True)

    return model


# ── Test evaluation ──────────────────────────────────────────────────────────

def run_test_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    save_dir: Path,
) -> None:
    """Evaluate the best model on the test set and save results."""
    cfg = settings.distillation
    criterion = CombinedDistillationLoss(huber_delta=cfg.huber_delta)

    logger.info("\n" + "=" * 55)
    logger.info("FINAL EVALUATION  (best checkpoint)")
    logger.info("=" * 55)

    metrics = evaluate(model, test_loader, criterion, device)

    logger.info("  %s", "─" * 45)
    logger.info("  TEST RESULTS")
    logger.info("  %s", "─" * 45)
    for k, v in metrics.items():
        logger.info("    %-12s: %s", k, v)
    logger.info("  %s", "─" * 45)

    # Save metrics
    try:
        results_path = save_dir / "evaluation_metrics.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("Evaluation metrics saved → %s", results_path)
    except Exception as e:
        logger.error("Failed to save evaluation metrics: %s", e, exc_info=True)

    # Probability distribution analysis on test set
    try:
        model.eval()
        all_probs = []
        with torch.no_grad():
            for X, y_soft, y_hard in test_loader:
                probs = model(X.to(device))
                all_probs.extend(probs.cpu().numpy())

        all_probs = np.array(all_probs)
        threshold = cfg.threshold

        logger.info("\n  Probability distribution:")
        logger.info("    Min    : %.4f", all_probs.min())
        logger.info("    Mean   : %.4f", all_probs.mean())
        logger.info("    Median : %.4f", np.median(all_probs))
        logger.info("    P95    : %.4f", np.percentile(all_probs, 95))
        logger.info("    Max    : %.4f", all_probs.max())

        preds = (all_probs >= threshold).astype(int)
        n_anomaly = preds.sum()
        n_normal = len(preds) - n_anomaly
        logger.info("  Segmentation (threshold=%.4f):", threshold)
        logger.info("    Anomaly: %s  (%.3f%%)", f"{n_anomaly:>8,}", 100 * n_anomaly / len(preds))
        logger.info("    Normal : %s  (%.3f%%)", f"{n_normal:>8,}", 100 * n_normal / len(preds))

    except Exception as e:
        logger.error("Failed probability analysis: %s", e, exc_info=True)


# ── Entry point ──────────────────────────────────────────────────────────────

def run_distillation_pipeline() -> None:
    """End-to-end distillation training pipeline: data loading → training → evaluation."""
    cfg = settings.distillation
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 60)
    logger.info("Anomaly Detection — Distillation Training Pipeline")
    logger.info("Device: %s", device)
    logger.info("=" * 60)

    data_dir = cfg.data_dir
    save_dir = cfg.model_output_dir

    # ── Stage 1: Load data ──────────────────────────────────────
    logger.info("\n=== Stage 1: Loading data ===")

    try:
        logger.info("\n[TRAIN] parts %s", cfg.train_parts)
        train_df = _load_parquet_files(data_dir, cfg.train_parts)

        logger.info("\n[VAL]   parts %s", cfg.val_parts)
        val_df = _load_parquet_files(data_dir, cfg.val_parts)

        logger.info("\n[TEST]  parts %s", cfg.test_parts)
        test_df = _load_parquet_files(data_dir, cfg.test_parts)
    except ValueError as e:
        logger.error("Data loading failed: %s", e)
        return

    # Build datasets & loaders
    try:
        train_loader = DataLoader(
            DistillationDataset(train_df),
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            DistillationDataset(val_df),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True,
        )
        test_loader = DataLoader(
            DistillationDataset(test_df),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True,
        )

        logger.info(
            "Datasets ready — train=%d, val=%d, test=%d samples",
            len(train_df), len(val_df), len(test_df),
        )
    except Exception as e:
        logger.error("Failed to create datasets: %s", e, exc_info=True)
        return

    # Free DataFrame memory
    del train_df, val_df, test_df
    gc.collect()

    # ── Stage 2: Train ──────────────────────────────────────────
    try:
        model = MLPStudentModel(
            window_size=cfg.window_size,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
        )
        model = train_model(model, train_loader, val_loader, device, save_dir)
    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        return

    # ── Stage 3: Evaluate on test set ───────────────────────────
    try:
        best_path = save_dir / "best_student_mlp.pt"
        if best_path.exists():
            model.load_state_dict(
                torch.load(best_path, map_location=device, weights_only=True)
            )
            logger.info("Loaded best model for test evaluation.")
        else:
            logger.warning("No best_student_mlp.pt found — evaluating last epoch model.")

        run_test_evaluation(model, test_loader, device, save_dir)
    except Exception as e:
        logger.error("Test evaluation failed: %s", e, exc_info=True)

    logger.info("\nDistillation pipeline complete. Outputs in: %s", save_dir)


if __name__ == "__main__":
    run_distillation_pipeline()

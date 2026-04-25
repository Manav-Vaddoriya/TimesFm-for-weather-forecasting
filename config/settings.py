"""
Loads pipeline_config.yaml into typed dataclasses.
All modules import `settings` from here — never read YAML directly in
business logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

_CONFIG_PATH = Path(__file__).parent / "pipeline_config.yaml"


@dataclass(frozen=True)
class PatchConfig:
    patch_size: int
    pad_value: float


@dataclass(frozen=True)
class StorageConfig:
    output_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for TimesFMLiteGPT training."""
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    context_len: int
    batch_size: int
    lr: float
    weight_decay: float
    epochs: int
    num_workers: int
    train_files: int
    val_files: int
    test_files: int
    model_output_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_output_dir", Path(self.model_output_dir))

    def to_model_config(self, patch_len: int) -> dict:
        """Build the config dict expected by TimesFMLiteGPT."""
        return {
            "patch_len": patch_len,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "context_len": self.context_len,
        }


@dataclass(frozen=True)
class DistillationConfig:
    """Hyperparameters for anomaly-detection knowledge distillation."""
    data_dir: Path
    train_parts: List[int]
    val_parts: List[int]
    test_parts: List[int]
    window_size: int
    hidden_dims: List[int]
    dropout: float
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    huber_delta: float
    lr_min: float
    threshold: float
    num_workers: int
    model_output_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "model_output_dir", Path(self.model_output_dir))


@dataclass(frozen=True)
class PipelineSettings:
    grib_file: Path
    years: List[int]
    raw_chunks_per_year: int
    series_id_batch_size: int
    temp_dtype: str
    coord_dtype: str
    coord_round_decimals: int
    patches: PatchConfig
    storage: StorageConfig
    training: TrainingConfig
    distillation: DistillationConfig


def load_settings(config_path: Path = _CONFIG_PATH) -> PipelineSettings:
    """Parse YAML and return a validated PipelineSettings instance."""
    with open(config_path) as fh:
        raw = yaml.safe_load(fh)

    return PipelineSettings(
        grib_file=Path(raw["pipeline"]["grib_file"]),
        years=raw["pipeline"]["years"],
        raw_chunks_per_year=raw["data"]["raw_chunks_per_year"],
        series_id_batch_size=raw["data"]["series_id_batch_size"],
        temp_dtype=raw["temperature"]["dtype"],
        coord_dtype=raw["coordinates"]["dtype"],
        coord_round_decimals=raw["coordinates"]["round_decimals"],
        patches=PatchConfig(**raw["patches"]),
        storage=StorageConfig(
            output_dir=raw["storage"]["output_dir"],
        ),
        training=TrainingConfig(**raw["training"]),
        distillation=DistillationConfig(**raw["distillation"]),
    )


# Module-level singleton — import this everywhere
settings: PipelineSettings = load_settings()
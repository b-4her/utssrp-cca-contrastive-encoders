"""Interpretability probes for deterministic relation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .data import PairedDataset
from .training import TrainingArtifacts


@dataclass
class RidgeProbe:
    """Closed-form ridge probe from learned embeddings to clean targets."""

    z_mean: np.ndarray
    target_mean: np.ndarray
    weights: np.ndarray
    alpha: float


def fit_ridge_probe(
    z_train: np.ndarray,
    target_train: np.ndarray,
    alpha: float = 1e-3,
) -> RidgeProbe:
    """
    Fit a ridge probe with an unpenalized intercept.

    This mirrors the notebook's R^2 probe logic, but returns the fitted weights
    so we can evaluate a swept input curve.
    """
    if alpha < 0:
        raise ValueError("alpha must be nonnegative.")

    z_train = np.asarray(z_train)
    target_train = np.asarray(target_train)

    z_mean = z_train.mean(axis=0, keepdims=True)
    target_mean = target_train.mean(axis=0, keepdims=True)

    z_centered = z_train - z_mean
    target_centered = target_train - target_mean

    regularizer = alpha * np.eye(z_centered.shape[1])
    weights = np.linalg.solve(
        z_centered.T @ z_centered + regularizer,
        z_centered.T @ target_centered,
    )

    return RidgeProbe(
        z_mean=z_mean,
        target_mean=target_mean,
        weights=weights,
        alpha=alpha,
    )


def predict_ridge_probe(probe: RidgeProbe, z: np.ndarray) -> np.ndarray:
    """Predict clean targets from embeddings using a fitted ridge probe."""
    z = np.asarray(z)
    return (z - probe.z_mean) @ probe.weights + probe.target_mean


def _standardize_like_training(
    X: np.ndarray,
    X_train: np.ndarray,
) -> np.ndarray:
    """Apply the same train-only X standardization used before model training."""
    x_mean = X_train.mean(axis=0, keepdims=True)
    x_scale = X_train.std(axis=0, ddof=1, keepdims=True)
    x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
    return (X - x_mean) / x_scale


@torch.no_grad()
def _encode_x_numpy(
    model: torch.nn.Module,
    X_standardized: np.ndarray,
    device: str,
) -> np.ndarray:
    model.eval()
    X_tensor = torch.tensor(X_standardized, dtype=torch.float32, device=device)
    return model.encoder_x(X_tensor).detach().cpu().numpy()


def cubic_coordinate_probe_curve(
    run: TrainingArtifacts,
    raw_dataset: PairedDataset,
    observation_index: int,
    coordinate: int = 0,
    sweep_values: Sequence[float] | None = None,
    device: str = "cpu",
    probe_alpha: float = 1e-3,
) -> pd.DataFrame:
    """
    Sweep one raw X coordinate and compare probed predictions to x_j^3.

    The model is evaluated on standardized inputs because that is how it was
    trained. The probe target remains the raw clean deterministic target stored
    in raw_dataset.Z_y_train.
    """
    if raw_dataset.Z_y_train is None:
        raise ValueError("raw_dataset.Z_y_train must contain clean cubic targets.")
    if coordinate < 0 or coordinate >= raw_dataset.Y_train.shape[1]:
        raise ValueError("coordinate must index a Y coordinate.")
    if observation_index < 0 or observation_index >= raw_dataset.X_test.shape[0]:
        raise ValueError("observation_index must index a held-out observation.")

    sweep = (
        np.linspace(-3.0, 3.0, 121)
        if sweep_values is None
        else np.asarray(sweep_values, dtype=float)
    )

    probe = fit_ridge_probe(
        z_train=run.z_x_train,
        target_train=raw_dataset.Z_y_train,
        alpha=probe_alpha,
    )

    X_sweep = np.repeat(raw_dataset.X_test[[observation_index]], len(sweep), axis=0)
    X_sweep[:, coordinate] = sweep
    X_sweep_standardized = _standardize_like_training(X_sweep, raw_dataset.X_train)

    z_sweep = _encode_x_numpy(run.model, X_sweep_standardized, device=device)
    y_pred = predict_ridge_probe(probe, z_sweep)

    return pd.DataFrame(
        {
            "observation": observation_index,
            "coordinate": coordinate,
            "x_value": sweep,
            "true_y": sweep**3,
            "probed_y": y_pred[:, coordinate],
        }
    )

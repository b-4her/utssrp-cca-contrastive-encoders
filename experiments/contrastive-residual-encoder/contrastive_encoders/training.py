"""Training loop and diagnostics for one model/dataset pair.

This module is the bridge between the architectures, paper loss, penalties, and
metrics. It trains one configuration and returns a flat dictionary of results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from .architectures import build_model, count_parameters
from .data import PairedDataset, StandardizedDataset, standardize_train_test
from .losses import paper_contrastive_loss
from .metrics import (
    alignment_gap,
    columnwise_correlations,
    latent_probe_r2,
    latent_recovery_scalar,
    to_numpy,
    top1_retrieval_accuracy,
    topk_retrieval_accuracy,
)
from .regularization import (
    l1_penalty_model,
    l2_penalty_model,
    nonlinear_output_penalty,
)


@dataclass
class TrainConfig:
    name: str
    architecture: str = "residual"

    embedding_dim: int = 4
    hidden_dim: int = 16
    nonlinear_scale: float = 0.1
    normalize_linear_part: bool = True

    epochs: int = 300
    learning_rate: float = 1e-3

    normalize_embeddings_in_loss: bool = False
    temperature: float = 1.0

    l1_linear: float = 0.0
    l1_nonlinear: float = 0.0
    l2_linear: float = 0.0
    l2_nonlinear: float = 0.0

    nonlinear_output_strength: float = 0.0


@dataclass
class TrainingArtifacts:
    """
    Returned by the diagnostic training path.

    The regular experiment table only needs final metrics, but plots such as
    branch-ratio-over-training and similarity heatmaps need the trained model,
    standardized data, saved embeddings, and per-epoch history.
    """

    model: torch.nn.Module
    data: StandardizedDataset
    metrics: Dict[str, float]
    history: List[Dict[str, float]]
    z_x_train: np.ndarray
    z_y_train: np.ndarray
    z_x_test: np.ndarray
    z_y_test: np.ndarray


# Branch-size diagnostic for checking whether alpha is meaningful
@torch.no_grad()
def branch_size_metrics(
    model: torch.nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
) -> Dict[str, float]:
    """
    Measure how large the nonlinear branch is relative to the linear branch.

    For linear-only models these diagnostics are not defined, so the values are
    NaN. For normalized residual models, the ratios should move systematically
    with alpha.
    """
    if not hasattr(model.encoder_x, "components"):
        return {
            "x_linear_branch_norm": float("nan"),
            "x_nonlinear_branch_norm": float("nan"),
            "x_nonlinear_to_linear_ratio": float("nan"),
            "y_linear_branch_norm": float("nan"),
            "y_nonlinear_branch_norm": float("nan"),
            "y_nonlinear_to_linear_ratio": float("nan"),
            "mean_nonlinear_to_linear_ratio": float("nan"),
        }

    linear_x, nonlinear_x = model.encoder_x.components(X)
    linear_y, nonlinear_y = model.encoder_y.components(Y)

    x_linear_norm = float(linear_x.norm(dim=1).mean().detach().cpu())
    x_nonlinear_norm = float(nonlinear_x.norm(dim=1).mean().detach().cpu())
    y_linear_norm = float(linear_y.norm(dim=1).mean().detach().cpu())
    y_nonlinear_norm = float(nonlinear_y.norm(dim=1).mean().detach().cpu())

    x_ratio = x_nonlinear_norm / max(x_linear_norm, 1e-12)
    y_ratio = y_nonlinear_norm / max(y_linear_norm, 1e-12)

    return {
        "x_linear_branch_norm": x_linear_norm,
        "x_nonlinear_branch_norm": x_nonlinear_norm,
        "x_nonlinear_to_linear_ratio": x_ratio,
        "y_linear_branch_norm": y_linear_norm,
        "y_nonlinear_branch_norm": y_nonlinear_norm,
        "y_nonlinear_to_linear_ratio": y_ratio,
        "mean_nonlinear_to_linear_ratio": 0.5 * (x_ratio + y_ratio),
    }


def _train_and_evaluate(
    dataset: PairedDataset,
    config: TrainConfig,
    seed: int,
    device: str,
    standardize: bool = True,
    collect_history: bool = False,
    history_interval: int = 10,
) -> TrainingArtifacts:
    """
    Shared training implementation for final metrics and richer diagnostics.

    Both architecture options, linear and residual, enter the same paper-loss
    path. When collect_history is True, the function records branch norms on the
    held-out data during training.
    """
    if history_interval <= 0:
        raise ValueError("history_interval must be positive.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    data = standardize_train_test(dataset, standardize=standardize)

    X_train = torch.tensor(data.X_train, dtype=torch.float32, device=device)
    Y_train = torch.tensor(data.Y_train, dtype=torch.float32, device=device)
    X_test = torch.tensor(data.X_test, dtype=torch.float32, device=device)
    Y_test = torch.tensor(data.Y_test, dtype=torch.float32, device=device)

    model = build_model(
        architecture=config.architecture,
        p=X_train.shape[1],
        q=Y_train.shape[1],
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        nonlinear_scale=config.nonlinear_scale,
        normalize_linear_part=config.normalize_linear_part,
    ).to(device)

    parameter_count = count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    final_loss = float("nan")
    history: List[Dict[str, float]] = []

    def append_history(epoch: int, loss_value: float) -> None:
        model.eval()
        history.append(
            {
                "epoch": epoch,
                "loss": loss_value,
                **branch_size_metrics(model, X_test, Y_test),
            }
        )

    if collect_history:
        append_history(epoch=0, loss_value=final_loss)

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        z_x, z_y = model(X_train, Y_train)

        loss = paper_contrastive_loss(
            z_x,
            z_y,
            normalize_embeddings=config.normalize_embeddings_in_loss,
            temperature=config.temperature,
        )
        loss = loss + l1_penalty_model(
            model,
            l1_linear=config.l1_linear,
            l1_nonlinear=config.l1_nonlinear,
        )
        loss = loss + l2_penalty_model(
            model,
            l2_linear=config.l2_linear,
            l2_nonlinear=config.l2_nonlinear,
        )
        loss = loss + nonlinear_output_penalty(
            model,
            X_train,
            Y_train,
            strength=config.nonlinear_output_strength,
        )

        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

        if collect_history and (
            epoch == config.epochs or epoch % history_interval == 0
        ):
            append_history(epoch=epoch, loss_value=final_loss)

    model.eval()
    with torch.no_grad():
        z_x_train, z_y_train = model(X_train, Y_train)
        z_x_test, z_y_test = model(X_test, Y_test)

    z_x_train_np = to_numpy(z_x_train)
    z_y_train_np = to_numpy(z_y_train)
    z_x_test_np = to_numpy(z_x_test)
    z_y_test_np = to_numpy(z_y_test)

    train_corrs = columnwise_correlations(z_x_train_np, z_y_train_np)
    test_corrs = columnwise_correlations(z_x_test_np, z_y_test_np)

    shuffled_idx = np.random.default_rng(seed + 999).permutation(z_y_test_np.shape[0])
    shuffled_gap = alignment_gap(z_x_test_np, z_y_test_np[shuffled_idx])
    branch_metrics = branch_size_metrics(model, X_test, Y_test)

    x_recovery_of_z_x = latent_recovery_scalar(z_x_test_np, data.Z_x_test)
    x_recovery_of_z_y = latent_recovery_scalar(z_x_test_np, data.Z_y_test)
    y_recovery_of_z_y = latent_recovery_scalar(z_y_test_np, data.Z_y_test)

    x_probe_r2_z_x = latent_probe_r2(
        z_x_train_np,
        data.Z_x_train,
        z_x_test_np,
        data.Z_x_test,
    )
    x_probe_r2_z_y = latent_probe_r2(
        z_x_train_np,
        data.Z_y_train,
        z_x_test_np,
        data.Z_y_test,
    )
    y_probe_r2_z_y = latent_probe_r2(
        z_y_train_np,
        data.Z_y_train,
        z_y_test_np,
        data.Z_y_test,
    )

    final_metrics = {
        "final_loss": final_loss,
        "parameter_count": parameter_count,
        "parameter_count_per_train_sample": parameter_count / X_train.shape[0],
        "train_best_view_correlation": float(np.nanmax(np.abs(train_corrs))),
        "test_best_view_correlation": float(np.nanmax(np.abs(test_corrs))),
        "train_pair_separation": alignment_gap(z_x_train_np, z_y_train_np),
        "test_pair_separation": alignment_gap(z_x_test_np, z_y_test_np),
        "shuffled_pair_separation": shuffled_gap,
        "train_pair_match_accuracy": top1_retrieval_accuracy(
            z_x_train_np,
            z_y_train_np,
        ),
        "test_pair_match_accuracy": top1_retrieval_accuracy(
            z_x_test_np,
            z_y_test_np,
        ),
        "train_top5_pair_match_accuracy": topk_retrieval_accuracy(
            z_x_train_np,
            z_y_train_np,
            k=5,
        ),
        "test_top5_pair_match_accuracy": topk_retrieval_accuracy(
            z_x_test_np,
            z_y_test_np,
            k=5,
        ),
        "x_signal_recovery": x_recovery_of_z_x,
        "x_related_signal_recovery": x_recovery_of_z_y,
        "y_signal_recovery": y_recovery_of_z_y,
        "x_probe_r2_z_x": x_probe_r2_z_x,
        "x_probe_r2_z_y": x_probe_r2_z_y,
        "y_probe_r2_z_y": y_probe_r2_z_y,
        **branch_metrics,
    }

    return TrainingArtifacts(
        model=model,
        data=data,
        metrics=final_metrics,
        history=history,
        z_x_train=z_x_train_np,
        z_y_train=z_y_train_np,
        z_x_test=z_x_test_np,
        z_y_test=z_y_test_np,
    )


# Train and evaluate one model on one paired dataset
def train_one_model(
    dataset: PairedDataset,
    config: TrainConfig,
    seed: int,
    device: str,
    standardize: bool = True,
) -> Dict[str, float]:
    """
    Train one contrastive model using the exact paper loss.

    This lightweight path returns only final metrics for tables and aggregate
    plots. Use train_one_model_with_artifacts when a plot needs history or
    embeddings.
    """
    return _train_and_evaluate(
        dataset=dataset,
        config=config,
        seed=seed,
        device=device,
        standardize=standardize,
    ).metrics


# Diagnostic path for history curves and heatmaps
def train_one_model_with_artifacts(
    dataset: PairedDataset,
    config: TrainConfig,
    seed: int,
    device: str,
    standardize: bool = True,
    history_interval: int = 10,
) -> TrainingArtifacts:
    """
    Train one model and keep the extra artifacts needed for richer plots.

    This is intentionally separate from train_one_model so the main experiment
    DataFrame stays small and clean.
    """
    return _train_and_evaluate(
        dataset=dataset,
        config=config,
        seed=seed,
        device=device,
        standardize=standardize,
        collect_history=True,
        history_interval=history_interval,
    )

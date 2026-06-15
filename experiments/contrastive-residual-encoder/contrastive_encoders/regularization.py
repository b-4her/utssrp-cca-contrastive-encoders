"""Regularization penalties used during model training.

The penalties are kept separate from the paper contrastive loss so the notebook
can compare architecture effects and regularization effects cleanly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Weight penalties
def l1_penalty_model(
    model: nn.Module,
    l1_linear: float = 0.0,
    l1_nonlinear: float = 0.0,
) -> torch.Tensor:
    """
    L1 penalty.

    l1_linear penalizes G_X and G_Y.
    l1_nonlinear penalizes A2 in both residual encoders.
    For a pure linear model, l1_nonlinear contributes zero.

    With normalized nonlinear branches, shrinking A1 does not reliably shrink
    the branch output. A2 is input-facing, so L1 on A2 is the clearer feature
    selection penalty.
    """
    device = next(model.parameters()).device
    penalty = torch.tensor(0.0, device=device)

    if l1_linear > 0:
        penalty = penalty + l1_linear * (
            model.encoder_x.linear.weight.abs().sum()
            + model.encoder_y.linear.weight.abs().sum()
        )

    if l1_nonlinear > 0 and hasattr(model.encoder_x, "nonlinear_weights"):
        _, A2x = model.encoder_x.nonlinear_weights()
        _, A2y = model.encoder_y.nonlinear_weights()
        penalty = penalty + l1_nonlinear * (
            A2x.abs().sum()
            + A2y.abs().sum()
        )

    return penalty


def l2_penalty_model(
    model: nn.Module,
    l2_linear: float = 0.0,
    l2_nonlinear: float = 0.0,
) -> torch.Tensor:
    """
    L2 penalty on linear and nonlinear parts.

    For a pure linear model, l2_nonlinear contributes zero.
    """
    device = next(model.parameters()).device
    penalty = torch.tensor(0.0, device=device)

    if l2_linear > 0:
        penalty = penalty + l2_linear * (
            model.encoder_x.linear.weight.pow(2).sum()
            + model.encoder_y.linear.weight.pow(2).sum()
        )

    if l2_nonlinear > 0 and hasattr(model.encoder_x, "nonlinear_weights"):
        A1x, A2x = model.encoder_x.nonlinear_weights()
        A1y, A2y = model.encoder_y.nonlinear_weights()
        penalty = penalty + l2_nonlinear * (
            A1x.pow(2).sum()
            + A2x.pow(2).sum()
            + A1y.pow(2).sum()
            + A2y.pow(2).sum()
        )

    return penalty


# Output penalty on the actual nonlinear correction
def nonlinear_output_penalty(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    strength: float = 0.0,
) -> torch.Tensor:
    """
    Penalize the actual nonlinear correction output on the current batch.

    For a pure linear model, this contributes zero.
    """
    if strength <= 0 or not hasattr(model.encoder_x, "nonlinear_correction"):
        return torch.tensor(0.0, device=X.device)

    cx = model.encoder_x.nonlinear_correction(X)
    cy = model.encoder_y.nonlinear_correction(Y)
    return strength * (cx.pow(2).mean() + cy.pow(2).mean())

    # This function penalizes the magnitude of the nonlinear adjustments. 
    # It acts as a regularizer that keeps your model clean and interpretable, 
    # ensuring that the nonlinear sub-networks only wake up and contribute 
    # when the underlying data relationship is genuinely non-linear.
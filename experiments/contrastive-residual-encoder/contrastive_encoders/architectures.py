"""Neural network architectures used by the two-view contrastive experiments.

This module defines the pure linear baseline and the residual nonlinear encoder.
The encoders use BatchNorm1d affine=False to control output scale. In the
residual encoder, this makes alpha meaningfully control the nonlinear correction
size relative to the normalized linear branch.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class LinearEncoder(nn.Module):
    """
    Purely linear encoder:

        g(u) = Normalize(G u)
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        normalize_output: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, embedding_dim, bias=False)
        self.output_norm = (
            nn.BatchNorm1d(embedding_dim, affine=False)
            if normalize_output
            else nn.Identity()
        )

        # Start with non-redundant projection directions. This gives the linear
        # baseline a well-conditioned starting point before training updates it.
        nn.init.orthogonal_(self.linear.weight)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.linear(u))

    def linear_weight(self) -> torch.Tensor:
        return self.linear.weight


# Residual encoder with controlled nonlinear branch size
class ResidualNonlinearEncoder(nn.Module):
    """
    (The recommended architecture in our meeting)

    Encoder of the form:

        g(u) = Normalize(G u) + alpha * Normalize(A1 sigma(A2 u + b))

    affine=False in the normalization layers matters: there are no trainable
    scale parameters that can secretly undo alpha.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        nonlinear_scale: float = 0.1,
        activation: str = "gelu",
        normalize_linear_part: bool = True,
    ):
        super().__init__()

        self.linear = nn.Linear(input_dim, embedding_dim, bias=False)
        self.A2 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.A1 = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.nonlinear_scale = nonlinear_scale
        self.normalize_linear_part = normalize_linear_part

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("activation must be 'relu', 'tanh', or 'gelu'.")

        self.linear_norm = (
            nn.BatchNorm1d(embedding_dim, affine=False)
            if normalize_linear_part
            else nn.Identity()
        )
        self.nonlinear_norm = nn.BatchNorm1d(embedding_dim, affine=False)

        # The linear backbone starts as a stable set of non-redundant projection
        # directions, so the model begins with a sensible linear encoder.
        nn.init.orthogonal_(self.linear.weight)

        # A2 is the input-to-hidden layer of the MLP branch. Kaiming uniform keeps
        # the hidden activations from being too large or too small at startup.
        nn.init.kaiming_uniform_(self.A2.weight, a=np.sqrt(5))

        # A zero bias means the nonlinear branch does not start with a preferred
        # offset; it initially responds to the input rather than a learned shift.
        nn.init.zeros_(self.A2.bias)

        # A1 maps the hidden MLP features back to the embedding. Small weights make
        # the nonlinear correction tiny initially, so training starts near the
        # linear model and only grows the correction if it helps the loss.
        nn.init.normal_(self.A1.weight, mean=0.0, std=0.01)

    def raw_nonlinear_correction(self, u: torch.Tensor) -> torch.Tensor:
        """Return A1 sigma(A2 u + b), before normalization and alpha."""
        nonlinear_hidden = self.activation(self.A2(u))
        return self.A1(nonlinear_hidden)

    def components(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return normalized linear part and scaled normalized nonlinear part."""
        linear_part = self.linear_norm(self.linear(u))
        nonlinear_part = self.nonlinear_norm(self.raw_nonlinear_correction(u))
        scaled_nonlinear_part = self.nonlinear_scale * nonlinear_part
        return linear_part, scaled_nonlinear_part

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        linear_part, scaled_nonlinear_part = self.components(u)
        return linear_part + scaled_nonlinear_part

    def linear_weight(self) -> torch.Tensor:
        return self.linear.weight

    def nonlinear_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.A1.weight, self.A2.weight

    def nonlinear_correction(self, u: torch.Tensor) -> torch.Tensor:
        _, scaled_nonlinear_part = self.components(u)
        return scaled_nonlinear_part


class TwoViewLinearModel(nn.Module):
    """Two pure linear encoders, one for X and one for Y."""

    def __init__(
        self,
        p: int,
        q: int,
        embedding_dim: int,
        normalize_output: bool = True,
    ):
        super().__init__()
        self.encoder_x = LinearEncoder(
            p,
            embedding_dim,
            normalize_output=normalize_output,
        )
        self.encoder_y = LinearEncoder(
            q,
            embedding_dim,
            normalize_output=normalize_output,
        )

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder_x(X), self.encoder_y(Y)


class TwoViewResidualModel(nn.Module):
    """Two residual nonlinear encoders, one for X and one for Y."""

    def __init__(
        self,
        p: int,
        q: int,
        embedding_dim: int,
        hidden_dim: int,
        nonlinear_scale: float = 0.1,
        activation: str = "gelu",
        normalize_linear_part: bool = True,
    ):
        super().__init__()
        self.encoder_x = ResidualNonlinearEncoder(
            input_dim=p,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            nonlinear_scale=nonlinear_scale,
            activation=activation,
            normalize_linear_part=normalize_linear_part,
        )
        self.encoder_y = ResidualNonlinearEncoder(
            input_dim=q,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            nonlinear_scale=nonlinear_scale,
            activation=activation,
            normalize_linear_part=normalize_linear_part,
        )

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder_x(X), self.encoder_y(Y)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters so capacity can be compared across models."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


# Single builder used by training and experiment tables
def build_model(
    architecture: str,
    p: int,
    q: int,
    embedding_dim: int,
    hidden_dim: int,
    nonlinear_scale: float,
    normalize_linear_part: bool = True,
) -> nn.Module:
    """Build either the pure linear baseline or the residual nonlinear model."""
    if architecture == "linear":
        return TwoViewLinearModel(
            p=p,
            q=q,
            embedding_dim=embedding_dim,
            normalize_output=normalize_linear_part,
        )

    if architecture == "residual":
        return TwoViewResidualModel(
            p=p,
            q=q,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            nonlinear_scale=nonlinear_scale,
            activation="gelu",
            normalize_linear_part=normalize_linear_part,
        )

    raise ValueError("architecture must be 'linear' or 'residual'.")

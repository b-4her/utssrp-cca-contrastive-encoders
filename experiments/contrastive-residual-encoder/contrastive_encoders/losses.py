"""Loss functions for the contrastive encoder experiments.

The main function implements the paper objective directly, keeping it separate
from architecture choices and regularization penalties.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# Exact paper loss
def paper_contrastive_loss(
    z_x: torch.Tensor,
    z_y: torch.Tensor,
    normalize_embeddings: bool = False,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Return the negative paper objective, divided by batch size.

    The paper maximizes:

        sum_i <z_x_i, z_y_i>
        - sum_i log sum_j exp(<z_x_i, z_y_j>)
        - sum_i log sum_j exp(<z_x_j, z_y_i>)

    PyTorch minimizes losses, so this function returns -objective / N.
    By default this uses raw inner products, no temperature change, and no
    embedding normalization.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    if normalize_embeddings:
        z_x = F.normalize(z_x, dim=1)
        z_y = F.normalize(z_y, dim=1)

    scores = (z_x @ z_y.T) / temperature

    positive_score_sum = torch.diagonal(scores).sum()
    x_to_y_log_partition = torch.logsumexp(scores, dim=1).sum()
    y_to_x_log_partition = torch.logsumexp(scores, dim=0).sum()

    objective = positive_score_sum - x_to_y_log_partition - y_to_x_log_partition
    return -objective / z_x.shape[0]

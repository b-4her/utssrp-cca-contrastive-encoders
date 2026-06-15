"""Evaluation metrics for trained two-view encoders.

These metrics ask whether X/Y embeddings align on true pairs, fail on shuffled
pairs, retrieve the correct partner, and recover the known synthetic latents.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def columnwise_correlations(
    A: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Pearson correlation between corresponding columns."""
    A = np.asarray(A)
    B = np.asarray(B)

    A_centered = A - A.mean(axis=0, keepdims=True)
    B_centered = B - B.mean(axis=0, keepdims=True)

    denom = np.sqrt(
        np.sum(A_centered**2, axis=0)
        * np.sum(B_centered**2, axis=0)
    )

    return np.divide(
        np.sum(A_centered * B_centered, axis=0),
        denom,
        out=np.full(A.shape[1], np.nan),
        where=denom > 1e-12,
    )


def normalize_rows_np(Z: np.ndarray) -> np.ndarray:
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


# Pair-alignment metric
def alignment_gap(
    z_x: np.ndarray,
    z_y: np.ndarray,
) -> float:
    """Matched cosine similarity minus mismatched cosine similarity."""
    zx = normalize_rows_np(z_x)
    zy = normalize_rows_np(z_y)
    S = zx @ zy.T

    matched = np.diag(S).mean()
    if S.shape[0] <= 1:
        return float(matched)

    mismatched = (S.sum() - np.trace(S)) / (S.size - S.shape[0])
    return float(matched - mismatched)


def topk_retrieval_accuracy(
    z_x: np.ndarray,
    z_y: np.ndarray,
    k: int = 1,
) -> float:
    """For each x_i, check whether the paired y_i is among the top-k matches."""
    if k <= 0:
        raise ValueError("k must be positive.")

    zx = normalize_rows_np(z_x)
    zy = normalize_rows_np(z_y)
    S = zx @ zy.T

    k = min(k, S.shape[1])
    topk_predictions = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    truth = np.arange(S.shape[0])
    return float(np.mean(np.any(topk_predictions == truth[:, None], axis=1)))


def top1_retrieval_accuracy(
    z_x: np.ndarray,
    z_y: np.ndarray,
) -> float:
    """For each x_i, retrieve the closest y_j. Correct if j = i."""
    return topk_retrieval_accuracy(z_x, z_y, k=1)


def r2_score_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Return uniform-average R^2 using NumPy only."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    residual_sum = np.sum((y_true - y_pred) ** 2, axis=0)
    total_sum = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)

    valid = total_sum > 1e-12
    scores = np.full(y_true.shape[1], np.nan)
    scores[valid] = 1.0 - residual_sum[valid] / total_sum[valid]

    if not np.any(valid):
        return float("nan")

    return float(np.nanmean(scores))


def latent_probe_r2(
    z_train: np.ndarray,
    Z_train: Optional[np.ndarray],
    z_test: np.ndarray,
    Z_test: Optional[np.ndarray],
    alpha: float = 1e-3,
) -> float:
    """
    Train a ridge probe from learned embeddings to true latents and return test R^2.

    This uses a closed-form ridge solution so the project does not depend on
    scikit-learn. The intercept is handled by centering train data, and only the
    embedding-to-latent weights are ridge-penalized.
    """
    if Z_train is None or Z_test is None:
        return float("nan")

    if alpha < 0:
        raise ValueError("alpha must be nonnegative.")

    z_train = np.asarray(z_train)
    z_test = np.asarray(z_test)
    Z_train = np.asarray(Z_train)
    Z_test = np.asarray(Z_test)

    z_mean = z_train.mean(axis=0, keepdims=True)
    Z_mean = Z_train.mean(axis=0, keepdims=True)

    z_train_centered = z_train - z_mean
    Z_train_centered = Z_train - Z_mean
    z_test_centered = z_test - z_mean

    regularizer = alpha * np.eye(z_train_centered.shape[1])
    weights = np.linalg.solve(
        z_train_centered.T @ z_train_centered + regularizer,
        z_train_centered.T @ Z_train_centered,
    )

    Z_pred = z_test_centered @ weights + Z_mean
    return r2_score_np(Z_test, Z_pred)


# Synthetic latent recovery metric
def latent_recovery_scalar(
    z: np.ndarray,
    Z: Optional[np.ndarray],
) -> float:
    """
    For latent_dim = 1, measure best absolute correlation between any embedding
    dimension and the true latent Z.
    """
    if Z is None:
        return float("nan")

    if Z.shape[1] != 1:
        return float("nan")

    best = 0.0
    for j in range(z.shape[1]):
        corr = np.corrcoef(z[:, j], Z[:, 0])[0, 1]
        best = max(best, abs(float(corr)))

    return best

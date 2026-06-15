"""Synthetic paired-data generators for the contrastive experiments.

The generators create random X/Y datasets with either no relationship, a linear
latent relationship, or a cubic latent relationship. The data objects keep
separate X and Y latents so the setup resembles two paired modalities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PairedDataset:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    Z_x_train: Optional[np.ndarray] = None
    Z_y_train: Optional[np.ndarray] = None
    Z_x_test: Optional[np.ndarray] = None
    Z_y_test: Optional[np.ndarray] = None


@dataclass
class StandardizedDataset:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    Z_x_train: Optional[np.ndarray] = None
    Z_y_train: Optional[np.ndarray] = None
    Z_x_test: Optional[np.ndarray] = None
    Z_y_test: Optional[np.ndarray] = None


def generate_null_dataset(
    n_train: int,
    n_test: int,
    p: int,
    q: int,
    rng: np.random.Generator,
) -> PairedDataset:
    """Generate independent Gaussian noise in both views."""
    return PairedDataset(
        X_train=rng.standard_normal((n_train, p)),
        Y_train=rng.standard_normal((n_train, q)),
        X_test=rng.standard_normal((n_test, p)),
        Y_test=rng.standard_normal((n_test, q)),
    )


def orthonormal_loadings(
    rng: np.random.Generator,
    n_features: int,
    latent_dim: int,
) -> np.ndarray:
    """
    Generate random feature-space directions for the hidden latent factors.

    ===========================================================================
    MATHEMATICAL INTUITION & PURPOSE:
    ===========================================================================
    This function takes abstract, low-dimensional hidden concepts (Latent Factors) 
    and scatters them into a high-dimensional feature space (Observed Features).
    
    Why use QR decomposition instead of a standard random normal matrix?
    
    1) Orthogonality (Perpendicularity):
       It guarantees that each latent factor is mapped along a direction that is 
       exactly 90 degrees (perpendicular) to all other factors. This prevents 
       different hidden signals from bleeding into each other, eliminating 
       accidental redundancy or collinearity.
    
    2) Normalization (Unit Length):
       It forces the geometric length of every directional vector to be exactly 1.0. 
       This turns the loading matrix into a "pure geometric rotation". It spins 
       the latent space into the feature space without stretching, shrinking, 
       or distorting the size of the signal.
       
    The Result:
    You maintain absolute experimental control over the `signal_strength` and 
    variance. If your downstream model struggles to find the relationship, you 
    know it is because the data problem itself is genuinely hard, not because 
    the random data generator accidentally wiped out or tangled the signal.
    ===========================================================================
    
    Each column represents how one latent factor appears across the observed
    features. QR decomposition makes these loading directions orthonormal:
    each direction has unit length and different latent directions are
    perpendicular to one another.

    This keeps the signal strength controlled when changing the number of
    features and avoids accidentally making the latent factors redundant.
    """
    A = rng.standard_normal((n_features, latent_dim))
    Q, _ = np.linalg.qr(A)
    return Q[:, :latent_dim]


def standardize_latent_pair(
    Z_train: np.ndarray,
    Z_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize a latent variable using training statistics only."""
    mean = Z_train.mean(axis=0, keepdims=True)
    scale = Z_train.std(axis=0, ddof=1, keepdims=True)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (Z_train - mean) / scale, (Z_test - mean) / scale


# Latent relationship generator
def related_latents(
    n_train: int,
    n_test: int,
    latent_dim: int,
    relationship: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one latent for X and a related latent for Y.

    This is the caption/image-style setup: X and Y do not literally share the
    same latent variable. Instead, Y's latent is generated from X's latent.
    """
    Z_x_train = rng.standard_normal((n_train, latent_dim))
    Z_x_test = rng.standard_normal((n_test, latent_dim))
    Z_x_train, Z_x_test = standardize_latent_pair(Z_x_train, Z_x_test)

    if relationship == "linear":
        Z_y_train = Z_x_train.copy()
        Z_y_test = Z_x_test.copy()
    elif relationship == "cubic":
        Z_y_train = Z_x_train**3
        Z_y_test = Z_x_test**3
    else:
        raise ValueError("relationship must be 'linear' or 'cubic'.")

    Z_y_train, Z_y_test = standardize_latent_pair(Z_y_train, Z_y_test)

    return Z_x_train, Z_y_train, Z_x_test, Z_y_test


# Main paired-signal generator
def generate_related_signal_dataset(
    n_train: int,
    n_test: int,
    p: int,
    q: int,
    latent_dim: int,
    signal_strength: float,
    noise_std: float,
    relationship: str,
    rng: np.random.Generator,
) -> PairedDataset:
    """
    Generate paired views with related but view-specific latent variables.

    X uses Z_x. Y uses Z_y.

    Linear signal:
        Z_y = Z_x

    Nonlinear/cubic signal:
        Z_y = standardized(Z_x ** 3)
    """
    if latent_dim > min(p, q):
        raise ValueError("latent_dim must be <= min(p, q).")

    A_x = orthonormal_loadings(rng, p, latent_dim)
    A_y = orthonormal_loadings(rng, q, latent_dim)

    Z_x_train, Z_y_train, Z_x_test, Z_y_test = related_latents(
        n_train=n_train,
        n_test=n_test,
        latent_dim=latent_dim,
        relationship=relationship,
        rng=rng,
    )

    X_train = signal_strength * (Z_x_train @ A_x.T) + noise_std * rng.standard_normal((n_train, p))
    Y_train = signal_strength * (Z_y_train @ A_y.T) + noise_std * rng.standard_normal((n_train, q))
    X_test = signal_strength * (Z_x_test @ A_x.T) + noise_std * rng.standard_normal((n_test, p))
    Y_test = signal_strength * (Z_y_test @ A_y.T) + noise_std * rng.standard_normal((n_test, q))

    return PairedDataset(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        Z_x_train=Z_x_train,
        Z_y_train=Z_y_train,
        Z_x_test=Z_x_test,
        Z_y_test=Z_y_test,
    )


def generate_shared_signal_dataset(
    n_train: int,
    n_test: int,
    p: int,
    q: int,
    latent_dim: int,
    signal_strength: float,
    noise_std: float,
    rng: np.random.Generator,
) -> PairedDataset:
    """Generate paired views where the latent relationship is Z_y = Z_x."""
    return generate_related_signal_dataset(
        n_train=n_train,
        n_test=n_test,
        p=p,
        q=q,
        latent_dim=latent_dim,
        signal_strength=signal_strength,
        noise_std=noise_std,
        relationship="linear",
        rng=rng,
    )


def generate_nonlinear_shared_signal_dataset(
    n_train: int,
    n_test: int,
    p: int,
    q: int,
    latent_dim: int,
    signal_strength: float,
    noise_std: float,
    rng: np.random.Generator,
) -> PairedDataset:
    """
    Generate paired views where the latent relationship is cubic.

    X uses Z_x, while Y uses Z_y = standardized(Z_x ** 3). This asks whether a
    nonlinear encoder can learn a paired relationship that a linear encoder
    should struggle to represent from the X side.
    """
    return generate_related_signal_dataset(
        n_train=n_train,
        n_test=n_test,
        p=p,
        q=q,
        latent_dim=latent_dim,
        signal_strength=signal_strength,
        noise_std=noise_std,
        relationship="cubic",
        rng=rng,
    )


def deterministic_relation(
    X: np.ndarray,
    q: int,
    relationship: str,
) -> np.ndarray:
    """
    Apply a direct coordinate-wise relation from X to a clean Y signal.

    This is different from the latent-generator path above: there is no hidden
    loading matrix and no noise added to X. The clean target is literally
    f(X), and controlled Gaussian noise is added only afterward.
    """
    if q > X.shape[1]:
        raise ValueError("q must be <= p for the coordinate-wise deterministic relation.")

    X_used = X[:, :q]

    if relationship == "linear":
        return X_used.copy()
    if relationship == "quadratic":
        return X_used**2
    if relationship == "cubic":
        return X_used**3
    if relationship == "sine":
        return np.sin(X_used)

    raise ValueError("relationship must be 'linear', 'quadratic', 'cubic', or 'sine'.")


def generate_deterministic_relation_dataset(
    n_train: int,
    n_test: int,
    p: int,
    q: int,
    relationship: str,
    target_snr: float,
    rng: np.random.Generator,
) -> PairedDataset:
    """
    Generate Y = f(X) + epsilon with an explicit signal-to-noise ratio.

    The target SNR follows the whiteboard/paper convention:

        SNR = Var(f(X)) / Var(epsilon)

    For vector-valued Y, Var(f(X)) is measured as the average coordinate
    variance of the clean deterministic target on the training set. The clean
    target is stored in Z_y_* so the notebook can use ridge-probe R^2 as a PVE
    diagnostic.
    """
    if target_snr <= 0:
        raise ValueError("target_snr must be positive.")

    X_train = rng.standard_normal((n_train, p))
    X_test = rng.standard_normal((n_test, p))

    Y_clean_train = deterministic_relation(X_train, q=q, relationship=relationship)
    Y_clean_test = deterministic_relation(X_test, q=q, relationship=relationship)

    signal_variance = float(np.mean(np.var(Y_clean_train, axis=0, ddof=1)))
    noise_std = np.sqrt(signal_variance / target_snr)

    Y_train = Y_clean_train + noise_std * rng.standard_normal((n_train, q))
    Y_test = Y_clean_test + noise_std * rng.standard_normal((n_test, q))

    return PairedDataset(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        Z_x_train=X_train,
        Z_y_train=Y_clean_train,
        Z_x_test=X_test,
        Z_y_test=Y_clean_test,
    )


def deterministic_dataset_snr(dataset: PairedDataset) -> dict[str, float]:
    """
    Estimate realized SNR and oracle PVE for a deterministic-relation dataset.

    If Y_clean = f(X) and Y = Y_clean + epsilon, then the best possible
    prediction function f has:

        PVE(f) = SNR / (1 + SNR)
    """
    if dataset.Z_y_train is None:
        raise ValueError("dataset.Z_y_train must contain the clean deterministic target.")

    noise_train = dataset.Y_train - dataset.Z_y_train
    signal_variance = float(np.mean(np.var(dataset.Z_y_train, axis=0, ddof=1)))
    noise_variance = float(np.mean(np.var(noise_train, axis=0, ddof=1)))
    realized_snr = signal_variance / max(noise_variance, 1e-12)

    return {
        "realized_signal_variance": signal_variance,
        "realized_noise_variance": noise_variance,
        "realized_snr": realized_snr,
        "oracle_pve": realized_snr / (1.0 + realized_snr),
    }


# Train-only standardization
def standardize_train_test(
    dataset: PairedDataset,
    standardize: bool = True,
) -> StandardizedDataset:
    """Center and optionally scale using training rows only."""
    x_mean = dataset.X_train.mean(axis=0, keepdims=True)
    y_mean = dataset.Y_train.mean(axis=0, keepdims=True)

    if standardize:
        x_scale = dataset.X_train.std(axis=0, ddof=1, keepdims=True)
        y_scale = dataset.Y_train.std(axis=0, ddof=1, keepdims=True)
        x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
        y_scale = np.where(y_scale > 1e-12, y_scale, 1.0)
    else:
        x_scale = 1.0
        y_scale = 1.0

    return StandardizedDataset(
        X_train=(dataset.X_train - x_mean) / x_scale,
        Y_train=(dataset.Y_train - y_mean) / y_scale,
        X_test=(dataset.X_test - x_mean) / x_scale,
        Y_test=(dataset.Y_test - y_mean) / y_scale,
        Z_x_train=dataset.Z_x_train,
        Z_y_train=dataset.Z_y_train,
        Z_x_test=dataset.Z_x_test,
        Z_y_test=dataset.Z_y_test,
    )

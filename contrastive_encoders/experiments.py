"""Experiment orchestration for the contrastive encoder comparison.

This module defines the model grid, creates readable model-spec tables, and runs
each model on the noise-only, linear-signal, and cubic-signal datasets.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .architectures import build_model, count_parameters
from .data import (
    PairedDataset,
    deterministic_dataset_snr,
    generate_deterministic_relation_dataset,
    generate_nonlinear_shared_signal_dataset,
    generate_null_dataset,
    generate_shared_signal_dataset,
)
from .training import TrainConfig, train_one_model


def make_first_experiment_configs(epochs: int = 300) -> List[TrainConfig]:
    """
    Initial small comparison grid.

    The pure linear and residual models use the same paper loss. Differences in
    results should therefore come from architecture and regularization choices.
    """
    hidden_dim = 16

    return [
        TrainConfig(
            name="Linear encoder (alpha=0)",
            architecture="linear",
            hidden_dim=0,
            nonlinear_scale=0.0,
            epochs=epochs,
        ),
        TrainConfig(
            name="MLP nonlinear (alpha=0.01)",
            architecture="residual",
            hidden_dim=hidden_dim,
            nonlinear_scale=0.01,
            epochs=epochs,
        ),
        TrainConfig(
            name="MLP nonlinear (alpha=0.10)",
            architecture="residual",
            hidden_dim=hidden_dim,
            nonlinear_scale=0.10,
            epochs=epochs,
        ),
        TrainConfig(
            name="MLP nonlinear (alpha=1.00)",
            architecture="residual",
            hidden_dim=hidden_dim,
            nonlinear_scale=1.00,
            nonlinear_output_strength=0.0,
            epochs=epochs,
        ),
        TrainConfig(
            name="L1-regularized nonlinear (alpha=0.10)",
            architecture="residual",
            hidden_dim=hidden_dim,
            nonlinear_scale=0.10,
            l1_nonlinear=1e-4,
            epochs=epochs,
        ),
        TrainConfig(
            name="L2-regularized nonlinear (alpha=0.10)",
            architecture="residual",
            hidden_dim=hidden_dim,
            nonlinear_scale=0.10,
            l2_nonlinear=1e-4,
            epochs=epochs,
        ),
    ]


def make_alpha_sweep_configs(
    alphas: Sequence[float] = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0),
    epochs: int = 300,
) -> List[TrainConfig]:
    """Create residual-encoder configs for testing alpha as a continuous knob."""
    hidden_dim = 16

    return [
        TrainConfig(
            name=f"MLP nonlinear (alpha={alpha:.2f})",
            architecture="residual",
            hidden_dim=hidden_dim,
            nonlinear_scale=float(alpha),
            epochs=epochs,
        )
        for alpha in alphas
    ]


# Human-readable model table for the notebook
def make_model_spec_table(
    configs: List[TrainConfig],
    p: int = 128,
    q: int = 128,
) -> pd.DataFrame:
    """Create a human-readable table describing each model in the grid."""
    rows = []

    for config in configs:
        is_linear = config.architecture == "linear"
        model = build_model(
            architecture=config.architecture,
            p=p,
            q=q,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            nonlinear_scale=config.nonlinear_scale,
            normalize_linear_part=config.normalize_linear_part,
        )

        if is_linear:
            description = "Only a normalized linear map for X and Y"
            nonlinear_layers = "none"
            nonlinear_role = "alpha=0, so no nonlinear correction is added"
            encoder_formula = "g(u) = Normalize(G u)"
            alpha_meaning = "alpha is not used for the pure linear baseline"
            normalization = "encoder output uses BatchNorm1d affine=False"
        else:
            description = "Linear map plus a one-hidden-layer MLP correction"
            nonlinear_layers = (
                f"X A2: {p}->{config.hidden_dim}, X A1: "
                f"{config.hidden_dim}->{config.embedding_dim}; "
                f"Y A2: {q}->{config.hidden_dim}, Y A1: "
                f"{config.hidden_dim}->{config.embedding_dim}"
            )
            nonlinear_role = (
                f"alpha={config.nonlinear_scale:g} scales the normalized "
                "MLP correction"
            )
            encoder_formula = (
                "g(u) = Normalize(G u) + alpha * "
                "Normalize(A1 sigma(A2 u + b))"
            )
            alpha_meaning = (
                "alpha multiplies the normalized nonlinear correction "
                "before it is added to the normalized linear part"
            )
            normalization = "residual branches use BatchNorm1d affine=False"

        l1_parts = []
        if config.l1_linear > 0:
            l1_parts.append(f"linear={config.l1_linear:g}")
        if config.l1_nonlinear > 0:
            l1_parts.append(f"A2={config.l1_nonlinear:g}")

        l2_parts = []
        if config.l2_linear > 0:
            l2_parts.append(f"linear={config.l2_linear:g}")
        if config.l2_nonlinear > 0:
            l2_parts.append(f"A1/A2={config.l2_nonlinear:g}")

        rows.append(
            {
                "Model": config.name,
                "Plain meaning": description,
                "Encoder formula": encoder_formula,
                "Linear layer sizes": (
                    f"X G: {p}->{config.embedding_dim}; "
                    f"Y G: {q}->{config.embedding_dim}"
                ),
                "Nonlinear layer sizes": nonlinear_layers,
                "Embedding dim": config.embedding_dim,
                "Hidden dim": "none" if is_linear else config.hidden_dim,
                "Alpha": config.nonlinear_scale,
                "Alpha meaning": alpha_meaning,
                "Normalization": normalization,
                "Nonlinear role": nonlinear_role,
                "L1 penalty": ", ".join(l1_parts) if l1_parts else "none",
                "L2 penalty": ", ".join(l2_parts) if l2_parts else "none",
                "Trainable parameters": count_parameters(model),
            }
        )

    return pd.DataFrame(rows)


def make_experiment_datasets(
    seed: int,
    n_train: int = 160,
    n_test: int = 800,
    p: int = 128,
    q: int = 128,
    signal_strength: float = 2.0,
    noise_std: float = 1.0,
) -> Dict[str, PairedDataset]:
    """
    Build the three datasets used in the main comparison.

    Returning the datasets separately lets the notebook reuse the exact same
    data generation logic for heatmaps and branch-history diagnostics.
    """
    rng = np.random.default_rng(seed)

    null_data = generate_null_dataset(
        n_train=n_train,
        n_test=n_test,
        p=p,
        q=q,
        rng=rng,
    )

    signal_data = generate_shared_signal_dataset(
        n_train=n_train,
        n_test=n_test,
        p=p,
        q=q,
        latent_dim=1,
        signal_strength=signal_strength,
        noise_std=noise_std,
        rng=rng,
    )

    nonlinear_signal_data = generate_nonlinear_shared_signal_dataset(
        n_train=n_train,
        n_test=n_test,
        p=p,
        q=q,
        latent_dim=1,
        signal_strength=signal_strength,
        noise_std=noise_std,
        rng=rng,
    )

    return {
        "Noise only": null_data,
        "Linear signal": signal_data,
        "Cubic signal": nonlinear_signal_data,
    }


# Full experiment runner
def run_first_experiment(
    configs: List[TrainConfig],
    seed: int,
    device: str,
    n_train: int = 160,
    n_test: int = 800,
    p: int = 128,
    q: int = 128,
    signal_strength: float = 2.0,
    noise_std: float = 1.0,
) -> pd.DataFrame:
    """
    Run each config on pure-noise, linear-signal, and cubic-signal data.

    This returns one row per setting/config pair.
    """
    datasets = make_experiment_datasets(
        seed=seed,
        n_train=n_train,
        n_test=n_test,
        p=p,
        q=q,
        signal_strength=signal_strength,
        noise_std=noise_std,
    )

    records = []
    for config_index, config in enumerate(configs):
        for setting_name, dataset in datasets.items():
            metrics = train_one_model(
                dataset=dataset,
                config=config,
                seed=seed + config_index,
                device=device,
                standardize=True,
            )

            records.append(
                {
                    "setting": setting_name,
                    "config": config.name,
                    "architecture": config.architecture,
                    "embedding_dim": config.embedding_dim,
                    "hidden_dim": config.hidden_dim,
                    "nonlinear_scale": config.nonlinear_scale,
                    "normalize_linear_part": config.normalize_linear_part,
                    "l1_linear": config.l1_linear,
                    "l1_nonlinear": config.l1_nonlinear,
                    "l2_linear": config.l2_linear,
                    "l2_nonlinear": config.l2_nonlinear,
                    "nonlinear_output_strength": config.nonlinear_output_strength,
                    **metrics,
                }
            )

    return pd.DataFrame(records)


def run_alpha_sweep(
    seed: int,
    device: str,
    alphas: Sequence[float] = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0),
    epochs: int = 300,
    n_train: int = 160,
    n_test: int = 800,
    p: int = 128,
    q: int = 128,
    signal_strength: float = 2.0,
    noise_std: float = 1.0,
) -> pd.DataFrame:
    """Run the alpha sweep on noise-only, linear-signal, and cubic-signal data."""
    configs = make_alpha_sweep_configs(alphas=alphas, epochs=epochs)
    return run_first_experiment(
        configs=configs,
        seed=seed,
        device=device,
        n_train=n_train,
        n_test=n_test,
        p=p,
        q=q,
        signal_strength=signal_strength,
        noise_std=noise_std,
    )


def run_signal_noise_sweep(
    configs: List[TrainConfig],
    seed: int,
    device: str,
    signal_strength_values: Sequence[float] = (1.0, 2.0, 3.0, 5.0),
    noise_std_values: Sequence[float] = (0.25, 0.5, 1.0, 2.0),
    setting: str = "Cubic signal",
    n_train: int = 160,
    n_test: int = 800,
    p: int = 128,
    q: int = 128,
) -> pd.DataFrame:
    """
    Sweep signal_strength / noise_std for one paired-signal setting.

    This helps separate a weak-signal problem from an architecture problem.
    """
    if setting not in {"Linear signal", "Cubic signal"}:
        raise ValueError("setting must be 'Linear signal' or 'Cubic signal'.")

    records = []
    for strength_index, signal_strength in enumerate(signal_strength_values):
        for noise_index, noise_std in enumerate(noise_std_values):
            dataset_seed = seed + 1000 * strength_index + noise_index
            rng = np.random.default_rng(dataset_seed)

            if setting == "Linear signal":
                dataset = generate_shared_signal_dataset(
                    n_train=n_train,
                    n_test=n_test,
                    p=p,
                    q=q,
                    latent_dim=1,
                    signal_strength=float(signal_strength),
                    noise_std=float(noise_std),
                    rng=rng,
                )
            else:
                dataset = generate_nonlinear_shared_signal_dataset(
                    n_train=n_train,
                    n_test=n_test,
                    p=p,
                    q=q,
                    latent_dim=1,
                    signal_strength=float(signal_strength),
                    noise_std=float(noise_std),
                    rng=rng,
                )

            for config_index, config in enumerate(configs):
                metrics = train_one_model(
                    dataset=dataset,
                    config=config,
                    seed=dataset_seed + config_index,
                    device=device,
                    standardize=True,
                )

                records.append(
                    {
                        "setting": setting,
                        "signal_strength": float(signal_strength),
                        "noise_std": float(noise_std),
                        "signal_to_noise": float(signal_strength) / float(noise_std),
                        "config": config.name,
                        "architecture": config.architecture,
                        "embedding_dim": config.embedding_dim,
                        "hidden_dim": config.hidden_dim,
                        "nonlinear_scale": config.nonlinear_scale,
                        "l1_nonlinear": config.l1_nonlinear,
                        "l2_nonlinear": config.l2_nonlinear,
                        **metrics,
                    }
                )

    return pd.DataFrame(records)


def run_deterministic_relation_experiment(
    configs: List[TrainConfig],
    seed: int,
    device: str,
    relationships: Sequence[str] = ("linear", "cubic"),
    snr_values: Sequence[float] = (0.5, 2.0, 8.0),
    n_train: int = 160,
    n_test: int = 800,
    p: int = 128,
    q: int = 128,
) -> pd.DataFrame:
    """
    Run models on direct deterministic relations Y = f(X) + epsilon.

    This experiment controls noise with the paper-style signal-to-noise ratio:

        SNR = Var(f(X)) / Var(epsilon)

    It complements the earlier latent experiments, where X and Y were generated
    by projecting low-dimensional latents into noisy high-dimensional views.
    """
    records = []

    for relationship_index, relationship in enumerate(relationships):
        for snr_index, target_snr in enumerate(snr_values):
            dataset_seed = seed + 1000 * relationship_index + snr_index
            rng = np.random.default_rng(dataset_seed)
            dataset = generate_deterministic_relation_dataset(
                n_train=n_train,
                n_test=n_test,
                p=p,
                q=q,
                relationship=relationship,
                target_snr=float(target_snr),
                rng=rng,
            )
            snr_info = deterministic_dataset_snr(dataset)
            setting_name = f"Deterministic {relationship}"

            for config_index, config in enumerate(configs):
                metrics = train_one_model(
                    dataset=dataset,
                    config=config,
                    seed=dataset_seed + config_index,
                    device=device,
                    standardize=True,
                )

                records.append(
                    {
                        "setting": setting_name,
                        "relationship": relationship,
                        "target_snr": float(target_snr),
                        "realized_snr": snr_info["realized_snr"],
                        "realized_signal_variance": snr_info[
                            "realized_signal_variance"
                        ],
                        "realized_noise_variance": snr_info[
                            "realized_noise_variance"
                        ],
                        "oracle_pve": snr_info["oracle_pve"],
                        "config": config.name,
                        "architecture": config.architecture,
                        "embedding_dim": config.embedding_dim,
                        "hidden_dim": config.hidden_dim,
                        "nonlinear_scale": config.nonlinear_scale,
                        "l1_nonlinear": config.l1_nonlinear,
                        "l2_nonlinear": config.l2_nonlinear,
                        **metrics,
                    }
                )

    return pd.DataFrame(records)

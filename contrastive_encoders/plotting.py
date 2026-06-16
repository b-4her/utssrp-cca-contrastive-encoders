"""Plotting and display helpers for the experiment notebook.

The raw result columns keep code-friendly names, while this module provides
friendlier labels and compact bar plots for notebook interpretation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PathLike = Union[str, Path]


REPORT_PALETTE = [
    "#2F6B9A",  # deep blue
    "#D66B2A",  # warm orange
    "#2F8F5B",  # green
    "#8E5EA2",  # muted purple
    "#4C9DA8",  # teal
    "#C84C5A",  # soft red
    "#6C757D",  # gray
]

HEATMAP_CMAP = "cividis"


def set_report_plot_style() -> None:
    """Apply a clean, colorblind-friendly style for report-ready figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2F3A45",
            "axes.labelcolor": "#24313D",
            "axes.titlecolor": "#18222D",
            "axes.titleweight": "semibold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.color": "#34424F",
            "ytick.color": "#34424F",
            "font.size": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.facecolor": "white",
            "legend.edgecolor": "#D8DEE6",
            "grid.color": "#D8DEE6",
            "grid.linestyle": "-",
            "grid.linewidth": 0.75,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


set_report_plot_style()


FRIENDLY_COLUMN_NAMES: Dict[str, str] = {
    "setting": "Dataset",
    "config": "Model",
    "architecture": "Architecture",
    "embedding_dim": "Embedding dim",
    "hidden_dim": "Hidden dim",
    "nonlinear_scale": "Alpha",
    "normalize_linear_part": "Normalize linear branch",
    "l1_linear": "Linear L1 penalty",
    "l1_nonlinear": "Nonlinear L1 penalty",
    "l2_linear": "Linear L2 penalty",
    "l2_nonlinear": "Nonlinear L2 penalty",
    "nonlinear_output_strength": "Nonlinear output penalty",
    "relationship": "Deterministic relation",
    "target_snr": "Target SNR",
    "realized_snr": "Realized SNR",
    "realized_signal_variance": "Realized signal variance",
    "realized_noise_variance": "Realized noise variance",
    "oracle_pve": "Oracle PVE",
    "parameter_count": "Trainable parameters",
    "parameter_count_per_train_sample": "Parameters per training pair",
    "train_best_view_correlation": "Train X/Y embedding correlation",
    "test_best_view_correlation": "Test X/Y embedding correlation",
    "train_pair_separation": "Train true-pair separation",
    "test_pair_separation": "Test true-pair separation",
    "shuffled_pair_separation": "Shuffled-pair check",
    "train_pair_match_accuracy": "Train exact-pair accuracy",
    "test_pair_match_accuracy": "Test exact-pair accuracy",
    "train_top5_pair_match_accuracy": "Train top-5 retrieval accuracy",
    "test_top5_pair_match_accuracy": "Test top-5 retrieval accuracy",
    "x_signal_recovery": "X signal recovery",
    "x_related_signal_recovery": "X recovery of paired Y latent",
    "y_signal_recovery": "Y signal recovery",
    "x_probe_r2_z_x": "X probe R^2 for Z_x",
    "x_probe_r2_z_y": "X probe R^2 for Z_y",
    "y_probe_r2_z_y": "Y probe R^2 for Z_y",
    "x_linear_branch_norm": "X linear branch norm",
    "x_nonlinear_branch_norm": "X nonlinear branch norm",
    "x_nonlinear_to_linear_ratio": "X nonlinear/linear ratio",
    "y_linear_branch_norm": "Y linear branch norm",
    "y_nonlinear_branch_norm": "Y nonlinear branch norm",
    "y_nonlinear_to_linear_ratio": "Y nonlinear/linear ratio",
    "mean_nonlinear_to_linear_ratio": "Mean nonlinear/linear ratio",
}


_TABLE_ONLY_COLUMNS = {
    "setting",
    "config",
    "architecture",
    "parameter_count",
    "parameter_count_per_train_sample",
}


FRIENDLY_METRIC_NAMES: Dict[str, str] = {
    key: value
    for key, value in FRIENDLY_COLUMN_NAMES.items()
    if key not in _TABLE_ONLY_COLUMNS
}


def _slugify(text: str) -> str:
    """Convert a plot title into a readable file name."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def _save_path(
    title: str,
    save_dir: Optional[PathLike],
    filename: Optional[str],
) -> Optional[Path]:
    if save_dir is None:
        return None

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{_slugify(title)}.png"
    elif not filename.endswith(".png"):
        filename = f"{filename}.png"

    return output_dir / filename


def _finish_plot(
    fig: plt.Figure,
    title: str,
    save_dir: Optional[PathLike] = None,
    filename: Optional[str] = None,
    dpi: int = 240,
) -> Optional[Path]:
    """Tighten, optionally save, and display a figure."""
    fig.tight_layout()
    output_path = _save_path(title=title, save_dir=save_dir, filename=filename)

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, facecolor="white")

    plt.show()
    return output_path


def friendly_results_table(results: pd.DataFrame) -> pd.DataFrame:
    """Rename result columns for display without changing the stored results."""
    return results.rename(columns=FRIENDLY_COLUMN_NAMES)


# Notebook bar-plot helper
def plot_metric_by_config(
    results: pd.DataFrame,
    setting: str,
    metrics: List[str],
    title: str,
    ylabel: str = "Score",
    metric_labels: Optional[Dict[str, str]] = None,
    save_dir: Optional[PathLike] = None,
    filename: Optional[str] = None,
    reference_line_y: Optional[float] = None,
    reference_line_label: Optional[str] = None,
) -> Optional[Path]:
    metric_labels = metric_labels or FRIENDLY_METRIC_NAMES
    subset = results[results["setting"] == setting].copy()

    x = np.arange(len(subset))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(10, 5))

    for k, metric in enumerate(metrics):
        ax.bar(
            x + (k - (len(metrics) - 1) / 2) * width,
            subset[metric],
            width=width,
            label=metric_labels.get(metric, metric),
            color=REPORT_PALETTE[k % len(REPORT_PALETTE)],
            edgecolor="white",
            linewidth=0.7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(subset["config"], rotation=18, ha="right")
    ax.axhline(0, linestyle=":", color="#202830", linewidth=1.1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if reference_line_y is not None:
        ax.axhline(
            reference_line_y,
            linestyle="--",
            color="#9A3D3D",
            linewidth=1.8,
            label=reference_line_label or f"Reference = {reference_line_y:.2f}",
        )

    ax.legend()
    return _finish_plot(fig, title=title, save_dir=save_dir, filename=filename)


def plot_train_test_separation_by_setting(
    results: pd.DataFrame,
    setting: str,
    save_dir: Optional[PathLike] = None,
) -> Optional[Path]:
    """Plot train/test true-pair separation for one dataset setting."""
    return plot_metric_by_config(
        results,
        setting=setting,
        metrics=["train_pair_separation", "test_pair_separation"],
        title=f"{setting}: train versus held-out true-pair separation",
        ylabel="True-pair separation",
        save_dir=save_dir,
    )


def plot_top5_retrieval_by_setting(
    results: pd.DataFrame,
    setting: str,
    save_dir: Optional[PathLike] = None,
) -> Optional[Path]:
    """Plot held-out top-5 retrieval accuracy for one dataset setting."""
    return plot_metric_by_config(
        results,
        setting=setting,
        metrics=["test_top5_pair_match_accuracy"],
        title=f"{setting}: held-out top-5 retrieval accuracy",
        ylabel="Top-5 retrieval accuracy",
        save_dir=save_dir,
    )


def plot_signal_recovery_by_config(
    results: pd.DataFrame,
    setting: str,
    title: Optional[str] = None,
    save_dir: Optional[PathLike] = None,
) -> Optional[Path]:
    """
    Compare which latent each embedding is recovering.

    The middle bar asks whether the X embedding is correlated with the paired Y
    latent. In the current nonlinear experiment, that means the transformed
    latent used to generate Y.
    """
    title = title or f"{setting}: embedding correlation with synthetic latents"
    return plot_metric_by_config(
        results,
        setting=setting,
        metrics=[
            "x_signal_recovery",
            "x_related_signal_recovery",
            "y_signal_recovery",
        ],
        title=title,
        ylabel="Absolute correlation with latent",
        metric_labels={
            **FRIENDLY_METRIC_NAMES,
            "x_signal_recovery": "X embedding vs Z_x",
            "x_related_signal_recovery": "X embedding vs Z_y",
            "y_signal_recovery": "Y embedding vs Z_y",
        },
        save_dir=save_dir,
    )


def plot_latent_probe_r2_by_config(
    results: pd.DataFrame,
    setting: str,
    title: Optional[str] = None,
    save_dir: Optional[PathLike] = None,
    show_oracle_pve: bool = False,
) -> Optional[Path]:
    """
    Plot ridge-probe R^2 as percent of latent variation explained.

    The probe is trained from the learned embedding to the true latent on train
    data and evaluated on held-out data.
    """
    title = title or f"{setting}: ridge-probe R^2 latent variation explained"
    plot_data = results.copy()
    metric_map = {
        "x_probe_r2_z_x": "x_probe_r2_z_x_percent",
        "x_probe_r2_z_y": "x_probe_r2_z_y_percent",
        "y_probe_r2_z_y": "y_probe_r2_z_y_percent",
    }

    for source, target in metric_map.items():
        plot_data[target] = 100.0 * plot_data[source]

    reference_line_y = None
    reference_line_label = None
    if show_oracle_pve and "oracle_pve" in plot_data.columns:
        oracle_pve = float(plot_data["oracle_pve"].dropna().mean())
        reference_line_y = 100.0 * oracle_pve
        reference_line_label = f"Oracle PVE = {reference_line_y:.1f}%"

    return plot_metric_by_config(
        plot_data,
        setting=setting,
        metrics=list(metric_map.values()),
        title=title,
        ylabel="Test R^2 (% of latent variation explained)",
        metric_labels={
            "x_probe_r2_z_x_percent": "X embedding -> Z_x",
            "x_probe_r2_z_y_percent": "X embedding -> Z_y",
            "y_probe_r2_z_y_percent": "Y embedding -> Z_y",
        },
        save_dir=save_dir,
        reference_line_y=reference_line_y,
        reference_line_label=reference_line_label,
    )


def plot_branch_ratio_history(
    history: pd.DataFrame,
    title: str = "Branch contribution ratio over training",
    save_dir: Optional[PathLike] = None,
    filename: Optional[str] = None,
) -> Optional[Path]:
    """
    Plot nonlinear/linear branch norm ratio across training epochs.

    If a config column is present, each model is drawn as a separate curve.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    if "config" in history.columns:
        for index, (config_name, group) in enumerate(history.groupby("config", sort=False)):
            group = group.sort_values("epoch")
            ax.plot(
                group["epoch"],
                group["mean_nonlinear_to_linear_ratio"],
                marker="o",
                markersize=4.5,
                linewidth=2.0,
                label=config_name,
                color=REPORT_PALETTE[index % len(REPORT_PALETTE)],
            )
        ax.legend()
    else:
        history = history.sort_values("epoch")
        ax.plot(
            history["epoch"],
            history["mean_nonlinear_to_linear_ratio"],
            marker="o",
            markersize=4.5,
            linewidth=2.0,
            color=REPORT_PALETTE[0],
        )

    ax.axhline(0, linestyle=":", color="#202830", linewidth=1.1)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Nonlinear branch norm / linear branch norm")
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _finish_plot(fig, title=title, save_dir=save_dir, filename=filename)


def plot_similarity_heatmap(
    z_x: np.ndarray,
    z_y: np.ndarray,
    n: int = 40,
    title: str = "X/Y similarity matrix on a held-out batch",
    normalize_rows: bool = False,
    save_dir: Optional[PathLike] = None,
    filename: Optional[str] = None,
) -> Optional[Path]:
    """
    Plot S = z_x z_y^T for a small held-out batch.

    Rows are X samples and columns are Y samples. A bright diagonal means the
    model gives true pairs higher similarity than mismatched pairs.
    """
    z_x_batch = z_x[:n]
    z_y_batch = z_y[:n]

    if normalize_rows:
        z_x_batch = z_x_batch / np.maximum(
            np.linalg.norm(z_x_batch, axis=1, keepdims=True),
            1e-12,
        )
        z_y_batch = z_y_batch / np.maximum(
            np.linalg.norm(z_y_batch, axis=1, keepdims=True),
            1e-12,
        )

    similarity = z_x_batch @ z_y_batch.T

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if normalize_rows:
        image = ax.imshow(similarity, cmap=HEATMAP_CMAP, vmin=-1, vmax=1)
        colorbar_label = "Cosine similarity"
    else:
        image = ax.imshow(similarity, cmap=HEATMAP_CMAP)
        colorbar_label = "Dot-product similarity"

    ax.set_title(title)
    ax.set_xlabel("Y sample index")
    ax.set_ylabel("X sample index")
    fig.colorbar(image, ax=ax, label=colorbar_label)
    return _finish_plot(fig, title=title, save_dir=save_dir, filename=filename)


def plot_alpha_sweep_curve(
    results: pd.DataFrame,
    metric: str = "test_pair_separation",
    title: str = "Alpha sweep: held-out true-pair separation",
    save_dir: Optional[PathLike] = None,
    filename: Optional[str] = None,
) -> Optional[Path]:
    """Plot a curve showing how performance changes as alpha changes."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for index, (setting, group) in enumerate(results.groupby("setting", sort=False)):
        group = group.sort_values("nonlinear_scale")
        ax.plot(
            group["nonlinear_scale"],
            group[metric],
            marker="o",
            markersize=5,
            linewidth=2.1,
            label=setting,
            color=REPORT_PALETTE[index % len(REPORT_PALETTE)],
        )

    ax.axhline(0, linestyle=":", color="#202830", linewidth=1.1)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xticks(sorted(results["nonlinear_scale"].unique()))
    ax.set_xticklabels(
        [f"{alpha:g}" for alpha in sorted(results["nonlinear_scale"].unique())]
    )
    ax.set_title(title)
    ax.set_xlabel("alpha")
    ax.set_ylabel(FRIENDLY_METRIC_NAMES.get(metric, metric))
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    return _finish_plot(fig, title=title, save_dir=save_dir, filename=filename)


def plot_signal_noise_sweep(
    results: pd.DataFrame,
    metric: str = "test_pair_separation",
    title: str = "Signal/noise sweep: held-out true-pair separation",
    save_dir: Optional[PathLike] = None,
    filename: Optional[str] = None,
) -> Optional[Path]:
    """Plot performance as a function of signal_strength / noise_std."""
    averaged = (
        results.groupby(["config", "signal_to_noise"], as_index=False)[metric]
        .mean()
        .sort_values("signal_to_noise")
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for index, (config_name, group) in enumerate(averaged.groupby("config", sort=False)):
        ax.plot(
            group["signal_to_noise"],
            group[metric],
            marker="o",
            markersize=5,
            linewidth=2.1,
            label=config_name,
            color=REPORT_PALETTE[index % len(REPORT_PALETTE)],
        )

    ax.axhline(0, linestyle=":", color="#202830", linewidth=1.1)
    ax.set_title(title)
    ax.set_xlabel("signal_strength / noise_std")
    ax.set_ylabel(FRIENDLY_METRIC_NAMES.get(metric, metric))
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    return _finish_plot(fig, title=title, save_dir=save_dir, filename=filename)


def plot_deterministic_snr_sweep(
    results: pd.DataFrame,
    metric: str = "test_pair_separation",
    relationship: Optional[str] = None,
    title: Optional[str] = None,
    save_dir: Optional[PathLike] = None,
    filename: Optional[str] = None,
    show_oracle_pve: bool = False,
) -> Optional[Path]:
    """Plot deterministic-relation performance against target SNR."""
    plot_data = results.copy()
    if relationship is not None:
        plot_data = plot_data[plot_data["relationship"] == relationship].copy()

    if title is None:
        relation_label = relationship or "all deterministic relations"
        title = f"{relation_label}: metric versus target SNR"

    averaged = (
        plot_data.groupby(["config", "target_snr"], as_index=False)[metric]
        .mean()
        .sort_values("target_snr")
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for index, (config_name, group) in enumerate(averaged.groupby("config", sort=False)):
        ax.plot(
            group["target_snr"],
            group[metric],
            marker="o",
            markersize=5,
            linewidth=2.1,
            label=config_name,
            color=REPORT_PALETTE[index % len(REPORT_PALETTE)],
        )

    if show_oracle_pve and "oracle_pve" in plot_data.columns:
        oracle = (
            plot_data.groupby("target_snr", as_index=False)["oracle_pve"]
            .mean()
            .sort_values("target_snr")
        )
        ax.plot(
            oracle["target_snr"],
            oracle["oracle_pve"],
            linestyle="--",
            marker="s",
            markersize=4.5,
            linewidth=1.9,
            label="Oracle PVE",
            color="#9A3D3D",
        )

    ax.axhline(0, linestyle=":", color="#202830", linewidth=1.1)
    if (averaged["target_snr"] > 0).all():
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("target SNR = Var(f(X)) / Var(epsilon)")
    ax.set_ylabel(FRIENDLY_METRIC_NAMES.get(metric, metric))
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    return _finish_plot(fig, title=title, save_dir=save_dir, filename=filename)

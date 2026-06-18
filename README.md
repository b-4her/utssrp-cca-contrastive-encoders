# UTSSRP Contrastive Encoder Comparison

Project dates: June 8-19, 2026

This repository contains materials from an open-ended UTSSRP research project
on CCA and contrastive encoders. My final contribution focuses on comparing
linear encoders, residual nonlinear encoders, and regularized nonlinear
variants on controlled synthetic paired-data problems.

The question I focused on was:

```text
When Y = f(X) + noise, does the nonlinear contrastive encoder recover the
real paired signal, or does the extra capacity mostly memorize training pairs?
```

That contribution focuses on deterministic signal recovery, overfitting against
an ideal clean/noisy reference, and branch-probe diagnostics that ask what the
residual nonlinear branch actually makes readable.

## Start Here

| Path | What to use it for |
| --- | --- |
| [encoder_experiment_summary.pdf](encoder_experiment_summary.pdf) | Read the exported final summary first. |
| [research-outputs/encoder_experiment_summary.pptx](research-outputs/encoder_experiment_summary.pptx) | Edit or present from the final summary deck. |
| [experiment-notebooks/](experiment-notebooks/) | Follow the seven experiment notebooks in final reading order. |
| [contrastive_encoders/](contrastive_encoders/) | Inspect the reusable Python package imported by the notebooks. |
| [research-outputs/](research-outputs/) | Find final report plots, poster assets, and the editable deck. |
| [additional-research-materials/](additional-research-materials/) | Browse earlier CCA notebooks, references, and intermediate material. |

The notebooks are intentionally thin around the research flow: they set up each
experiment, call reusable package functions, and save figures. Most reusable
implementation details live in `contrastive_encoders/`.

## Main Findings

- Noise-only data is a memorization sanity check. Models can increase training
  alignment even when X and Y are independent, so held-out and shuffled-pair
  scores are the useful evidence.
- In deterministic linear and cubic settings, held-out separation improves as
  SNR rises, but learned encoders remain below the ideal clean/noisy
  cosine-separation reference. The cubic relationship is harder than the
  linear one.
- Low-SNR settings show the clearest overfitting risk. The `alpha=1` nonlinear
  model has the strongest excess train-over-ideal warning, while smaller alpha
  and regularized variants are more controlled.
- Scalar branch probes give positive evidence that the residual nonlinear
  branch carries nonlinear structure. For square and cubic targets, curvature
  is more readable from the nonlinear branch and combined embedding than from
  the linear branch alone. Recovery is partial; the exponential target still
  under-shoots in the tail.

## Notebook Guide

1. [01-latent-signal-comparison.ipynb](experiment-notebooks/01-latent-signal-comparison.ipynb)
   runs the first paired-view comparison on noise-only, linear-signal, and
   cubic-signal synthetic datasets.
2. [02-alpha-and-signal-noise-sweeps.ipynb](experiment-notebooks/02-alpha-and-signal-noise-sweeps.ipynb)
   sweeps nonlinear branch scale `alpha` and signal/noise strength.
3. [03-deterministic-relations.ipynb](experiment-notebooks/03-deterministic-relations.ipynb)
   tests deterministic relationships `Y = f(X) + epsilon` with explicit SNR,
   including linear, cubic, exponential, and signed-log targets.
4. [04-deterministic-pve-to-cosine-reference.ipynb](experiment-notebooks/04-deterministic-pve-to-cosine-reference.ipynb)
   converts ideal PVE into the cosine-separation scale used by the plots and
   builds overfitting diagnostics.
5. [05-function-and-weight-checks.ipynb](experiment-notebooks/05-function-and-weight-checks.ipynb)
   checks scalar function readouts, coordinate probes, and linear-branch
   weight behavior.
6. [06-alpha1-branch-decomposition.ipynb](experiment-notebooks/06-alpha1-branch-decomposition.ipynb)
   decomposes one `alpha=1` residual model into linear-branch, nonlinear-branch,
   and combined readouts.
7. [07-alpha1-scalar-cubic-result-only.ipynb](experiment-notebooks/07-alpha1-scalar-cubic-result-only.ipynb)
   contains the final scalar branch-probe results for cubic, quadratic,
   exponential, log, and reciprocal targets, plus poster-ready summary figures.

If you only want the final story, read the summary deck/PDF first, then inspect
notebooks 03, 04, 06, and 07.

## Experiment Design

The main experiments compare the model grid on:

- `Noise only`: X and Y are independent Gaussian noise.
- `Linear signal`: paired views share a linear latent relationship.
- `Cubic signal`: paired views share a standardized cubic latent relationship.
- Deterministic relations: `Y = f(X) + epsilon` with explicit SNR and an
  empirical ideal cosine-separation reference.
- Scalar branch probes: `p=q=1` deterministic targets used to visualize what is
  readable from the linear and nonlinear branches.

The main model families are:

- `Linear encoder (alpha=0)`
- `MLP nonlinear (alpha=0.01)`
- `MLP nonlinear (alpha=0.10)`
- `MLP nonlinear (alpha=1.00)`
- `L1-regularized nonlinear (alpha=0.10)`
- `L2-regularized nonlinear (alpha=0.10)`

For residual nonlinear encoders:

```text
g(u) = Normalize(G u) + alpha * Normalize(A1 sigma(A2 u + b))
```

For the pure linear baseline:

```text
g(u) = Normalize(G u)
```

The final architecture keeps the embedding dimension at 4 and the nonlinear
hidden layer at 16 for the main comparisons. `BatchNorm1d(..., affine=False)`
normalizes branch outputs so `alpha` meaningfully controls the residual
nonlinear correction size.

## Cosine Separation Metric

Most of the main plots use the same pair-separation metric. This metric is
inspired by the contrastive loss function, which aims to make paired X/Y
embeddings more similar than unpaired X/Y embeddings. In this project, the
diagnostic uses cosine similarity so the score measures whether true pairs
stand out from mismatched pairs after normalization.

For a batch of paired embeddings `(z_i^x, z_i^y)`, first normalize each
embedding row:

```text
hat(z_i^x) = z_i^x / ||z_i^x||_2
hat(z_j^y) = z_j^y / ||z_j^y||_2
```

Then form the cosine-similarity matrix:

```text
S_ij = hat(z_i^x) dot hat(z_j^y)
```

The separation score is:

```text
separation =
    mean_i S_ii
    - mean_{i != j} S_ij
```

Equivalently:

```text
separation =
    (1 / n) sum_i S_ii
    - (1 / (n^2 - n)) sum_{i != j} S_ij
```

The diagonal terms `S_ii` are the true X/Y pairs. The off-diagonal terms
`S_ij`, where `i != j`, are mismatched pairs. A high separation score means the
model gives true pairs higher cosine similarity than mismatched pairs. A score
near zero means true pairs are not standing out from mismatches.

## Main Metrics

- `train_pair_separation`: this separation score on training pairs.
- `test_pair_separation`: the same separation score on held-out pairs.
- `shuffled_pair_separation`: the same score after breaking the X/Y pairing.
- `test_top5_pair_match_accuracy`: held-out retrieval accuracy.
- `x_signal_recovery`, `x_related_signal_recovery`, `y_signal_recovery`:
  latent-recovery correlations.
- `x_probe_r2_z_x`, `x_probe_r2_z_y`, `y_probe_r2_z_y`: held-out ridge-probe
  R^2 diagnostics.
- `mean_nonlinear_to_linear_ratio`: measured nonlinear branch size relative to
  the linear branch.
- `ideal_cosine_separation`: empirical clean/noisy deterministic reference on
  the same scale as the separation plots.
- `excess_train_over_ideal`: clipped overfitting warning,
  `max(train separation - ideal separation, 0)`.

## Outputs

Final curated outputs live in [research-outputs/](research-outputs/):

- [research-outputs/report-plots/](research-outputs/report-plots/) contains
  the report-ready PNG plots generated for the completed contribution.
- [research-outputs/poster-assets/](research-outputs/poster-assets/) contains
  the final scalar branch-probe poster figures.
- [research-outputs/encoder_experiment_summary.pptx](research-outputs/encoder_experiment_summary.pptx)
  is the editable final summary deck.

Some notebook output logs still show the historical `reports/` path from
earlier runs. For regeneration in the final layout, set notebook output cells to
use:

```python
PLOT_DIR = module_root / "research-outputs" / "report-plots"
POSTER_PLOT_DIR = module_root / "research-outputs" / "poster-assets"
```

## Package Organization

- [contrastive_encoders/architectures.py](contrastive_encoders/architectures.py)
  defines the linear encoder, residual nonlinear encoder, two-view wrappers,
  and model builder.
- [contrastive_encoders/data.py](contrastive_encoders/data.py) generates null,
  paired latent, shared-signal, nonlinear shared-signal, and deterministic SNR
  datasets.
- [contrastive_encoders/losses.py](contrastive_encoders/losses.py) implements
  the paired contrastive objective.
- [contrastive_encoders/regularization.py](contrastive_encoders/regularization.py)
  contains L1, L2, and nonlinear-output penalties.
- [contrastive_encoders/metrics.py](contrastive_encoders/metrics.py) computes
  alignment, retrieval, correlation, and latent-probe metrics.
- [contrastive_encoders/training.py](contrastive_encoders/training.py) trains
  one model and returns diagnostics or full artifacts.
- [contrastive_encoders/experiments.py](contrastive_encoders/experiments.py)
  builds model grids and runs the notebook experiment batches.
- [contrastive_encoders/interpretability.py](contrastive_encoders/interpretability.py)
  contains ridge-probe helpers used by the branch-readout notebooks.
- [contrastive_encoders/plotting.py](contrastive_encoders/plotting.py) creates
  notebook and report plots.
- [contrastive_encoders/__init__.py](contrastive_encoders/__init__.py)
  re-exports the main functions and classes for cleaner notebook imports.

## Running The Notebooks

Open notebooks from the repository root or from `experiment-notebooks/`. The
setup cells add the repository root to `sys.path`, so local package imports work
without installing the package.

The notebooks use Python with Jupyter, NumPy, Pandas, PyTorch, and Matplotlib.
If imports fail after code edits, restart the notebook kernel and run from the
top.

## Credits

This project was supervised by Professor Ricardo Baptista and TA Luis Sierra.
The project contributors were Baher Alabbar, Niv Karo, Doris Ding, and Masha
Glasman.

My main contribution focused on the contrastive-encoder side of the project.
Niv Karo, Doris Ding, and Masha Glasman mainly focused on the CCA analysis and
on investigating solutions to the correlation-inflation problem.

Parts of the encoder-side implementation and documentation were developed with
help from Codex. I guided the experiment design, reviewed the code, and
interpreted the results.

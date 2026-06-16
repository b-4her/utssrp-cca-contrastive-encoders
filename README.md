# UTSSRP Contrastive Encoder Comparison

This project was part of the UTSSRP program. My main contribution is the
encoder comparison experiment, which compares linear encoders, residual
nonlinear encoders, and regularized nonlinear variants on synthetic paired-data
settings.

The central question is:

```text
Does the nonlinear correction help recover a real paired signal,
or does it mostly increase the risk of fitting noise?
```

## Start Here

- [encoder-comparison.ipynb](encoder-comparison.ipynb) is the main experiment
  notebook. Run this from top to bottom.
- [contrastive_encoders/](contrastive_encoders/) contains the reusable Python
  code imported by the notebook.
- [reports/](reports/) contains report-facing outputs, including the final
  slide deck, report plots, and space for the poster when it is added.
- [reports/report-plots/](reports/report-plots/) contains PNG figures generated
  from the experiment.
- [reports/contrastive-encoder-comparison-final-slides.pptx](reports/contrastive-encoder-comparison-final-slides.pptx)
  is the final slide deck for the encoder comparison.
- [supporting-materials/](supporting-materials/) contains earlier notebooks,
  reference PDFs, generated presentation outputs, and other supporting work.

The notebook is intentionally thin: it sets up the experiment, displays the
model table, runs the comparison, and plots the metrics. Most implementation
details live in the package files.

The later notebook sections add diagnostic plots: train/test separation for
each dataset, top-5 retrieval accuracy, latent-correlation comparisons,
ridge-probe R^2 plots, branch-ratio history and similarity matrix heatmaps for
the linear and cubic signal settings, an alpha sweep, and a signal/noise sweep.

When the notebook runs, every plot is also saved as a report-ready PNG in
[reports/report-plots/](reports/report-plots/). File names are generated from
the plot titles so they are easy to match with the notebook sections.

## Experiment Settings

The experiment compares models on:

- `Noise only`: X and Y are independent Gaussian noise.
- `Linear signal`: X has latent `Z_x`, Y has latent `Z_y`, and `Z_y = Z_x`.
- `Cubic signal`: X has latent `Z_x`, Y has latent `Z_y`, and
  `Z_y = standardized(Z_x ** 3)`.

This is meant to mimic a two-view setup, such as captions and images, where the
views are different but paired through an underlying relationship.

## Model Families

The model grid includes:

- `Linear encoder (alpha=0)`
- `MLP nonlinear (alpha=0.01)`
- `MLP nonlinear (alpha=0.10)`
- `MLP nonlinear (alpha=1.00)`
- `L1-regularized nonlinear (alpha=0.10)`
- `L2-regularized nonlinear (alpha=0.10)`

All nonlinear models in the main grid use the same one-hidden-layer MLP size:
`128 -> 16 -> 4` for X and `128 -> 16 -> 4` for Y. Keeping this fixed makes
the alpha comparison cleaner because alpha changes while MLP capacity stays the
same.

The L1-regularized model applies lasso to the input-facing nonlinear layer
`A2`. The L2-regularized model applies ridge-style shrinkage to the nonlinear
weights `A1` and `A2`.

For the residual nonlinear models:

```text
g(u) = Normalize(G u) + alpha * Normalize(A1 sigma(A2 u + b))
```

For the pure linear baseline:

```text
g(u) = Normalize(G u)
```

The normalization layers use `BatchNorm1d(..., affine=False)`. In the residual
models, this makes `alpha` meaningfully control the size of the nonlinear
correction. In the pure linear model, it keeps the embedding scale controlled
before the raw-dot-product loss sees it.

The normalization is applied to branch outputs, not to the weights. In other
words, the model normalizes `G u` and `A1 sigma(A2 u + b)` after they have been
computed. This matters because increasing `A1` cannot simply cancel a small
`alpha`; the nonlinear branch is normalized before `alpha` is applied.

Normalizing the linear branch inside the residual model also puts the linear and
nonlinear branches on comparable scale. Then `alpha=0.10` can be interpreted as
the nonlinear correction being much smaller than the linear branch. The absolute
embedding size is less important here than whether the embedding preserves the
paired structure we care about. One caveat is that the paper loss uses raw inner
products, so scale can still affect optimization; normalization is used here to
make the architecture comparison more controlled.

## Package Organization

The implementation is split so the notebook stays readable:

- [contrastive_encoders/architectures.py](contrastive_encoders/architectures.py)
  defines the linear encoder, residual nonlinear encoder, two-view model
  wrappers, and model builder.
- [contrastive_encoders/data.py](contrastive_encoders/data.py) generates the
  synthetic paired datasets, including independent noise, linear latent
  relationships, and cubic latent relationships.
- [contrastive_encoders/losses.py](contrastive_encoders/losses.py) implements
  the paper contrastive objective exactly as a PyTorch loss.
- [contrastive_encoders/regularization.py](contrastive_encoders/regularization.py)
  contains L1, L2, and nonlinear-output penalties.
- [contrastive_encoders/metrics.py](contrastive_encoders/metrics.py) computes
  alignment, retrieval, correlation, and latent recovery metrics.
- [contrastive_encoders/training.py](contrastive_encoders/training.py) trains
  one model on one dataset and returns all diagnostics.
- [contrastive_encoders/experiments.py](contrastive_encoders/experiments.py)
  builds the model grid, creates all datasets, and runs the full comparison.
- [contrastive_encoders/plotting.py](contrastive_encoders/plotting.py) creates
  the notebook plots.
- [contrastive_encoders/__init__.py](contrastive_encoders/__init__.py)
  re-exports the main functions and classes for cleaner notebook imports.

## Main Metrics

- `train_pair_separation`
  True X/Y training pairs are more similar than mismatched pairs.

- `test_pair_separation`
  The same separation on held-out pairs.

- `shuffled_pair_separation`
  Sanity check after breaking the X/Y pairing; should stay near zero.

- `test_top5_pair_match_accuracy`
  Held-out retrieval check: for each X sample, whether its true paired Y sample
  is among the 5 most similar Y samples.

- `x_signal_recovery`
  How well the X embedding recovers the true X latent `Z_x`.

- `x_related_signal_recovery`
  How well the X embedding recovers the paired Y latent `Z_y`. In the cubic
  setting, this means `Z_y = standardized(Z_x ** 3)`.

- `y_signal_recovery`
  How well the Y embedding recovers the true Y latent `Z_y`.

- `x_probe_r2_z_x`, `x_probe_r2_z_y`, `y_probe_r2_z_y`
  Held-out ridge-probe R^2 scores. These measure how much of the true latent
  variation can be explained from the learned embedding using a simple ridge
  regression fit on the training data. In the plots these are shown as
  percentages.

- `mean_nonlinear_to_linear_ratio`
  Diagnostic showing the measured nonlinear branch size relative to the linear
  branch. This should increase as `alpha` increases.

## Typical Workflow

1. Open [encoder-comparison.ipynb](encoder-comparison.ipynb).
2. Run the imports and model table cells.
3. Run the experiment cell.
4. Inspect the focused comparison table.
5. Inspect the plots, especially the noise-only and cubic-signal panels.
6. Run the sweep sections when you want the slower alpha and signal/noise
   comparisons.
7. Use the generated PNG files in [reports/report-plots/](reports/report-plots/)
   for your report.

If the import cell complains after code edits, restart the notebook kernel and
run the notebook from the top.

## Credits

Parts of this project were implemented with help from Codex. I guided the
experiment design, reviewed the code, and interpreted the results.

Earlier CCA, tapering, and exploratory materials are kept out of the main path
in [supporting-materials/](supporting-materials/).

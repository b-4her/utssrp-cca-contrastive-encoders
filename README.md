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
  notebook.
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

## Experiment Overview

The notebook compares models on noise-only, linear-signal, and cubic-signal
paired datasets. It tracks train and held-out pair separation, shuffled-pair
sanity checks, top-5 retrieval accuracy, latent recovery metrics, ridge-probe
R^2 diagnostics, nonlinear-to-linear branch ratios, and similarity matrix
heatmaps.

The implementation is split so the notebook stays readable:

- [contrastive_encoders/architectures.py](contrastive_encoders/architectures.py)
  defines the encoder architectures and two-view model wrappers.
- [contrastive_encoders/data.py](contrastive_encoders/data.py) generates the
  synthetic paired datasets.
- [contrastive_encoders/losses.py](contrastive_encoders/losses.py) implements
  the contrastive objective.
- [contrastive_encoders/experiments.py](contrastive_encoders/experiments.py)
  builds and runs the model grid.
- [contrastive_encoders/plotting.py](contrastive_encoders/plotting.py) creates
  the report-ready plots.

Earlier CCA, tapering, and exploratory materials are kept out of the main path
in [supporting-materials/](supporting-materials/).

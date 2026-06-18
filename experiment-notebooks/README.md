# Experiment Notebooks

These notebooks split the encoder comparison work into smaller experiment
sections. Each notebook includes the shared import/setup cells and can be run
from inside this folder or from the repository root.

## Notebooks

- [01-latent-signal-comparison.ipynb](01-latent-signal-comparison.ipynb)
  runs the main paired-view experiment on noise-only, linear-signal, and
  cubic-signal synthetic datasets. It compares the model grid and includes the
  main diagnostic plots, branch-ratio histories, and held-out similarity
  heatmaps.

- [02-alpha-and-signal-noise-sweeps.ipynb](02-alpha-and-signal-noise-sweeps.ipynb)
  tests how the residual nonlinear branch scale `alpha` changes held-out
  alignment, then sweeps signal strength versus noise in the cubic-signal
  setting.

- [03-deterministic-relations.ipynb](03-deterministic-relations.ipynb)
  runs the direct deterministic experiments where `Y = f(X) + epsilon` with
  explicit SNR control. It covers linear, cubic, exponential, and signed-log
  relations, plus coordinate-probe and held-out similarity-matrix diagnostics.

- [04-deterministic-pve-to-cosine-reference.ipynb](04-deterministic-pve-to-cosine-reference.ipynb)
  shows the cosine-separation function and converts ideal PVE into an ideal
  cosine-separation reference for the deterministic plots, then adds
  overfitting diagnostics versus SNR.

- [05-function-and-weight-checks.ipynb](05-function-and-weight-checks.ipynb)
  turns the whiteboard `f(x)` and weight-checking ideas into concrete
  diagnostics. It includes scalar function probe curves, coordinate-sweep
  probes for deterministic relations, and linear-branch weight/kernel checks.

- [06-alpha1-branch-decomposition.ipynb](06-alpha1-branch-decomposition.ipynb)
  focuses on one residual nonlinear model with `alpha=1.0` and separates what
  is readable from the linear branch, nonlinear branch, and combined embedding.
  It includes branch-only probes, shared-probe decompositions, and additional
  verification plots for the scalar cubic experiment.

- [07-alpha1-scalar-cubic-result-only.ipynb](07-alpha1-scalar-cubic-result-only.ipynb)
  is the focused branch-probe result notebook for scalar `p=q=1` deterministic
  relations with `alpha=1.0`. It compares cubic, square/quadratic, and
  exponential targets using the same branch-story plots: branches together,
  combined readout versus true target, and all curves overlaid.

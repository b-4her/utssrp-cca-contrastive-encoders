# Experiment Notebooks

These notebooks are the final reproducible path for the contrastive-encoder
comparison. They are split by research question so each notebook can be read or
rerun independently after the shared setup cells.

The broad sequence is:

```text
baseline paired-view comparison
  -> alpha and signal/noise sweeps
  -> deterministic SNR experiments
  -> ideal PVE/cosine reference
  -> function, weight, and branch-readout checks
  -> final scalar branch-probe/poster results
```

## Final Notebook Order

- [01-latent-signal-comparison.ipynb](01-latent-signal-comparison.ipynb)
  runs the first paired-view experiment on noise-only, linear-signal, and
  cubic-signal datasets. It includes the initial model grid, comparison tables,
  held-out plots, branch-ratio histories, and similarity heatmaps.

- [02-alpha-and-signal-noise-sweeps.ipynb](02-alpha-and-signal-noise-sweeps.ipynb)
  tests how residual nonlinear scale `alpha` affects held-out alignment and
  sweeps signal strength versus noise in the cubic-signal setting.

- [03-deterministic-relations.ipynb](03-deterministic-relations.ipynb)
  moves to direct deterministic relationships `Y = f(X) + epsilon` with
  explicit SNR. It covers linear, cubic, exponential, and signed-log relations,
  plus coordinate probes and held-out similarity matrices.

- [04-deterministic-pve-to-cosine-reference.ipynb](04-deterministic-pve-to-cosine-reference.ipynb)
  converts ideal PVE into the same cosine-separation scale used by the
  deterministic plots. It adds the red ideal reference line and the final
  overfitting diagnostics:
  `max(train separation - ideal separation, 0)` and
  `max(train separation - test separation, 0)`.

- [05-function-and-weight-checks.ipynb](05-function-and-weight-checks.ipynb)
  turns the scalar `f(x)` and weight-checking ideas into concrete diagnostics:
  scalar probe curves, coordinate-sweep probes, and linear-branch
  weight/kernel checks.

- [06-alpha1-branch-decomposition.ipynb](06-alpha1-branch-decomposition.ipynb)
  focuses on one residual nonlinear model with `alpha=1.0`. It separates what
  is readable from the linear branch, nonlinear branch, and combined embedding
  using branch-only probes and shared-probe decompositions.

- [07-alpha1-scalar-cubic-result-only.ipynb](07-alpha1-scalar-cubic-result-only.ipynb)
  is the final scalar branch-probe notebook. It uses `p=q=1` deterministic
  targets so the learned readouts can be plotted directly. It covers cubic,
  quadratic, exponential, log, and reciprocal targets, and it saves the
  poster-ready summary figures.

## Reading Paths

- For the final contribution story, start with notebook 03 for deterministic SNR
  behavior, notebook 04 for the ideal reference and overfitting diagnostics,
  and notebooks 06-07 for branch-probe interpretability.
- For a full reproduction pass, run notebooks 01 through 07 in order.
- For poster figures only, use notebook 07 and the saved assets in
  [../research-outputs/poster-assets/](../research-outputs/poster-assets/).

## Output Notes

The curated final outputs live in:

- [../research-outputs/report-plots/](../research-outputs/report-plots/)
- [../research-outputs/poster-assets/](../research-outputs/poster-assets/)

Notebook output cells now write final artifacts directly to:

```python
PLOT_DIR = module_root / "research-outputs" / "report-plots"
POSTER_PLOT_DIR = module_root / "research-outputs" / "poster-assets"
```

The notebooks can be run from this folder or from the repository root. The setup
cells add the repository root to `sys.path` before importing
`contrastive_encoders`.

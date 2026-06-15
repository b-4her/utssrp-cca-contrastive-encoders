# UTSSRP CCA and Contrastive Encoder Notes

Research notes, notebooks, experiment code, figures, and presentation outputs for
CCA, correlation inflation, tapering, and contrastive residual encoder work.

## Repository Layout

- [reference-materials](reference-materials/README.md) - background papers and slides.
- [provided-notebooks](provided-notebooks/README.md) - starter/reference notebooks and figures.
- [exploratory-notebooks](exploratory-notebooks/README.md) - personal exploratory analyses.
- [experiments](experiments/README.md) - organized experiment notebooks and reusable code.
- [presentation-outputs](presentation-outputs/README.md) - generated slide decks and presentation artifacts.

## Experiment Index

- [Contrastive residual encoder comparison](experiments/contrastive-residual-encoder/encoder-comparison.ipynb)
  compares linear encoders, residual nonlinear encoders, and regularized variants.
- [CCA tapering baseline](experiments/tapering-cca/tapering-cca-baseline.ipynb)
  explores tapering behavior in CCA-style settings.
- [CCA tapering exploration](experiments/tapering-cca/tapering-cca-exploration.ipynb)
  contains additional tapering experiments and checks.
- [Spurious correlation check](experiments/tapering-cca/spurious-correlation-check.ipynb)
  explores spurious-correlation examples.
- [CCA correlation inflation experiments](exploratory-notebooks/cca-correlation-inflation.ipynb)
  investigates when CCA correlations appear inflated.
- [Linear encoder regularization experiments](exploratory-notebooks/linear-encoder-regularization.ipynb)
  compares regularization behavior for linear encoders.
- [Nonlinear contrastive noise experiment](exploratory-notebooks/nonlinear-contrastive-noise.ipynb)
  studies contrastive encoders under noise and capacity changes.
- [General exploratory experiments](exploratory-notebooks/general-cca-experiments.ipynb)
  collects early exploratory CCA and encoder tests.

## Useful Outputs

- [Contrastive encoder final slide deck](presentation-outputs/contrastive-encoder-final-deck/contrastive-encoder-final-12-slide-deck.pptx)
- [Current contrastive encoder architecture deck](presentation-outputs/current-contrastive-encoder-architecture/current-contrastive-encoder-architecture.pptx)
- [Contrastive encoder report plots](experiments/contrastive-residual-encoder/report-plots/README.md)

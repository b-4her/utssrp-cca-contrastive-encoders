"""Public imports for the contrastive encoder experiment package.

The notebook imports from this file so it does not need to know the internal
module layout. Most implementation details live in the sibling modules.
"""

from .architectures import (
    LinearEncoder,
    ResidualNonlinearEncoder,
    TwoViewLinearModel,
    TwoViewResidualModel,
    build_model,
    count_parameters,
)
from .data import (
    PairedDataset,
    StandardizedDataset,
    deterministic_dataset_snr,
    deterministic_relation,
    generate_deterministic_relation_dataset,
    generate_nonlinear_shared_signal_dataset,
    generate_null_dataset,
    generate_related_signal_dataset,
    generate_shared_signal_dataset,
    standardize_train_test,
)
from .experiments import (
    make_alpha_sweep_configs,
    make_experiment_datasets,
    make_first_experiment_configs,
    make_model_spec_table,
    run_alpha_sweep,
    run_deterministic_relation_experiment,
    run_first_experiment,
    run_signal_noise_sweep,
)
from .losses import paper_contrastive_loss
from .interpretability import (
    RidgeProbe,
    cubic_coordinate_probe_curve,
    fit_ridge_probe,
    predict_ridge_probe,
)
from .metrics import (
    alignment_gap,
    columnwise_correlations,
    latent_probe_r2,
    latent_recovery_scalar,
    top1_retrieval_accuracy,
    topk_retrieval_accuracy,
)
from .plotting import (
    FRIENDLY_COLUMN_NAMES,
    FRIENDLY_METRIC_NAMES,
    friendly_results_table,
    plot_alpha_sweep_curve,
    plot_branch_ratio_history,
    plot_coordinate_probe_curves,
    plot_deterministic_snr_sweep,
    plot_latent_probe_r2_by_config,
    plot_metric_by_config,
    plot_signal_noise_sweep,
    plot_signal_recovery_by_config,
    plot_similarity_heatmap,
    plot_top5_retrieval_by_setting,
    plot_train_test_separation_by_setting,
    set_report_plot_style,
)
from .training import (
    TrainConfig,
    TrainingArtifacts,
    train_one_model,
    train_one_model_with_artifacts,
)

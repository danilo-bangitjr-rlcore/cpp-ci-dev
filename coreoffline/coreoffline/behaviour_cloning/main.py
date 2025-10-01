import logging

import numpy as np
from corerl.data_pipeline.datatypes import Transition
from corerl.state import AppState
from lib_config.loader import load_config
from lib_defs.config_defs.tag_config import TagType

from coreoffline.behaviour_cloning.data import (
    ModelData,
    prepare_features_and_targets,
)
from coreoffline.behaviour_cloning.evaluation import calculate_per_action_metrics
from coreoffline.behaviour_cloning.models import BaseRegressor, LinearRegressor, MLPRegressor
from coreoffline.behaviour_cloning.plotting import create_single_action_scatter_plot
from coreoffline.utils.config import OfflineMainConfig
from coreoffline.core.setup import create_standard_setup
from coreoffline.utils.data_loading import load_offline_transitions

log = logging.getLogger(__name__)


def get_ai_setpoint_tag_names(cfg: OfflineMainConfig) -> list[str]:
    """Extract all ai_setpoint tag names from the configuration."""
    ai_setpoint_tags = [tag.name for tag in cfg.pipeline.tags if tag.type == TagType.ai_setpoint]

    if not ai_setpoint_tags:
        raise ValueError("No ai_setpoint tag found in configuration")

    return ai_setpoint_tags


def run_cross_validation(
    model: BaseRegressor,
    data: ModelData,
    n_splits: int,
):
    """Run cross validation using k-fold splits from ModelData."""
    all_y_true = []
    all_y_pred = []

    for train_data, test_data in data.k_fold_split(n_splits):
        model.fit(
            train_data.X,
            train_data.y,
            X_test=test_data.X,
            y_test=test_data.y,
        )
        y_pred = model.predict(test_data.X)

        all_y_true.append(test_data.y)
        all_y_pred.append(y_pred)

    # Combine all predictions and targets
    all_y_true = np.vstack(all_y_true)
    all_y_pred = np.vstack(all_y_pred)

    return all_y_true, all_y_pred


def run_baseline_cross_validation(data: ModelData, n_splits: int):
    """Run cross validation for copy-forward baseline."""
    all_y_true = []
    all_y_pred = []

    for _, test_data in data.k_fold_split(n_splits):
        # Copy-forward baseline: use previous actions as predictions
        y_pred = np.clip(test_data.baseline_y, 0, 1)

        all_y_true.append(test_data.y)
        all_y_pred.append(y_pred)

    # Combine all predictions and targets
    all_y_true = np.vstack(all_y_true)
    all_y_pred = np.vstack(all_y_pred)

    # Calculate and log losses
    train_loss = np.mean((np.clip(data.baseline_y, 0, 1) - data.y) ** 2)
    test_loss = np.mean((all_y_pred - all_y_true) ** 2)
    log.info(f"Copy Forward - Training Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

    return all_y_true, all_y_pred


def run_behaviour_cloning(app_state: AppState, transitions: list[Transition]):
    assert isinstance(app_state.cfg, OfflineMainConfig)

    # Get all action names from configuration
    action_names = get_ai_setpoint_tag_names(app_state.cfg)
    log.info(f"Training models for {len(action_names)} actions: {action_names}")

    data = prepare_features_and_targets(
        transitions,
        action_names=action_names,
    )

    # Copy Forward Baseline
    log.info("Training Copy Forward baseline model...")
    all_y_true, all_y_pred_baseline = run_baseline_cross_validation(
        data,
        n_splits=app_state.cfg.behaviour_clone.k_folds,
    )
    baseline_per_action_metrics = calculate_per_action_metrics(
        all_y_true,
        all_y_pred_baseline,
        data.action_names,
    )
    log.info("Copy Forward baseline.")
    for action_name, metrics in baseline_per_action_metrics.items():
        log.info(f"  {action_name}: MAE={metrics['mae']:.6f}, Sign Acc={metrics['sign_acc']:.6f}")

    # Linear Regression
    log.info("Training Linear Regression model...")
    mlp = LinearRegressor(app_state)
    all_y_true, all_y_pred_linear = run_cross_validation(
        mlp,
        data,
        n_splits=app_state.cfg.behaviour_clone.k_folds,
    )
    linear_per_action_metrics = calculate_per_action_metrics(
        all_y_true,
        all_y_pred_linear,
        data.action_names,
    )
    log.info("Done training Linear Regression model.")
    for action_name, metrics in linear_per_action_metrics.items():
        log.info(f"  {action_name}: MAE={metrics['mae']:.6f}, Sign Acc={metrics['sign_acc']:.6f}")

    # Deep Learning
    log.info("Training MLP model...")
    mlp = MLPRegressor(
        app_state.cfg.behaviour_clone.mlp,
        app_state,
    )
    all_y_true, all_y_pred_mlp = run_cross_validation(
        mlp,
        data,
        n_splits=app_state.cfg.behaviour_clone.k_folds,
    )
    deep_per_action_metrics = calculate_per_action_metrics(
        all_y_true,
        all_y_pred_mlp,
        data.action_names,
    )
    log.info("Done training MLP Regression model.")
    for action_name, metrics in deep_per_action_metrics.items():
        log.info(f"  {action_name}: MAE={metrics['mae']:.6f}, Sign Acc={metrics['sign_acc']:.6f}")

    # Generate plots for each action
    log.info(f"Generating plots for {len(data.action_names)} actions...")
    for i, action_name in enumerate(data.action_names):
        create_single_action_scatter_plot(
            y_true=all_y_true[:, i],
            y_pred_linear=all_y_pred_linear[:, i],
            y_pred_deep=all_y_pred_mlp[:, i],
            action_name=action_name,
            baseline_metrics=baseline_per_action_metrics[action_name],
            linear_metrics=linear_per_action_metrics[action_name],
            deep_metrics=deep_per_action_metrics[action_name],
            output_dir=app_state.cfg.report.output_dir,
        )


@load_config(OfflineMainConfig)
def main(cfg: OfflineMainConfig):
    """Main function for finding the best observation period."""
    app_state, pipeline = create_standard_setup(cfg)

    pr, _ = load_offline_transitions(app_state, pipeline)
    if pr is None:
        log.info("No Pipereturn, exiting...")
        return

    if not pr.transitions:
        log.info("No Transitions, exiting...")
        return

    run_behaviour_cloning(app_state, pr.transitions)


if __name__ == "__main__":
    main()

import logging
from pathlib import Path

import numpy as np
from corerl.state import AppState
from lib_agent.buffer.datatypes import Trajectory
from lib_config.loader import load_config
from lib_defs.config_defs.tag_config import TagType
from lib_progress.tracker import track

from coreoffline.utils.behaviour_cloning.data import (
    ModelData,
    prepare_features_and_targets,
)
from coreoffline.utils.behaviour_cloning.evaluation import calculate_per_action_metrics
from coreoffline.utils.behaviour_cloning.models import BaseRegressor, LinearRegressor, MLPRegressor
from coreoffline.utils.behaviour_cloning.plotting import create_single_action_scatter_plot
from coreoffline.utils.config import OfflineMainConfig
from coreoffline.utils.data_loading import load_offline_trajectories
from coreoffline.utils.setup import create_standard_setup

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

    for train_data, test_data in track(data.k_fold_split(n_splits), total=n_splits):
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
    log.info(f"  Training Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

    return all_y_true, all_y_pred


def run_behaviour_cloning(app_state: AppState, trajectories: list[Trajectory]):
    assert isinstance(app_state.cfg, OfflineMainConfig)

    # Get all action names from configuration
    action_names = get_ai_setpoint_tag_names(app_state.cfg)
    log.info(f"Training models for {len(action_names)} action(s): {action_names}")

    data = prepare_features_and_targets(
        trajectories,
        action_names=action_names,
    )

    # Baseline model (copy-forward)
    log.info("Training baseline (copy-forward) model...")
    all_y_true, all_y_pred_baseline = run_baseline_cross_validation(
        data,
        n_splits=app_state.cfg.behaviour_clone.k_folds,
    )
    baseline_per_action_metrics = calculate_per_action_metrics(
        all_y_true,
        all_y_pred_baseline,
        data.action_names,
    )
    log.info("Baseline training complete")
    for action_name, metrics in baseline_per_action_metrics.items():
        log.info(f"  {action_name}: MAE={metrics['mae']:.6f}, Sign Acc={metrics['sign_acc']:.6f}")

    # Linear regression model
    log.info("Training linear regression model...")
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
    log.info("Linear regression training complete")
    for action_name, metrics in linear_per_action_metrics.items():
        log.info(f"  {action_name}: MAE={metrics['mae']:.6f}, Sign Acc={metrics['sign_acc']:.6f}")

    # MLP model
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
    log.info("MLP training complete")
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

    log.info("=" * 80)
    log.info("Behaviour cloning complete!")
    log.info(f"📊 Generated {len(data.action_names)} scatter plot(s)")
    output_path = Path(app_state.cfg.report.output_dir) / 'plots' / 'behaviour_clone'
    log.info(f"📁 Artifacts saved to: {output_path.resolve()}")
    log.info("=" * 80)


@load_config(OfflineMainConfig)
def main(cfg: OfflineMainConfig):
    """Main function for finding the best observation period."""
    log.info("=" * 80)
    log.info("Starting behaviour cloning")
    log.info("=" * 80)

    app_state, pipeline = create_standard_setup(cfg)

    log.info("Loading offline trajectories...")
    pr, _ = load_offline_trajectories(app_state, pipeline)
    if pr is None:
        log.info("No pipeline output found, exiting")
        return

    if not pr.trajectories:
        log.info("No trajectories found, exiting")
        return

    log.info(f"Loaded {len(pr.trajectories)} trajectories")

    run_behaviour_cloning(app_state, pr.trajectories)


if __name__ == "__main__":
    main()

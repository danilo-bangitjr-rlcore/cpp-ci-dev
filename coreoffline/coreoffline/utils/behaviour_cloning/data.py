import logging
from dataclasses import dataclass

import numpy as np
from lib_agent.buffer.datatypes import Trajectory
from sklearn.model_selection import KFold

log = logging.getLogger(__name__)


@dataclass
class ModelData:
    """Container for model training data."""
    X: np.ndarray  # Feature matrix (n_samples, n_features)
    y: np.ndarray  # Target matrix (n_samples, n_targets)
    ts: np.ndarray  # Timestamps array (n_samples,)
    action_names: list[str]  # Names of actions corresponding to y columns
    baseline_y: np.ndarray  # Previous actions matrix (n_samples, n_targets)

    def __post_init__(self):
        """Validate that action_names length matches target dimensions."""
        if len(self.action_names) != self.target_dim:
            raise ValueError(
                f"Length of action_names ({len(self.action_names)}) must match "
                f"target dimensions ({self.target_dim})",
            )

    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return len(self.X)

    @property
    def feature_dim(self) -> int:
        """Number of features."""
        return self.X.shape[1]

    @property
    def target_dim(self) -> int:
        """Number of target dimensions."""
        return self.y.shape[1]

    def k_fold_split(self, n_splits: int):
        """
        Generate k-fold train/test splits.
        """
        kf = KFold(n_splits=n_splits, shuffle=True)
        fold_indices = kf.split(self.X)

        # Generate train/test data for each fold
        for train_idx, test_idx in fold_indices:
            train_data = ModelData(
                X=self.X[train_idx],
                y=self.y[train_idx],
                ts=self.ts[train_idx],
                action_names=self.action_names,
                baseline_y=self.baseline_y[train_idx],
            )
            test_data = ModelData(
                X=self.X[test_idx],
                y=self.y[test_idx],
                ts=self.ts[test_idx],
                action_names=self.action_names,
                baseline_y=self.baseline_y[test_idx],
            )
            yield train_data, test_data


def prepare_features_and_targets(
    trajectories: list[Trajectory],
    action_names: list[str] | None = None,
):
    """
    Extract states and targets from trajectories for regression.
    """
    states = []
    targets = []
    time_stamps = []
    previous_actions = []

    for trajectory in trajectories:
        state = trajectory.state
        curr_action = trajectory.action
        prev_action = trajectory.prior.action  # Get previous action from prior step

        states.append(np.array(state))
        time_stamps.append(trajectory.steps[0].timestamp)
        targets.append(np.array(curr_action))
        previous_actions.append(np.array(prev_action))

    if len(states) == 0:
        msg = "No valid state-action pairs found"
        raise ValueError(msg)

    X = np.array(states)
    y = np.array(targets)
    ts = np.array(time_stamps)
    prev_a = np.array(previous_actions)

    # Ensure y has 2 dimensions and handle multi-dimensional actions by flattening
    y = y.reshape(y.shape[0], -1)
    prev_a = prev_a.reshape(prev_a.shape[0], -1)
    assert y.ndim == 2, f"Expected y to have 2 dimensions, got {y.ndim}"
    assert prev_a.ndim == 2, f"Expected prev_a to have 2 dimensions, got {prev_a.ndim}"

    # Generate default action names if not provided
    if action_names is None:
        action_names = [f"action_{i}" for i in range(y.shape[1])]

    log.info(
        f"State dimension: {X.shape[1]}, Target dimension: {y.shape[1]}",
    )

    return ModelData(X=X, y=y, ts=ts, action_names=action_names, baseline_y=prev_a)

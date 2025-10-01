import logging
from dataclasses import dataclass

import numpy as np
from corerl.data_pipeline.datatypes import Transition
from sklearn.model_selection import KFold

log = logging.getLogger(__name__)


@dataclass
class ModelData:
    """Container for model training data."""
    X: np.ndarray  # Feature matrix (n_samples, n_features)
    y: np.ndarray  # Target matrix (n_samples, n_targets)
    ts: np.ndarray  # Timestamps array (n_samples,)
    action_names: list[str]  # Names of actions corresponding to y columns

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
            )
            test_data = ModelData(
                X=self.X[test_idx],
                y=self.y[test_idx],
                ts=self.ts[test_idx],
                action_names=self.action_names,
            )
            yield train_data, test_data


def prepare_features_and_targets(
    transitions: list[Transition],
    action_names: list[str] | None = None,
):
    """
    Extract states and targets from transitions for regression.
    """
    states = []
    targets = []
    time_stamps = []

    for transition in transitions:
        state = transition.state
        curr_action = transition.action

        states.append(np.array(state))
        time_stamps.append(transition.steps[0].timestamp)
        targets.append(np.array(curr_action))

    if len(states) == 0:
        msg = "No valid state-action pairs found"
        raise ValueError(msg)

    X = np.array(states)
    y = np.array(targets)
    ts = np.array(time_stamps)

    # Ensure y has 2 dimensions and handle multi-dimensional actions by flattening
    y = y.reshape(y.shape[0], -1)
    assert y.ndim == 2, f"Expected y to have 2 dimensions, got {y.ndim}"

    # Generate default action names if not provided
    if action_names is None:
        action_names = [f"action_{i}" for i in range(y.shape[1])]

    log.info(
        f"State dimension: {X.shape[1]}, Target dimension: {y.shape[1]}",
    )

    return ModelData(X=X, y=y, ts=ts, action_names=action_names)

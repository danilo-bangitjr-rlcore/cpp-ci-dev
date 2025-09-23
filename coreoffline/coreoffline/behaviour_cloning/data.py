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
    y: np.ndarray  # Target matrix (n_samples, 1)
    ts: np.ndarray  # Timestamps array (n_samples,)

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
            )
            test_data = ModelData(
                X=self.X[test_idx],
                y=self.y[test_idx],
                ts=self.ts[test_idx],
            )
            yield train_data, test_data


def prepare_features_and_targets(
    transitions: list[Transition],
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

    log.info(
        f"State dimension: {X.shape[1]}, Target dimension: {y.shape[1]})",
    )

    return ModelData(X=X, y=y, ts=ts)

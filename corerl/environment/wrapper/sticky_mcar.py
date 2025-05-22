from copy import copy

import gymnasium
import numpy as np
from gymnasium import ObservationWrapper


class StickyMCARWrapper(ObservationWrapper):
    """
    Sticky Missing completely at random (MCAR) wrapper.
    """

    def __init__(self, env: gymnasium.Env, dropout_prob: float = 0.1, recovery_prob: float = 0.1):
        super().__init__(env)
        self.dropout = dropout_prob
        self.recovery = recovery_prob
        self.dropped: set[int] = set()

    def observation(self, observation: object):
        assert isinstance(observation, np.ndarray), "MCAR wrapper only supports numpy observations"

        # drop
        drop = np.random.random_sample(size=observation.shape) < self.dropout
        observation[drop] = np.nan
        self.dropped |= set(np.argwhere(drop).flatten().tolist())

        # recover
        for idx in copy(self.dropped):
            if np.random.random_sample() < self.recovery:
                # recover
                self.dropped.remove(idx)
            else:
                # continue to set to nan
                observation[idx] = np.nan

        return observation

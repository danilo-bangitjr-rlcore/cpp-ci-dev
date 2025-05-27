import gymnasium
import numpy as np
from gymnasium import ObservationWrapper


class MCARWrapper(ObservationWrapper):
    """
    Missing completely at random (MCAR) wrapper.
    """

    def __init__(self, env: gymnasium.Env, dropout_prob: float = 0.1):
        super().__init__(env)
        self.dropout = dropout_prob

    def observation(self, observation: object):
        assert isinstance(observation, np.ndarray), "MCAR wrapper only supports numpy observations"
        drop = np.random.random_sample(size=observation.shape) < self.dropout
        observation[drop] = np.nan
        return observation

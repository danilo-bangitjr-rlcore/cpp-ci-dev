import numpy as np
import gymnasium
import time

from omegaconf import DictConfig
from corerl.state_constructor.base import BaseStateConstructor
from corerl.interaction.normalizer import NormalizerInteraction

class TimedInteraction(NormalizerInteraction):
    """
    Interaction that will repeat an action for some length of time, while the observation is still
    updated more frequently
    """
    def __init__(self, cfg: DictConfig, env: gymnasium.Env, state_constructor: BaseStateConstructor):
        super().__init__(cfg, env, state_constructor)
        self.step_length = cfg.step_length  # how long reach step should take in seconds
        assert self.step_length > 0, "Step length should be greater than 0."
        self.obs_length = cfg.obs_length  # how often to update the observation
        assert self.obs_length > 0, "Step length should be greater than 0."

    def step(self, action: np.ndarray) -> tuple:
        step_end_time = time.time() + self.step_length
        while time.time() < step_end_time:
            obs_end_time = time.time() + self.obs_length
            transition = NormalizerInteraction.step(self, action)
            time.sleep(obs_end_time - time.time())

        return transition


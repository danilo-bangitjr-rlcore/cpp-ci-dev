import numpy as np
import gymnasium
import time

from omegaconf import DictConfig
from corerl.state_constructor.base import BaseStateConstructor
from corerl.interaction.normalizer import NormalizerInteraction
from corerl.data import Transition
from corerl.alerts.composite_alert import CompositeAlert

class TimedInteraction(NormalizerInteraction):
    """
    Interaction that will repeat an action for some length of time, while the observation is still
    updated more frequently
    """
    def __init__(
        self,
        cfg: DictConfig,
        env: gymnasium.Env,
        state_constructor: BaseStateConstructor,
        alerts: CompositeAlert
    ):
        super().__init__(cfg, env, state_constructor, alerts)

    def step(self, action: np.ndarray) -> tuple[list[Transition], list[dict]]:
        transitions = []
        env_info_list = []
        for obs_step in range(self.steps_per_decision):
            obs_end_time = time.time() + self.obs_length
            transitions_step, env_infos_step = NormalizerInteraction.step(self, action)
            # since transitions_step is returned from normalizer NormalizerInteraction.step,
            # it should only be a list of length 1
            assert len(transitions_step) == 1
            assert len(env_infos_step) == 1
            transitions += transitions_step
            env_info_list += env_infos_step
            time.sleep(obs_end_time - time.time())

        return transitions, env_info_list

from abc import ABC, abstractmethod

import numpy as np
from omegaconf import DictConfig
import gymnasium

from corerl.alerts.composite_alert import CompositeAlert
from corerl.state_constructor.base import BaseStateConstructor
from corerl.data.data import Transition

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from corerl.data.transition_creator import BaseTransitionCreator


class BaseInteraction(ABC):
    def __init__(self,
                 cfg: DictConfig,
                 env: gymnasium.Env,
                 state_constructor: BaseStateConstructor, **kwargs):

        self.env = env
        self.state_constructor = state_constructor
        self.timeout = cfg.timeout  # When timeout is set to 0, there is no timeout.
        self.internal_clock = 0

        self.steps_per_decision = cfg.steps_per_decision  # how many observation steps per decision step
        self.obs_length = cfg.obs_length  # how often to update the observation
        assert self.obs_length >= 0

    @abstractmethod
    # step returns a list of transitions and a list of environment infos
    def step(
        self,
        action: np.ndarray,
    ) -> tuple[list[Transition], list[Transition], list[Transition], list[Transition], dict, dict]:
        """
        Execute the action in the environment and transition to the next decision point
        Returns:
        - agent_transitions: List of all produced agent transitions
        - agent_train_transitions: List of agent transitions that didn't trigger an alert
        - alert_transitions : List of all produced alert transitions
        - alert_train_transitions: List of alert transitions that didn't trigger an alert
        - alert_info: Dictionary describing which types of alerts were/weren't triggered
        - env_info: Dictionary describing env info
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> tuple[np.ndarray, dict]:
        """
        Reset the environment and the state constructor
        """
        raise NotImplementedError

    @abstractmethod
    def warmup_sc(self, *args, **kwargs) -> None:
        """
        The state constructor warmup will be project specific.
        It will depend upon whether the environment is episodic/continuing.
        You might pass the recent history to the function and then loop self.state_constructor(obs)
        """
        raise NotImplementedError

    def env_counter(self) -> bool:
        """
        Check for episode timeout -> truncation
        """
        self.internal_clock += 1
        trunc = (self.timeout > 0) and (self.internal_clock % self.timeout == 0)
        return trunc

    @abstractmethod
    def init_alerts(self, composite_alert: CompositeAlert, alert_transition_creator: 'BaseTransitionCreator'):
        ...

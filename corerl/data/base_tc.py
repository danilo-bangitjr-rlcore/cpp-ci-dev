import numpy as np
from dataclasses import dataclass
from omegaconf import MISSING
from abc import ABC, abstractmethod

from corerl.alerts.base import BaseAlert
from corerl.data.data import Transition, ObsTransition
from corerl.state_constructor.base import BaseStateConstructor
from corerl.utils.hydra import interpolate, Group


@dataclass
class BaseTCConfig:
    name: str = MISSING

    gamma: float = interpolate('${experiment.gamma}')
    steps_per_decision: int = interpolate('${interaction.steps_per_decision}')
    n_step: int = interpolate('${interaction.n_step}')
    only_dp_transitions: bool = interpolate('${interaction.only_dp_transitions}')


class BaseTransitionCreator(ABC):
    def __init__(self, cfg: BaseTCConfig, state_constuctor: BaseStateConstructor) -> None:
        self.steps_per_decision: int = cfg.steps_per_decision
        self.n_step: int = cfg.n_step
        self.gamma: float = cfg.gamma
        self.alert: BaseAlert | None = None
        self.state_constructor = state_constuctor
        self.transition_kind: str = cfg.name

        # n_step = 0: bootstrap off state at next decision point
        # n_step > 0: bootstrap off state n steps into the future without crossing decision boundary
        if self.n_step == 0 or self.n_step >= self.steps_per_decision:
            self.queue_len = self.steps_per_decision
        else:
            self.queue_len = self.n_step

        self.curr_obs_transitions: list[ObsTransition] = []
        self.curr_states: list[np.ndarray] = []
        self.curr_dps: list[bool] = []
        self.curr_steps_until_decisions: list[int] = []

    def reset(self,
              state: np.ndarray,
              dp: bool,
              steps_until_decision: int) -> None:
        self.curr_obs_transitions = []
        self.curr_states = [state]
        self.curr_dps = [dp]
        self.curr_steps_until_decisions = [steps_until_decision]

    def feed(self,
             obs_transition: ObsTransition,
             next_state: np.ndarray,
             next_dp: bool,
             next_steps_until_decision: int) -> list[Transition]:

        self.curr_obs_transitions.append(obs_transition)
        self.curr_states.append(next_state)
        self.curr_dps.append(next_dp)
        self.curr_steps_until_decisions.append(next_steps_until_decision)

        transitions: list[Transition] = []
        if next_dp:
            assert len(self.curr_states) == len(self.curr_obs_transitions) + 1, \
                'Should be one more state than obs transition. Did you forget to call reset()?'

            transitions = self.make_decision_window_transitions()  # transitions is a list of anytime transitions
            self.curr_obs_transitions = []
            self.curr_states = [next_state]
            self.curr_dps = [next_dp]
            self.curr_steps_until_decisions = [next_steps_until_decision]

        return transitions

    @abstractmethod
    def make_decision_window_transitions(self) -> list[Transition]:
        raise NotImplementedError

    def init_alerts(self, alert: BaseAlert):
        self.alert = alert

    """
    #### Alert stuff below ####
    """

    def _get_alert_cumulants(self) -> list:
        """
        Get cumulants used to train alert value functions
        Currently passes the information required for Action-Value and GVF alerts.
        """

        assert self.alert is not None, "Alert not initialized, did you forget to call init_alerts()?"

        cumulants = []
        reward_list = [curr_obs_transition.reward for curr_obs_transition in self.curr_obs_transitions]
        next_obs_list = [curr_obs_transition.next_obs for curr_obs_transition in self.curr_obs_transitions]

        for i in range(len(reward_list)):
            cumulant_args = {"reward": reward_list[i], "obs": next_obs_list[i]}
            curr_cumulants = self.alert.get_cumulants(**cumulant_args)
            curr_cumulants = np.array(curr_cumulants)
            cumulants.append(curr_cumulants)

        return cumulants



tc_group = Group[
    [BaseStateConstructor],
    BaseTransitionCreator,
](['agent_transition_creator', 'alert_transition_creator'])

import numpy as np
import gymnasium
import time
from copy import deepcopy
from collections import deque

from omegaconf import DictConfig
from corerl.state_constructor.base import BaseStateConstructor
from corerl.interaction.normalizer import NormalizerInteraction
from corerl.alerts.composite_alert import CompositeAlert
from corerl.data.data import Transition, ObsTransition
from corerl.data.transition_creator import AnytimeTransitionCreator
from corerl.data_loaders.utils import _normalize


class AnytimeInteraction(NormalizerInteraction):
    """
    Interaction that will repeat an action for some length of time, while the
    observation is still updated more frequently
    """

    def __init__(
            self,
            cfg: DictConfig,
            env: gymnasium.Env,
            state_constructor: BaseStateConstructor,
            alerts: CompositeAlert,
            transition_creator: AnytimeTransitionCreator
    ):
        super().__init__(cfg, env, state_constructor, alerts, transition_creator)
        self.n_step = cfg.n_step
        self.warmup_steps = cfg.warmup_steps

    def step(self, action: np.ndarray) -> tuple[list[Transition], list[Transition], list[dict], list[dict]]:
        """
        Execute 'action' in the environment for a duration of self.steps_per_decision * self.obs_length
        A new obs/state is created every self.obs_length seconds
        Returns:
        - new_agent_transitions: List of all produced agent transitions
        - agent_train_transitions: List of Agent transitions that didn't trigger an alert
        - alert_train_transitions: List of Alert transitions that didn't trigger an alert
        - alert_info_list: List of dictionaries describing which types of alerts were/weren't triggered
        - env_info_list: List of dictionaries describing env info
        """
        obs_transitions = []
        alert_info_list = []
        env_info_list = []

        raw_action = self.action_normalizer.denormalize(action)

        trunc = False
        prev_decision_point = True
        # Execute 'action' for self.steps_per_decision steps
        for obs_step in range(self.steps_per_decision):
            out = self.env.step(raw_action)  # env.step() already ensures self.obs_length has elapsed
            raw_next_obs, raw_reward, term, env_trunc, env_info = out
            env_info_list.append(env_info)
            truncate = self.env_counter()  # use the interaction counter to decide reset. Remove reset in environment
            decision_point = (obs_step == self.steps_per_decision - 1)

            if obs_step == self.steps_per_decision - 1:
                next_obs_step = 0
            else:
                next_obs_step = obs_step + 1

            obs_transition = ObsTransition(
                self.raw_last_action,
                self.raw_last_obs,
                obs_step,
                prev_decision_point,
                raw_action,
                raw_reward,
                raw_next_obs,
                next_obs_step,
                decision_point,
                term,
                truncate,
                gap=False  # assume no data gap
            )

            obs_transition = _normalize(obs_transition, self)  # TODO: does this work?
            obs_transitions.append(obs_transition)

            reward = obs_transition.reward  # normalized reward
            alert_info = self.get_step_alerts(raw_action, action, self.last_state, raw_next_obs, reward)
            alert_info_list.append(alert_info)

            prev_decision_point = decision_point
            self.raw_last_obs = raw_next_obs
            self.raw_last_action = action

            if term or trunc:
                break

        # Create transitions
        transitions = self.transition_creator.make_transitions_for_chunk(obs_transitions, self.state_constructor,
                                                                         return_scs=False)
        self.last_state = transitions[-1].next_state

        # Only train on transitions where there weren't any alerts
        train_transitions = self.get_train_transitions(transitions, alert_info_list)

        return transitions, train_transitions, alert_info_list, env_info_list

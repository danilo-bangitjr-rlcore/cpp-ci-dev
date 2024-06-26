import numpy as np
import gymnasium
from omegaconf import DictConfig
from gymnasium.spaces.utils import flatdim
from collections import deque

from corerl.interaction.base import BaseInteraction

from corerl.state_constructor.base import BaseStateConstructor
from corerl.alerts.composite_alert import CompositeAlert
from corerl.data.data import Transition, ObsTransition
from corerl.data.transition_creator import AnytimeTransitionCreator
from corerl.data.obs_normalizer import ObsTransitionNormalizer


class NormalizerInteraction(BaseInteraction):
    def __init__(
            self,
            cfg: DictConfig,
            env: gymnasium.Env,
            state_constructor: BaseStateConstructor,
            alerts: CompositeAlert,
            transition_creator: AnytimeTransitionCreator,
            normalizer: ObsTransitionNormalizer
    ):
        super().__init__(cfg, env, state_constructor, alerts)

        self.transition_creator = transition_creator
        self.normalizer = normalizer  # will be used for normalizing observation transitions
        self.gamma = cfg.gamma
        self.last_state = None
        self.raw_last_obs = None
        self.raw_last_action = None
        self.action_dim = flatdim(env.action_space)

        # Alerts for a given transition only triggered in the future. Need to store transitions until then
        self.transition_queue = deque([])
        self.alert_transition_queue = deque([])

    def step(self, action: np.ndarray) -> tuple[list[Transition], list[Transition], list[dict], list[dict]]:
        """
        Execute the action in the environment and transition to the next decision point.
        Not 'Anytime' - Single observation/state per decision
        Returns:
        - transitions: List of all produced agent transitions
        - train_transitions: List of Agent transitions that didn't trigger an alert
        - alert_info_list: List of dictionaries describing which types of alerts were/weren't triggered
        - env_info_list: List of dictionaries describing env info
        """
        raw_action = self.normalizer.action_normalizer.denormalize(action)
        # Take step in the environment
        raw_next_obs, raw_reward, terminated, env_truncate, env_info = self.env.step(raw_action)
        truncate = self.env_counter()  # use the interaction counter to decide reset. Remove reset in environment
        env_info_list = [env_info]

        obs_transition = ObsTransition(
            self.raw_last_action,  # TODO if None do something
            self.raw_last_obs,
            0,
            True,
            action,
            raw_reward,
            raw_next_obs,
            0,
            True,
            False,  # assume a continuing env
            truncate,  # assume a continuing env
            gap=False  # no data gap
        )

        # next, use the transition creator to make the SINGLE transition for obs transition
        obs_transition = self.normalizer.normalize(obs_transition)
        obs_transitions = [obs_transition]
        transitions, _ = self.transition_creator.make_transitions_for_chunk(obs_transitions,
                                                                         self.state_constructor,
                                                                         return_scs=False)

        # Check to see if alerts should be triggered
        # TODO: should this be normalized?
        alert_info = self.get_step_alerts(raw_action, action, self.last_state,
                                          obs_transition.next_obs, obs_transition.reward)
        alert_info_list = [alert_info]

        # Only train on transitions where there weren't any alerts

        # TODO: add this back
        #train_transitions = self.get_train_transitions(transitions, alert_info_list)
        train_transitions = transitions

        assert len(transitions) == 1
        self.last_state = transitions[0].next_state
        self.raw_last_obs = raw_next_obs
        self.raw_last_action = raw_action

        return transitions, train_transitions, alert_info_list, env_info_list

    def get_step_alerts(self, raw_action, action, state, next_obs, reward) -> dict:
        """
        Determine if there is an alert triggered at the given state-action pair.
        Currently, passes the information required for Action-Value and GVF alerts.
        """
        alert_info = {}
        alert_info["raw_action"] = [raw_action]
        alert_info["action"] = [action]
        alert_info["state"] = [state]
        alert_info["next_obs"] = [next_obs]
        alert_info["reward"] = [reward]

        step_alert_info = self.alerts.evaluate(**alert_info)
        for key in step_alert_info:
            alert_info[key] = step_alert_info[key]

        return alert_info

    def reset(self) -> (np.ndarray, dict):
        """
        Reset the environment and the state constructor
        """
        self.transition_queue = deque([])
        raw_obs, info = self.env.reset()
        self.state_constructor.reset()

        obs = self.normalizer.obs_normalizer(raw_obs)
        dummy_action = np.zeros(self.action_dim)
        state = self.state_constructor(obs, dummy_action, initial_state=True, decision_point=True)

        self.raw_last_obs = raw_obs
        self.last_state = state
        self.raw_last_action = dummy_action  # TODO: does this make any sense here?

        return state, info

    def warmup_sc(self) -> None:
        """
        The state constructor warmup will be project specific.
        It will depend upon whether the environment is episodic/continuing.
        You might pass the recent history to the function and then loop self.state_constructor(obs)
        """
        pass

    def get_train_transitions(self, new_transitions, alert_info_list) -> list[Transition]:
        """
        Filter out agent transitions that triggered an alert.
        """
        if self.alerts.get_dim() > 0:
            train_transitions = []
            for j in range(len(new_transitions)):
                self.transition_queue.appendleft(new_transitions[j])
                if len(alert_info_list[j]["alert"].keys()) > 0:
                    transition = self.transition_queue.pop()
                    if alert_info_list[j]["composite_alert"] == [False]:
                        train_transitions.append(transition)

            return train_transitions
        else:
            return new_transitions

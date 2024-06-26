import numpy as np
import gymnasium

from gymnasium.spaces.utils import flatdim
from collections import deque

from omegaconf import DictConfig
from corerl.state_constructor.base import BaseStateConstructor
from corerl.interaction.base import BaseInteraction
from corerl.alerts.composite_alert import CompositeAlert
from corerl.data.data import Transition, ObsTransition
from corerl.data.transition_creator import AnytimeTransitionCreator
from corerl.data.obs_normalizer import ObsTransitionNormalizer


class AnytimeInteraction(BaseInteraction):
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
            transition_creator: AnytimeTransitionCreator,
            normalizer: ObsTransitionNormalizer):
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

        raw_action = self.normalizer.action_normalizer.denormalize(action)

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

            obs_transition = self.normalizer.normalize(obs_transition)
            obs_transitions.append(obs_transition)

            reward = obs_transition.reward  # normalized reward
            # TODO: should this be normalized?
            alert_info = self.get_step_alerts(raw_action, action, self.last_state, raw_next_obs, reward)
            alert_info_list.append(alert_info)

            prev_decision_point = decision_point
            self.raw_last_obs = raw_next_obs
            self.raw_last_action = action

            if term or trunc:
                break

        # Create transitions
        transitions, _ = self.transition_creator.make_transitions_for_chunk(obs_transitions,
                                                                            self.state_constructor,
                                                                            return_scs=False,
                                                                            start_state=self.last_state
                                                                            )

        # print('len_transitions', len(transitions))
        # print(transitions[-1])

        self.last_state = transitions[-1].next_state

        # Only train on transitions where there weren't any alerts
        train_transitions = self.get_train_transitions(transitions, alert_info_list)

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

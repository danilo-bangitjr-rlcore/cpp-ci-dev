import numpy as np
import gymnasium
from omegaconf import DictConfig
from gymnasium.spaces.utils import flatdim
from collections import deque

from corerl.interaction.base import BaseInteraction
from corerl.interaction.normalizer_utils import init_action_normalizer, init_reward_normalizer, init_obs_normalizer
from corerl.state_constructor.base import BaseStateConstructor
from corerl.alerts.composite_alert import CompositeAlert
from corerl.data import Transition


class NormalizerInteraction(BaseInteraction):
    def __init__(
            self,
            cfg: DictConfig,
            env: gymnasium.Env,
            state_constructor: BaseStateConstructor,
            alerts: CompositeAlert
    ):
        super().__init__(cfg, env, state_constructor, alerts)
        self.action_normalizer = init_action_normalizer(cfg.action_normalizer, self.env)
        self.reward_normalizer = init_reward_normalizer(cfg.reward_normalizer)
        self.obs_normalizer = init_obs_normalizer(cfg.obs_normalizer, env)
        self.gamma = cfg.gamma
        self.last_state = None
        self.last_obs = None
        self.action_dim = flatdim(env.action_space)

        self.agent_transition_queue = deque([])
        self.alert_transition_queue = deque([])

    def step(self, action: np.ndarray) -> tuple[list[Transition], list[dict]]:
        # Revan: I'm not sure that this is the best place for decision_point ^
        # also adding next_decision_point, which is whether the next state is a decision point.
        denormalized_action = self.action_normalizer.denormalize(action)

        next_obs, raw_reward, terminated, env_truncate, env_info = self.env.step(denormalized_action)
        normalized_next_obs = self.obs_normalizer(next_obs)
        next_state = self.state_constructor(normalized_next_obs, action)
        reward = self.reward_normalizer(raw_reward)
        truncate = self.env_counter()  # use the interaction counter to decide reset. Remove reset in environment
        gamma_exponent = 1

        curr_cumulants = self.get_cumulants(reward, normalized_next_obs)

        alert_info = self.get_step_alerts(denormalized_action, action, self.last_state, normalized_next_obs, reward)
        alert_info_list = [alert_info]

        agent_transition = Transition(
            self.last_obs,
            self.last_state,
            action,
            normalized_next_obs,
            next_state,
            reward,
            normalized_next_obs,  # the obs for bootstrapping is the same as the next obs here
            next_state,  # the state for bootstrapping is the same as the next state here
            terminated,
            truncate,
            True,  # always a decision point
            True,  # always a decision point
            gamma_exponent)

        new_agent_transitions = [agent_transition]

        step_alert_transitions = []
        alert_start_ind = 0
        for alert in self.alerts.alerts:
            print('alert', alert)
            alert_end_ind = alert_start_ind + alert.get_dim()
            print(curr_cumulants[alert_start_ind: alert_end_ind])

            # TODO: something is not working here
            alert_transition = Transition(
                self.last_obs,
                self.last_state,
                action,
                normalized_next_obs,
                next_state,
                curr_cumulants[alert_start_ind: alert_end_ind].item(),
                normalized_next_obs,  # the obs for bootstrapping is the same as the next obs here
                next_state,  # the state for bootstrapping is the same as the next state here
                terminated,
                truncate,
                True,  # always a decision point
                True,  # always a decision point
                gamma_exponent)

            step_alert_transitions.append(alert_transition)
            alert_start_ind = alert_end_ind

        new_alert_transitions = [step_alert_transitions]

        # Only train on transitions where there weren't any alerts
        agent_train_transitions = self.get_agent_train_transitions(new_agent_transitions, alert_info_list)
        alert_train_transitions = self.get_alert_train_transitions(new_alert_transitions, alert_info_list)

        self.last_state = next_state
        self.last_obs = next_obs

        return new_agent_transitions, agent_train_transitions, alert_train_transitions, alert_info_list, [env_info]

    def reset(self) -> (np.ndarray, dict):
        self.agent_transition_queue = deque([])
        self.alert_transition_queue = deque([])

        observation, info = self.env.reset()
        self.state_constructor.reset()

        normalized_observation = self.obs_normalizer(observation)
        dummy_action = np.zeros(self.action_dim)
        state = self.state_constructor(normalized_observation, dummy_action, initial_state=True, decision_point=True)

        self.last_obs = normalized_observation
        self.last_state = state

        return state, info

    def warmup_sc(self) -> None:
        """
        The state constructor warmup will be project specific.
        It will depend upon whether the environment is episodic/continuing.
        You might pass the recent history to the function and then loop self.state_constructor(obs)
        """
        pass

    def get_agent_train_transitions(self, new_agent_transitions, alert_info_list):
        """
        Filter out agent transitions that triggered an alert
        """
        if self.alerts.get_dim() > 0:
            agent_train_transitions = []
            for j in range(len(new_agent_transitions)):
                self.agent_transition_queue.appendleft(new_agent_transitions[j])
                if len(alert_info_list[j]["alert"].keys()) > 0:
                    agent_transition = self.agent_transition_queue.pop()
                    if alert_info_list[j]["composite_alert"] == [False]:
                        agent_train_transitions.append(agent_transition)

            return agent_train_transitions
        else:
            return new_agent_transitions

    def get_alert_train_transitions(self, new_alert_transitions, alert_info_list):
        """
        Filter out alert transitions that triggered an alert
        """
        if self.alerts.get_dim() > 0:
            alert_train_transitions = []
            for j in range(len(new_alert_transitions)):
                self.alert_transition_queue.appendleft(new_alert_transitions[j])
                if len(alert_info_list[j]["alert"].keys()) > 0:
                    alert_transition = self.alert_transition_queue.pop()
                    if alert_info_list[j]["composite_alert"] == [False]:
                        alert_train_transitions.append(alert_transition)

            return alert_train_transitions
        else:
            return new_alert_transitions

    def get_cumulants(self, reward, next_obs):
        cumulant_args = {}
        cumulant_args["reward"] = reward
        cumulant_args["obs"] = next_obs
        curr_cumulants = self.alerts.get_cumulants(**cumulant_args)
        curr_cumulants = np.array(curr_cumulants)

        return curr_cumulants

    def get_step_alerts(self, raw_action, action, state, next_obs, reward):
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

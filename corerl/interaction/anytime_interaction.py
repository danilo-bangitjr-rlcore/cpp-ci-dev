import numpy as np
import gymnasium

from gymnasium.spaces.utils import flatdim
from collections import deque
from copy import deepcopy

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

        self.n_step = cfg.n_step
        self.warmup_steps = cfg.warmup_steps

        self.alert_info_list = []
        self.only_dp_transitions = cfg.only_dp_transitions
        self.curr_decision_obs_transitions = []
        self.reward_sum = 0
        self.curr_decision_states = []
        self.prev_decision_point = True
        self.prev_steps_until_decision = self.steps_per_decision
        self.steps_until_decision = self.steps_per_decision - 1

    def step(self, action: np.ndarray) -> tuple[list[Transition], list[Transition], list[Transition], dict, dict]:
        """
        Execute 'action' in the environment for a duration of self.steps_per_decision * self.obs_length
        A new obs/state is created every self.obs_length seconds
        Returns:
        - transitions: List of all produced agent transitions
        - train_transitions: List of Agent transitions that didn't trigger an alert
        - alert_info: Dictionary describing which types of alerts were/weren't triggered
        - env_info: Dictionary describing env info
        """

        raw_action = self.normalizer.action_normalizer.denormalize(action)
        out = self.env.step(raw_action)  # env.step() already ensures self.obs_length has elapsed
        raw_next_obs, raw_reward, term, env_trunc, env_info = out
        truncate = self.env_counter()  # use the interaction counter to decide reset. Remove reset in environment
        decision_point = (self.steps_until_decision == self.steps_per_decision)

        obs_transition = ObsTransition(
            self.raw_last_action,
            self.raw_last_obs,
            self.prev_steps_until_decision,  # I don't think this variable is actually used
            self.prev_decision_point,
            raw_action,
            raw_reward,
            raw_next_obs,
            self.steps_until_decision,
            decision_point,
            term,
            truncate,
            gap=False  # assume no data gap
        )
        obs_transition = self.normalizer.normalize(obs_transition)

        self.reward_sum += obs_transition.reward
        next_state = self.state_constructor(obs_transition.next_obs,
                                            obs_transition.action,
                                            initial_state=False,  # we are not in reset(), so never an initial state
                                            decision_point=obs_transition.next_obs_dp,
                                            steps_until_decision=obs_transition.next_obs_steps_until_decision)

        # if (self.only_dp_transitions and decision_point) or (not self.only_dp_transitions):
        self.curr_decision_obs_transitions.append(obs_transition)
        self.curr_decision_states.append(next_state)

        # Still want to evaluate alerts each observation step
        reward = obs_transition.reward  # normalized reward
        next_obs = obs_transition.next_obs  # normalized next_obs
        alert_info = self.get_step_alerts(action, self.last_state, next_obs, reward)
        self.alert_info_list.append(alert_info)

        self.prev_decision_point = decision_point
        self.prev_steps_until_decision = self.steps_until_decision
        self.steps_until_decision -= 1
        if self.steps_until_decision == 0:
            self.steps_until_decision = self.steps_per_decision

        self.raw_last_obs = raw_next_obs
        self.raw_last_action = action
        self.last_state = next_state

        # Create transitions
        transitions, train_transitions = [], []  # NOTE: these lists may sometimes be empty if we are not at a decision point
        agent_train_transitions = []
        if decision_point:
            transitions = self.transition_creator.make_online_transitions(self.curr_decision_obs_transitions,
                                                                          self.curr_decision_states)

            # Only train on transitions where there weren't any alerts
            train_transitions = self.get_train_transitions(transitions, self.alert_info_list)

            # if we are only returning dp transitions, we need to update the transition we are returning
            # we take the first transition, since it starts at the last decision point and goes to the next
            if self.only_dp_transitions:
                # the first transition could have been filtered out by self.get_train_transitions(), so
                # transitions[0].state_dp checks if it was that original first transition returned by
                # self.transition_creator.make_online_transitions()
                if len(train_transitions) == 0:
                    agent_train_transitions = []

                elif train_transitions[0].state_dp:
                    transition = deepcopy(train_transitions[0])
                    # transition.gamma_exponent = 1
                    transition.next_obs = transition.boot_obs
                    transition.next_state_dp = transition.boot_state_dp
                    transition.next_state = transition.boot_state
                    transition.steps_until_decision = 1
                    transition.next_steps_until_decision = 1
                    transition.reward = transition.n_step_reward
                    agent_train_transitions = [transition]
            else:
                agent_train_transitions = train_transitions

            self.curr_decision_obs_transitions = []
            self.reward_sum = 0
            self.curr_decision_states = [self.last_state]
            self.alert_info_list = []

        alert_train_transitions = train_transitions
        return transitions, agent_train_transitions, alert_train_transitions, alert_info, env_info

    def get_step_alerts(self, action, state, next_obs, reward) -> dict:
        """
        Determine if there is an alert triggered at the given state-action pair.
        Currently, passes the information required for Action-Value and GVF alerts.
        """
        alert_info = {}
        alert_info["action"] = action
        alert_info["state"] = state
        alert_info["next_obs"] = next_obs
        alert_info["reward"] = reward

        step_alert_info = self.alerts.evaluate(**alert_info)

        return step_alert_info

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
        # Just set the first action to a dummy action
        # if you want, use warmup_sc() to set this to something else
        self.raw_last_action = dummy_action

        self.reward_sum = 0
        self.prev_decision_point = True
        self.prev_steps_until_decision = self.steps_per_decision
        self.steps_until_decision = self.steps_per_decision - 1
        self.curr_decision_states = [state]

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

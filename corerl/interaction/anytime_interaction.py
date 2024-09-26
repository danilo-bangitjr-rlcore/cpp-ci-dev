import numpy as np
import gymnasium

from gymnasium.spaces.utils import flatdim
from collections import deque
from typing import Optional

from omegaconf import DictConfig
from corerl.state_constructor.base import BaseStateConstructor
from corerl.interaction.base import BaseInteraction
from corerl.alerts.composite_alert import CompositeAlert
from corerl.data.data import Transition, OldObsTransition, ObsTransition
from corerl.data.obs_normalizer import ObsTransitionNormalizer

# this is to avoid circular imports for type checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corerl.data.transition_creator import OldAnytimeTransitionCreator, BaseTransitionCreator

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
            normalizer: ObsTransitionNormalizer,
            agent_transition_creator: "BaseTransitionCreator",
    ):
        super().__init__(cfg, env, state_constructor)

        self.agent_transition_creator = agent_transition_creator
        # Alerts for a given transition only triggered in the future. Need to store transitions until then
        self.agent_transition_queue = deque([])

        self.alerts = None
        self.normalizer = normalizer  # will be used for normalizing observation transitions
        self.gamma = cfg.gamma
        self.last_state = None
        self.raw_last_obs = None
        self.raw_last_action = None
        self.action_dim = flatdim(env.action_space)

        self.n_step = cfg.n_step
        self.warmup_steps = cfg.warmup_steps

        self.alert_info_list = []
        self.only_dp_transitions = cfg.only_dp_transitions
        self.steps_until_decision = self.steps_per_decision

    def init_alerts(self, alerts: CompositeAlert, alert_transition_creator: "BaseTransitionCreator"):
        self.alerts = alerts
        self.alert_transition_creator = alert_transition_creator
        self.alert_transition_queue = deque([])

    def step(self, action: np.ndarray) -> tuple[
        list[Transition], list[Transition], list[Transition], list[Transition], dict, dict]:
        """
        Execute 'action' in the environment for a duration of self.steps_per_decision * self.obs_length
        A new obs/state is created every self.obs_length seconds
        Returns:
        - agent_transitions: List of all produced agent transitions
        - agent_train_transitions: List of agent transitions that didn't trigger an alert
        - alert_transitions : List of all produced alert transitions
        - alert_train_transitions: List of alert transitions that didn't trigger an alert
        - alert_info: Dictionary describing which types of alerts were/weren't triggered
        - env_info: Dictionary describing env info
        """
        raw_action = self.normalizer.action_normalizer.denormalize(action)
        out = self.env.step(raw_action)  # env.step() already ensures self.obs_length has elapsed
        raw_next_obs, raw_reward, term, env_trunc, env_info = out
        truncate = self.env_counter()  # use the interaction counter to decide reset. Remove reset in environment
        decision_point = (self.steps_until_decision == self.steps_per_decision)

        obs_transition = ObsTransition(
            obs=self.raw_last_obs,
            action=raw_action,
            reward=float(raw_reward),
            next_obs=raw_next_obs,
            terminated=term,
            truncate=truncate,
            gap=False  # assume no data gap
        )

        obs_transition = self.normalizer.normalize(obs_transition)

        next_state = self.state_constructor(obs_transition.next_obs,
                                            obs_transition.action,
                                            initial_state=False,  # we are not in reset(), so never an initial state
                                            decision_point=decision_point,
                                            steps_until_decision=self.steps_until_decision)

        # transition will only return transitions at a decision point
        agent_transitions = self.agent_transition_creator.feed(obs_transition,
                                                               next_state,
                                                               next_dp=decision_point,
                                                               next_steps_until_decision=self.steps_until_decision)

        alert_transitions = []
        if self.alerts is not None:
            alert_transitions = self.alert_transition_creator.feed(obs_transition,
                                                                   next_state,
                                                                   next_dp=decision_point,
                                                                   next_steps_until_decision=self.steps_until_decision)

            assert len(agent_transitions) == len(alert_transitions) or len(agent_transitions) == 1

        self.steps_until_decision -= 1
        if self.steps_until_decision == 0:
            self.steps_until_decision = self.steps_per_decision

        self.raw_last_obs = raw_next_obs
        self.raw_last_action = action
        self.last_state = next_state

        alert_train_transitions = []
        agent_train_transitions = agent_transitions
        alert_info = {}

        if self.alerts is not None:
            alert_info = self.get_step_alerts(action, self.last_state,
                                              obs_transition.next_obs,  # the normalized next_obs
                                              obs_transition.reward)  # the normalized reward

            agent_train_transitions, alert_train_transitions = self._update_queues(agent_transitions,
                                                                                   alert_transitions,
                                                                                   alert_info)

        return_tuple = (agent_transitions, agent_train_transitions,
                        alert_transitions, alert_train_transitions,
                        alert_info, env_info)

        return return_tuple

    def reset(self) -> (np.ndarray, dict):
        """
        Reset the environment and the state constructor
        """
        self.agent_transition_queue = deque([])
        self.alert_transition_queue = deque([])
        raw_obs, info = self.env.reset()
        self.state_constructor.reset()

        obs = self.normalizer.obs_normalizer(raw_obs)
        dummy_action = np.zeros(self.action_dim)
        state = self.state_constructor(obs, dummy_action, initial_state=True, decision_point=True)

        self.agent_transition_creator.reset(state, dp=True, steps_until_decision=self.steps_until_decision)

        if self.alerts is not None:
            self.alert_transition_creator.reset(state, dp=True, steps_until_decision=self.steps_until_decision)

        self.raw_last_obs = raw_obs
        self.last_state = state
        # Just set the first action to a dummy action
        # if you want, use warmup_sc() to set this to something else
        self.raw_last_action = dummy_action

        self.steps_until_decision = self.steps_per_decision - 1
        if self.steps_until_decision == 0:
            self.steps_until_decision = self.steps_per_decision

        return state, info

    def warmup_sc(self) -> None:
        """
        The state constructor warmup will be project specific.
        It will depend upon whether the environment is episodic/continuing.
        You might pass the recent history to the function and then loop self.state_constructor(obs)
        """
        pass

    """
    ### The following functions are only for use with alerts ###
    """

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

    def _update_queues(self,
                       agent_transitions,
                       alert_transitions,
                       alert_info) -> tuple[list[Transition], list[Transition]]:

        # TODO this is ripe for a refactor to increase readability

        self.alert_info_list.append(alert_info)
        # update queues for alert transitions
        alert_train_transitions = []
        for j, alert_transition in enumerate(alert_transitions):
            self.alert_transition_queue.appendleft(alert_transition)
            if len(self.alert_info_list[j]["alert"].keys()) > 0:  # if an alert is ready...
                transition = self.alert_transition_queue.pop()  # pop the transition that alert was for
                if self.alert_info_list[j]["composite_alert"] == [False]:
                    alert_train_transitions.append(transition)

        # next, for agent transitions
        agent_train_transitions = []
        if len(agent_transitions) == 1:  # Regular RL
            for j in range(len(alert_transitions)):
                if j == 0:
                    self.agent_transition_queue.appendleft(agent_transitions[0])
                else:  # add None for the remaining transitions as placeholders in the queue
                    self.agent_transition_queue.appendleft(None)

                if len(self.alert_info_list[j]["alert"].keys()) > 0:  # if an alert is ready...
                    transition = self.agent_transition_queue.pop()  # pop the transition that alert was for
                    if self.alert_info_list[j]["composite_alert"] == [False] and transition is not None:
                        agent_train_transitions.append(transition)

        else:  # Anytime
            for j, agent_transition in enumerate(agent_transitions):
                self.agent_transition_queue.appendleft(agent_transition)
                if len(self.alert_info_list[j]["alert"].keys()) > 0:  # if an alert is ready...
                    transition = self.agent_transition_queue.pop()  # pop the transition that alert was for
                    if self.alert_info_list[j]["composite_alert"] == [False]:
                        agent_train_transitions.append(transition)

        if len(alert_transitions):
            self.alert_info_list = []

        return agent_train_transitions, alert_train_transitions


class OldAnytimeInteraction(BaseInteraction):
    """
    Interaction that will repeat an action for some length of time, while the
    observation is still updated more frequently
    """

    def __init__(
            self,
            cfg: DictConfig,
            env: gymnasium.Env,
            state_constructor: BaseStateConstructor,
            alerts: CompositeAlert,  # TODO: can I remove alerts from this?
            transition_creator: "AnytimeTransitionCreator",  # this is to avoid circular imports for type checking
            normalizer: ObsTransitionNormalizer):
        super().__init__(cfg, env, state_constructor)

        self.alerts = alerts
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

        obs_transition = OldObsTransition(
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

        next_state = self.state_constructor(obs_transition.next_obs,
                                            obs_transition.action,
                                            initial_state=False,  # we are not in reset(), so never an initial state
                                            decision_point=obs_transition.next_obs_dp,
                                            steps_until_decision=obs_transition.next_obs_steps_until_decision)

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
        if decision_point:
            transitions, alert_transitions, agent_transitions = self.transition_creator.make_decision_window_transitions(
                self.curr_decision_obs_transitions,
                self.curr_decision_states,
                filter_with_alerts=True,
                interaction=self)

            self.curr_decision_obs_transitions = []
            self.curr_decision_states = [self.last_state]
            self.alert_info_list = []
        else:
            # NOTE: these lists may sometimes be empty if we are not at a decision point
            transitions, alert_transitions, agent_transitions = [], [], []

        return transitions, agent_transitions, alert_transitions, alert_info, env_info

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

        self.prev_decision_point = True
        self.prev_steps_until_decision = self.steps_per_decision
        self.steps_until_decision = self.steps_per_decision - 1
        if self.steps_until_decision == 0:
            self.steps_until_decision = self.steps_per_decision
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

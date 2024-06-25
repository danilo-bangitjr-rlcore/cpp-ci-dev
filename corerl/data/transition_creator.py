import random
import numpy as np
from tqdm import tqdm
from collections import deque
from copy import deepcopy

from corerl.alerts.composite_alert import CompositeAlert
from corerl.interaction.normalizer import NormalizerInteraction
from corerl.data.data import ObsTransition, Transition
from corerl.state_constructor.base import BaseStateConstructor


class AnytimeTransitionCreator(object):
    def __init__(self, cfg, interaction: NormalizerInteraction, alerts: CompositeAlert):
        self.gamma = cfg.gamma  # gamma for the agent
        self.steps_per_decision = cfg.steps_per_decision
        self.n_step = cfg.n_step
        self.sc_warmup = cfg.sc_warmup
        self.interaction = interaction
        self.alerts = alerts
        self.alert_gammas = np.array(self.alerts.get_discount_factors())

    def make_offline_transitions(self, obs_transitions: list[ObsTransition], return_scs: bool = False):
        """
        Given a dataset of offline observation transitions, make the anytime transitions.
        """
        obs_transitions = deepcopy(obs_transitions)
        agent_transitions = []
        alert_transitions = []
        done = False
        transition_idx = 0
        # pbar = tqdm(total=len(obs_transitions))
        scs = []
        while not done:  # first, get transitions until a data gap
            curr_chunk_obs_transitions = []
            gap = False
            while not (gap or done):
                obs_transition = obs_transitions[transition_idx]
                curr_chunk_obs_transitions.append(obs_transition)
                gap = obs_transition.gap
                transition_idx += 1
                done = transition_idx == len(obs_transitions)
                # pbar.update(1)

            new_agent_transitions, new_alert_transitions, new_scs = self._make_offline_transitions_for_chunk(
                curr_chunk_obs_transitions, return_scs)

            agent_transitions += new_agent_transitions
            alert_transitions += new_alert_transitions
            scs += new_scs

        return agent_transitions, alert_transitions, scs

    def _make_offline_transitions_for_chunk(self, curr_chunk_obs_transitions: list[ObsTransition],
                                            return_scs: bool = False):
        """
        Produce Anytime transitions for a continuous chunk of observation transitions (no data gaps) from an offline dataset
        """
        sc = self.interaction.state_constructor
        sc.reset()

        curr_chunk_agent_transitions, curr_chunk_alert_transitions, new_scs = self.make_transitions_for_chunk(
            curr_chunk_obs_transitions, sc, return_scs)

        # Remove the transitions that were created during the state constructor warmup period
        curr_chunk_agent_transitions = curr_chunk_agent_transitions[self.sc_warmup:]
        curr_chunk_alert_transitions = curr_chunk_alert_transitions[self.sc_warmup:]

        assert len(curr_chunk_obs_transitions) == len(curr_chunk_agent_transitions) + self.sc_warmup

        if return_scs:
            new_scs = new_scs[self.sc_warmup:]
            assert len(new_scs) == len(curr_chunk_agent_transitions)

        return curr_chunk_agent_transitions, curr_chunk_alert_transitions, new_scs

    def make_transitions_for_chunk(self,
                                   curr_chunk_obs_transitions: list[ObsTransition],
                                   sc: BaseStateConstructor,
                                   return_scs: bool = False):
        """
        Produce Anytime transitions for a continuous chunk of observation transitions (no data gaps)
        """
        curr_chunk_agent_transitions = []
        curr_chunk_alert_transitions = []
        new_scs = []

        # Using ObsTransition.next_obs to create remaining states so creating first state with ObsTransition.obs
        first_obs_transition = deepcopy(curr_chunk_obs_transitions[0])
        first_obs_transition = _normalize(first_obs_transition, self.interaction)
        state = sc(first_obs_transition.obs,
                   first_obs_transition.prev_action,
                   initial_state=True,
                   decision_point=first_obs_transition.obs_dp,
                   steps_since_decision=first_obs_transition.obs_steps_since_decision)

        states = [state]
        # Produce remaining states and create list of transitions when decision points are encountered
        curr_decision_obs_transitions = []

        for obs_transition in curr_chunk_obs_transitions:
            obs_transition = _normalize(obs_transition, self.interaction)
            next_state = sc(obs_transition.next_obs,
                            obs_transition.action,
                            initial_state=False,
                            decision_point=obs_transition.next_obs_dp,
                            steps_since_decision=obs_transition.next_obs_steps_since_decision)

            states.append(next_state)
            curr_decision_obs_transitions.append(obs_transition)

            if return_scs:
                new_scs.append(deepcopy(sc))

            # If at a decision point, create list of transitions for the states observed since the last decision point
            # if steps_per_decision is 1, curr_decision_obs_transitions could be empty
            if obs_transition.next_obs_dp and len(curr_decision_obs_transitions):
                assert len(states) == len(curr_decision_obs_transitions) + 1

                new_agent_transitions, new_alert_transitions = self._make_decision_window_transitions(
                    curr_decision_obs_transitions, states)

                curr_chunk_agent_transitions += new_agent_transitions
                curr_chunk_alert_transitions += new_alert_transitions

                curr_decision_obs_transitions = []
                states = [next_state]

        return curr_chunk_agent_transitions, curr_chunk_alert_transitions, new_scs

    def get_cumulants(self, reward, next_obs) -> np.ndarray:
        """
        Get cumulants used to train alert value functions
        Currently passes the information required for Action-Value and GVF alerts.
        """
        cumulant_args = {}
        cumulant_args["reward"] = reward
        cumulant_args["obs"] = next_obs
        curr_cumulants = self.alerts.get_cumulants(**cumulant_args)
        curr_cumulants = np.array(curr_cumulants)

        return curr_cumulants

    def get_step_alerts(self, raw_action, action, state, next_obs, reward) -> dict:
        """
        Determine if there is an alert triggered at the given state-action pair.
        Currently passes the information required for Action-Value and GVF alerts.
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

    def update_n_step_cumulants(self, n_step_cumulant_q, new_cumulant, gammas) -> np.ndarray:
        """
        Recursively updating n-step cumulant
        """
        num_cumulants = len(new_cumulant)
        n_step_cumulant_q.appendleft([0.0 for _ in range(num_cumulants)])
        np_n_step_cumulants = np.array(n_step_cumulant_q)
        np_new_cumulant = np.array([new_cumulant for _ in range(len(n_step_cumulant_q))])
        np_n_step_cumulants = np_new_cumulant + (gammas * np_n_step_cumulants)

        return np_n_step_cumulants

    def _make_decision_window_transitions(self, curr_decision_obs_transitions, states):
        """
        Produce the agent and alert state transitions using the observation transitions that occur between two decision points
        """
        # Alerts can use different discount factors than the agent's value functions

        rewards = [curr_obs_transition.reward for curr_obs_transition in curr_decision_obs_transitions]
        next_obs_list = [curr_obs_transition.next_obs for curr_obs_transition in curr_decision_obs_transitions]

        cumulants = []
        for i in range(len(rewards)):
            curr_cumulants = self.get_cumulants(rewards[i], next_obs_list[i])
            cumulants.append(curr_cumulants)

        new_transitions = []

        # n_step = 0: bootstrap off state at next decision point
        # n_step > 0: bootstrap off state n steps into the future without crossing decision boundary
        if self.n_step == 0 or self.n_step >= self.steps_per_decision:
            n_step_rewards = deque([], self.steps_per_decision)
            n_step_cumulants = deque([], self.steps_per_decision)
            boot_state_queue = deque([], self.steps_per_decision)
            boot_obs_queue = deque([], self.steps_per_decision)
        else:
            n_step_rewards = deque([], self.n_step)
            n_step_cumulants = deque([], self.n_step)
            boot_state_queue = deque([], self.n_step)
            boot_obs_queue = deque([], self.n_step)

        boot_state_queue.appendleft(states[-1])
        boot_obs_queue.appendleft(curr_decision_obs_transitions[-1].next_obs)

        dp_counter = 1
        # Iteratively create the transitions by working backwards from the decision point
        for i in range(len(curr_decision_obs_transitions) - 1, -1, -1):
            curr_obs_transition = curr_decision_obs_transitions[i]
            obs = curr_obs_transition.obs
            state = states[i]
            action = curr_obs_transition.action
            reward = curr_obs_transition.reward
            s_dp = curr_obs_transition.obs_dp
            next_obs = curr_obs_transition.next_obs
            next_state = states[i+1]
            cumulant = cumulants[i]
            term = curr_obs_transition.terminated
            trunc = curr_obs_transition.truncate

            # Create Agent Transition
            np_n_step_rewards = self.update_n_step_cumulants(n_step_rewards, np.array([reward]), self.gamma)

            # Shared amongst agent and alert transitions
            gamma_exp = len(np_n_step_rewards)
            ns_dp = dp_counter <= boot_state_queue.maxlen

            if self.alerts.get_dim() > 0:
                np_n_step_cumulants = self.update_n_step_cumulants(n_step_cumulants, cumulant, self.alert_gammas)
            else:
                np_n_step_cumulants = np.zeros(0)

            transition = Transition(
                obs,
                state,
                action,
                next_obs,  # the immediate next obs
                next_state,  # the immediate next state
                np_n_step_rewards[-1].item(),
                np_n_step_cumulants[-1],  # the array of cumulants
                boot_obs_queue[-1],  # the obs we bootstrap off
                boot_state_queue[-1],  # the state we bootstrap off
                term,
                trunc,
                s_dp,
                ns_dp,
                gamma_exp)

            new_transitions.append(transition)

            # Update queues and counters
            dp_counter += 1
            boot_state_queue.appendleft(state)
            boot_obs_queue.appendleft(obs)
            n_step_rewards = deque(np_n_step_rewards, n_step_rewards.maxlen)
            n_step_cumulants = deque(np_n_step_cumulants, n_step_cumulants.maxlen)

        new_transitions.reverse()

        return new_transitions


def _normalize(obs_transition, interaction):
    obs_transition.prev_action = interaction.action_normalizer(obs_transition.prev_action)
    obs_transition.obs = interaction.obs_normalizer(obs_transition.obs)
    obs_transition.action = interaction.action_normalizer(obs_transition.action)
    obs_transition.next_obs = interaction.obs_normalizer(obs_transition.next_obs)
    obs_transition.reward = interaction.reward_normalizer(obs_transition.reward)

    return obs_transition

import numpy as np
from tqdm import tqdm
from collections import deque
from copy import deepcopy
from typing import Optional

from corerl.alerts.composite_alert import CompositeAlert
from corerl.data.data import ObsTransition, Transition, Trajectory
from corerl.state_constructor.base import BaseStateConstructor
from corerl.interaction.anytime_interaction import AnytimeInteraction


class AnytimeTransitionCreator(object):
    def __init__(self, cfg, alerts: CompositeAlert):
        self.gamma = cfg.gamma  # gamma for the agent
        self.steps_per_decision = cfg.steps_per_decision
        self.n_step = cfg.n_step
        self.alerts = alerts
        self.alert_gammas = np.array(self.alerts.get_discount_factors())
        self.only_dp_transitions = cfg.only_dp_transitions

    def set_only_dp_transitions(self, only_dp_transitions: bool) -> None:
        self.only_dp_transitions = only_dp_transitions

    def make_offline_trajectories(self,
                                  obs_transitions: list[ObsTransition],
                                  sc: BaseStateConstructor,
                                  return_scs: bool = False,
                                  warmup: int = 0,
                                  use_pbar: bool = False) -> list[Trajectory]:

        obs_transitions = deepcopy(obs_transitions)
        trajectories = []
        done = False
        transition_idx = 0
        if use_pbar:
            pbar = tqdm(total=len(obs_transitions))

        while not done:  # first, get transitions until a data gap
            curr_chunk_obs_transitions = []
            gap = False
            while not (gap or done):
                obs_transition = obs_transitions[transition_idx]
                curr_chunk_obs_transitions.append(obs_transition)
                gap = obs_transition.gap
                transition_idx += 1
                done = transition_idx == len(obs_transitions)
                if use_pbar:
                    pbar.update(1)

            curr_chunk_transitions, _, new_scs = self._make_offline_transitions_for_chunk(curr_chunk_obs_transitions,
                                                                                          sc, return_scs, warmup)

            new_traj = Trajectory()
            for i in range(len(curr_chunk_transitions)):
                new_traj.add_transition(curr_chunk_transitions[i])
                if return_scs:
                    new_traj.add_sc(new_scs[i])

            trajectories.append(new_traj)

        return trajectories

    def make_offline_transitions(self,
                                 obs_transitions: list[ObsTransition],
                                 sc: BaseStateConstructor,
                                 return_scs: bool = False,
                                 warmup: int = 0,
                                 use_pbar: bool = False) -> tuple[
        list[Transition], list[Transition], list[BaseStateConstructor]]:
        """
        Given a dataset of offline observation transitions, make the anytime transitions.
        """
        obs_transitions = deepcopy(obs_transitions)

        agent_transitions = []
        alert_transitions = []
        done = False
        transition_idx = 0
        if use_pbar:
            pbar = tqdm(total=len(obs_transitions))
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
                if use_pbar:
                    pbar.update(1)

            curr_chunk_agent_transitions, curr_chunk_alert_transitions, new_scs = self._make_offline_transitions_for_chunk(
                curr_chunk_obs_transitions,
                sc, return_scs, warmup)
            agent_transitions += curr_chunk_agent_transitions
            alert_transitions += curr_chunk_alert_transitions
            scs += new_scs

        return agent_transitions, alert_transitions, scs

    def _make_offline_transitions_for_chunk(self,
                                            curr_chunk_obs_transitions: list[ObsTransition],
                                            sc: BaseStateConstructor,
                                            return_scs: bool = False,
                                            warmup: int = 0) -> tuple[
        list[Transition], list[Transition], list[BaseStateConstructor]]:
        """
        Produce Anytime transitions for a continuous chunk of observation transitions (no data gaps) from an offline dataset
        """
        for i in range(len(curr_chunk_obs_transitions) - 1):
            assert np.allclose(curr_chunk_obs_transitions[i].next_obs, curr_chunk_obs_transitions[i + 1].obs)
            assert not curr_chunk_obs_transitions[i].gap

        sc.reset()
        curr_chunk_agent_transitions = []
        curr_chunk_alert_transitions = []

        # Using ObsTransition.next_obs to create remaining states so creating first state with ObsTransition.obs
        first_obs_transition = deepcopy(curr_chunk_obs_transitions[0])
        start_state = sc(first_obs_transition.obs,
                         first_obs_transition.prev_action,
                         initial_state=True,
                         decision_point=first_obs_transition.obs_dp,
                         steps_since_decision=first_obs_transition.obs_steps_since_decision)

        new_scs = [deepcopy(sc)]
        states = [start_state]
        curr_decision_obs_transitions = []

        # Produce remaining states and create list of transitions when decision points are encountered
        for idx, obs_transition in enumerate(curr_chunk_obs_transitions):
            # assume observation transitions are normalized
            next_state = sc(obs_transition.next_obs,
                            obs_transition.action,
                            initial_state=False,
                            decision_point=obs_transition.next_obs_dp,
                            steps_since_decision=obs_transition.next_obs_steps_since_decision)

            states.append(next_state)
            curr_decision_obs_transitions.append(obs_transition)

            # if we are not only returning dp transitions, always append the new state constructor
            if return_scs and not self.only_dp_transitions:
                new_scs.append(deepcopy(sc))

            # If at a decision point, create list of transitions for the states observed since the last decision point
            # If steps_per_decision is 1, curr_decision_obs_transitions could be empty
            if obs_transition.next_obs_dp and len(curr_decision_obs_transitions):
                assert len(states) == len(curr_decision_obs_transitions) + 1
                transitions, _, agent_transitions = self.make_decision_window_transitions(curr_decision_obs_transitions,
                                                                                          states)

                # append the state constructors here if we are only returning dp transitions
                if return_scs and self.only_dp_transitions:
                    new_scs.append(deepcopy(sc))

                curr_chunk_agent_transitions += agent_transitions
                curr_chunk_alert_transitions += transitions
                curr_decision_obs_transitions = []
                states = [next_state]

        # Remove the transitions that were created during the state constructor warmup period
        if self.only_dp_transitions:
            agent_warmup = warmup // self.steps_per_decision
        else:
            agent_warmup = warmup

        curr_chunk_agent_transitions = curr_chunk_agent_transitions[agent_warmup:]
        curr_chunk_alert_transitions = curr_chunk_alert_transitions[warmup:]

        if return_scs:
            new_scs = new_scs[agent_warmup:-1]
            assert len(new_scs) == len(curr_chunk_agent_transitions)
            # check to see if scs are lined up to transitions
            for i in range(len(curr_chunk_agent_transitions)):
                assert np.allclose(curr_chunk_agent_transitions[i].state, new_scs[i].get_current_state())

        return curr_chunk_agent_transitions, curr_chunk_alert_transitions, new_scs

    def make_decision_window_transitions(self,
                                         curr_decision_obs_transitions: list[ObsTransition],
                                         curr_decision_states: list[np.ndarray],
                                         filter_with_alerts: bool = False,
                                         interaction: Optional[AnytimeInteraction] = None
                                         ) -> tuple[list[Transition], list[Transition], list[Transition]]:

        assert len(curr_decision_states) == len(curr_decision_obs_transitions) + 1
        action = curr_decision_obs_transitions[0].action
        for obs_transition in curr_decision_obs_transitions:
            assert np.allclose(obs_transition.action, action)

        # transitions is a list of ALL transitions
        transitions = self._make_decision_window_transitions(curr_decision_obs_transitions, curr_decision_states)

        # next, we may remove some transitions that were anomalous/triggered an alert.
        if filter_with_alerts:
            assert interaction is not None
            filtered_transitions = interaction.get_train_transitions(transitions, interaction.alert_info_list)
        else:
            filtered_transitions = transitions

        # next, we may modify the transitions depending on whether the agent is only training with decision point transitions
        # i.e. "regular RL"
        if self.only_dp_transitions:
            # we may have filtered out all transitions for the agent, so return an empty list in this case
            if len(filtered_transitions) == 0:
                agent_transitions = []
            # check to see if the first transition starts at a decision point. if so, we will modify this
            # transition to create our decision point to decision point transition
            elif filtered_transitions[0].state_dp:
                transition = deepcopy(filtered_transitions[0])
                transition.next_obs = transition.boot_obs
                transition.next_state_dp = transition.boot_state_dp
                transition.next_state = transition.boot_state
                transition.steps_since_decision = 1
                transition.next_steps_since_decision = 1
                transition.reward = transition.n_step_reward
                agent_transitions = [transition]
            # we filtered out the transition that starts at a decision point. return an emtpy list in this case
            else:
                agent_transitions = []
        else:
            agent_transitions = filtered_transitions

        # typically we will use transitions for loggins, filtered_transitions for alerts, and agent_transitions for training
        return transitions, filtered_transitions, agent_transitions

    def get_cumulants(self, reward: float, next_obs: np.ndarray) -> np.ndarray:
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

    def _make_decision_window_transitions(self,
                                          curr_decision_obs_transitions: list[ObsTransition],
                                          states: list[np.ndarray]) -> list[Transition]:
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
            state_dp = curr_obs_transition.obs_dp
            next_obs = curr_obs_transition.next_obs
            next_state = states[i + 1]
            next_state_dp = curr_obs_transition.next_obs_dp
            cumulant = cumulants[i]
            term = curr_obs_transition.terminated
            trunc = curr_obs_transition.truncate
            gap = curr_obs_transition.gap
            steps_since_decision = curr_obs_transition.obs_steps_since_decision
            next_steps_since_decision = curr_obs_transition.next_obs_steps_since_decision

            # Create Agent Transition
            np_n_step_rewards = self.update_n_step_cumulants(n_step_rewards, np.array([reward]), self.gamma)

            # Shared amongst agent and alert transitions
            gamma_exp = len(np_n_step_rewards)
            boot_state_dp = dp_counter <= boot_state_queue.maxlen

            if self.alerts.get_dim() > 0:
                np_n_step_cumulants = self.update_n_step_cumulants(n_step_cumulants, cumulant, self.alert_gammas)
                cumulant_i = np_n_step_cumulants[-1]
            else:
                np_n_step_cumulants = np.zeros(0)
                cumulant_i = np_n_step_cumulants

            transition = Transition(
                obs,
                state,
                action,
                next_obs,  # the immediate next obs
                next_state,  # the immediate next state
                reward,  # the reward
                np_n_step_rewards[-1].item(),
                cumulant_i,  # the array of cumulants
                boot_obs_queue[-1],  # the obs we bootstrap off
                boot_state_queue[-1],  # the state we bootstrap off
                term,
                trunc,
                state_dp,
                next_state_dp,
                boot_state_dp,
                gamma_exp,
                gap,
                steps_since_decision,
                next_steps_since_decision
            )

            new_transitions.append(transition)

            # Update queues and counters
            dp_counter += 1
            boot_state_queue.appendleft(state)
            boot_obs_queue.appendleft(obs)
            n_step_rewards = deque(np_n_step_rewards, n_step_rewards.maxlen)
            n_step_cumulants = deque(np_n_step_cumulants, n_step_cumulants.maxlen)

        new_transitions.reverse()

        return new_transitions

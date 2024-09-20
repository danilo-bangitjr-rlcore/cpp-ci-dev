import numpy as np
from tqdm import tqdm
from collections import deque
from omegaconf import DictConfig
from abc import ABC, abstractmethod

from copy import deepcopy
from typing import Optional

from corerl.alerts.composite_alert import CompositeAlert
from corerl.data.data import OldObsTransition, Transition, Trajectory, ObsTransition
from corerl.state_constructor.base import BaseStateConstructor
from corerl.interaction.anytime_interaction import AnytimeInteraction


class OldAnytimeTransitionCreator(object):
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
                                  obs_transitions: list[OldObsTransition],
                                  sc: BaseStateConstructor,
                                  return_all_scs: bool = False,
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

            curr_chunk_transitions, _, start_sc = self._make_offline_transitions_for_chunk(curr_chunk_obs_transitions,
                                                                                           sc, warmup)

            if len(curr_chunk_transitions) > 0:
                new_traj = Trajectory()
                new_traj.add_start_sc(start_sc)
                for i in range(len(curr_chunk_transitions)):
                    new_traj.add_transition(curr_chunk_transitions[i])

                if return_all_scs:
                    new_traj.cache_scs()
                    assert len(new_traj.scs) == new_traj.num_transitions
                    for i in range(len(new_traj.scs)):
                        assert np.allclose(new_traj.scs[i].get_current_state(), new_traj.transitions[i].state)

                trajectories.append(new_traj)

        return trajectories

    def make_offline_transitions(self,
                                 obs_transitions: list[OldObsTransition],
                                 sc: BaseStateConstructor,
                                 warmup: int = 0,
                                 use_pbar: bool = False) -> tuple[list[Transition], list[Transition]]:
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

            curr_chunk_agent_transitions, curr_chunk_alert_transitions, _ = self._make_offline_transitions_for_chunk(
                curr_chunk_obs_transitions,
                sc, warmup)

            agent_transitions += curr_chunk_agent_transitions
            alert_transitions += curr_chunk_alert_transitions

        return agent_transitions, alert_transitions

    def _make_offline_transitions_for_chunk(self,
                                            curr_chunk_obs_transitions: list[OldObsTransition],
                                            sc: BaseStateConstructor,
                                            warmup: int = 0) -> tuple[
        list[Transition], list[Transition], BaseStateConstructor]:
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
                         steps_until_decision=first_obs_transition.obs_steps_until_decision)

        states = [start_state]
        curr_decision_obs_transitions = []
        warmup_sc = None

        # Produce remaining states and create list of transitions when decision points are encountered
        for idx, obs_transition in enumerate(curr_chunk_obs_transitions):
            # assume observation transitions are normalized
            if idx == warmup:
                warmup_sc = deepcopy(sc)  # the state constructor immediately after the warmup period

            next_state = sc(obs_transition.next_obs,
                            obs_transition.action,
                            initial_state=False,
                            decision_point=obs_transition.next_obs_dp,
                            steps_until_decision=obs_transition.next_obs_steps_until_decision)

            states.append(next_state)
            curr_decision_obs_transitions.append(obs_transition)

            # If at a decision point, create list of transitions for the states observed since the last decision point
            # If steps_per_decision is 1, curr_decision_obs_transitions could be empty
            if obs_transition.next_obs_dp and len(curr_decision_obs_transitions):
                assert len(states) == len(curr_decision_obs_transitions) + 1
                transitions, _, agent_transitions = self.make_decision_window_transitions(curr_decision_obs_transitions,
                                                                                          states)
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

        if len(curr_chunk_alert_transitions) == 0:
            warmup_sc = None
        else:
            # assert np.allclose(warmup_sc.get_current_state(), curr_chunk_agent_transitions[0].state)
            assert np.allclose(warmup_sc.get_current_state(), curr_chunk_alert_transitions[0].state)

        return curr_chunk_agent_transitions, curr_chunk_alert_transitions, warmup_sc

    def make_decision_window_transitions(self,
                                         curr_decision_obs_transitions: list[OldObsTransition],
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
        agent_transitions = []
        if self.only_dp_transitions:
            # we may have filtered out all transitions for the agent, so return an empty list in this case
            if len(filtered_transitions) == 0:
                agent_transitions = []
            # check to see if the first transition starts at a decision point. if so, we will modify this
            # transition to create our decision point to decision point transition
            elif filtered_transitions[0].state_dp:
                transition = deepcopy(filtered_transitions[0])
                if transition.gamma_exponent == self.steps_per_decision:
                    transition.steps_until_decision = self.steps_per_decision
                    transition.next_steps_until_decision = self.steps_per_decision - 1
                    transition.boot_steps_until_decision = self.steps_per_decision
                    # NOTE, if the last transition was filtered out, this last line may not be correct
                    transition.gap = filtered_transitions[-1].gap

                    agent_transitions = [transition]
            # we filtered out the transition that starts at a decision point. return an emtpy list in this case
            else:
                agent_transitions = []
        else:
            agent_transitions = filtered_transitions

        # typically we will use transitions for logging, filtered_transitions for alerts, and agent_transitions for training
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
                                          curr_decision_obs_transitions: list[OldObsTransition],
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
            steps_until_decision = curr_obs_transition.obs_steps_until_decision
            next_steps_until_decision = curr_obs_transition.next_obs_steps_until_decision

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
                steps_until_decision,
                next_steps_until_decision
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


def _get_n_step_cumulants(cumulant_queue: deque, gammas: np.ndarray) -> np.ndarray:
    """
    Given a queue of cumulants, return the n step cumulant where n is determined by the length of the queue
    """
    n_step_cumulants = np.zeros_like(cumulant_queue[0])
    boot_len = len(cumulant_queue)
    for step in range(boot_len):
        cumulant_step = cumulant_queue[step]
        n_step_cumulants += (gammas ** step) * cumulant_step
    return n_step_cumulants


def _get_n_step_reward(reward_queue: deque, gamma: float) -> float:
    """
    Given a queue of rewards, return the n step cumulant where n is determined by the length of the queue
    """
    n_step_reward = 0
    boot_len = len(reward_queue)
    for step in range(boot_len):
        cumulant_step = reward_queue[step]
        n_step_reward += (gamma ** step) * cumulant_step
    return n_step_reward


class BaseTransitionCreator(ABC):
    def __init__(self, cfg: DictConfig, state_constuctor: BaseStateConstructor) -> None:
        self.state = None
        self.steps_per_decision = cfg.steps_per_decision
        self.n_step = cfg.n_step
        self.gamma = cfg.gamma
        self.alert = None
        self.state_constructor = state_constuctor

        # n_step = 0: bootstrap off state at next decision point
        # n_step > 0: bootstrap off state n steps into the future without crossing decision boundary
        if self.n_step == 0 or self.n_step >= self.steps_per_decision:
            self.queue_len = self.steps_per_decision
        else:
            self.queue_len = self.n_step

        self.curr_obs_transitions = []
        self.curr_states = []
        self.curr_dps = []
        self.curr_steps_until_decisions = []

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

        transitions = []
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

    def init_alerts(self, alert):
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


class AnytimeTransitionCreator(BaseTransitionCreator):
    def make_decision_window_transitions(self) -> list[Transition]:
        """
        Produce the agent and alert state transitions using the observation transitions
        that occur between two decision points
        """
        using_alerts = self.alert is not None and self.alert.get_dim() > 0
        alert_gammas, cumulants = None, None
        if using_alerts:
            # Alerts can use different discount factors than the agent's value functions
            alert_gammas = np.array(self.alert.get_discount_factors())
            cumulants = self._get_alert_cumulants()

        boot_state_queue = deque([], self.queue_len)
        boot_obs_queue = deque([], self.queue_len)
        boot_dp_queue = deque([], self.queue_len)
        reward_queue = deque([], self.queue_len)
        cumulant_queue = deque([], self.queue_len)

        boot_state_queue.appendleft(self.curr_states[-1])
        boot_obs_queue.appendleft(self.curr_obs_transitions[-1].next_obs)
        boot_dp_queue.appendleft(self.curr_dps[-1])

        new_transitions = []
        # Iteratively create the transitions by working backwards from the decision point
        for step_idx in range(len(self.curr_obs_transitions) - 1, -1, -1):
            curr_obs_transition = self.curr_obs_transitions[step_idx]
            reward = curr_obs_transition.reward
            reward_queue.appendleft(reward)
            cumulant_queue.appendleft(cumulants[step_idx])
            n_step_reward = _get_n_step_reward(reward_queue, self.gamma)
            if using_alerts:
                n_step_cumulants = _get_n_step_cumulants(cumulant_queue, alert_gammas)
            else:
                n_step_cumulants = None

            state = self.curr_states[step_idx]
            obs = self.curr_obs_transitions[step_idx].obs
            state_dp = self.curr_dps[step_idx]

            transition = Transition(
                obs=obs,
                state=state,
                action=curr_obs_transition.action,
                next_obs=curr_obs_transition.next_obs,
                next_state=self.curr_states[step_idx + 1],
                reward=curr_obs_transition.reward,
                n_step_reward=n_step_reward,
                n_step_cumulants=n_step_cumulants,
                boot_obs=boot_obs_queue[-1],
                boot_state=boot_state_queue[-1],
                terminated=curr_obs_transition.terminated,
                truncate=curr_obs_transition.truncate,
                state_dp=state_dp,
                next_state_dp=self.curr_dps[step_idx + 1],
                boot_state_dp=boot_dp_queue[-1],
                gamma_exponent=self.curr_steps_until_decisions[step_idx],
                gap=curr_obs_transition.gap,
                steps_until_decision=self.curr_steps_until_decisions[step_idx],
                next_steps_until_decision=self.curr_steps_until_decisions[step_idx + 1],
            )
            new_transitions.append(transition)
            boot_state_queue.appendleft(state)
            boot_obs_queue.appendleft(obs)
            boot_dp_queue.appendleft(state_dp)

        new_transitions.reverse()
        return new_transitions


class RegularRLTransitionCreator(BaseTransitionCreator):

    def make_decision_window_transitions(self) -> list[Transition]:
        """
        Produce the agent and alert state transitions using the observation transitions
        that occur between two decision points.
        """

        if len(self.curr_obs_transitions) < self.steps_per_decision:
            return []
        elif len(self.curr_obs_transitions) > self.steps_per_decision:
            assert False, "There should not be more than self.steps_per_decision obs transitions in len(self.curr_obs_transitions)"

        using_alerts = self.alert is not None and self.alert.get_dim() > 0
        n_step_cumulants = None
        if using_alerts:
            alert_gammas = np.array(
                self.alert.get_discount_factors())  # Alerts can use different discount factors than the agent's value functions
            cumulants = self._get_alert_cumulants()
            cumulant_queue = deque(cumulants)
            n_step_cumulants = _get_n_step_cumulants(cumulant_queue, alert_gammas)

        rewards = [obs_transition.reward for obs_transition in self.curr_obs_transitions]
        reward_queue = deque(rewards)
        n_step_reward = _get_n_step_reward(reward_queue, self.gamma)

        first_obs_transition = self.curr_obs_transitions[0]
        last_obs_transition = self.curr_obs_transitions[-1]
        first_state = self.curr_states[0]
        next_state = self.curr_states[1]
        last_state = self.curr_states[-1]

        first_dp = self.curr_dps[0]
        next_dp = self.curr_dps[1]
        last_dp = self.curr_dps[-1]
        assert first_dp
        assert last_dp

        gamma_exp = len(self.curr_obs_transitions)
        first_step_until_decision = self.curr_steps_until_decisions[0]
        assert first_step_until_decision == self.steps_per_decision

        transition = Transition(
            obs=first_obs_transition.obs,
            state=first_state,
            action=first_obs_transition.action,
            next_obs=first_obs_transition.next_obs,
            next_state=next_state,
            reward=first_obs_transition.reward,
            n_step_reward=n_step_reward,
            n_step_cumulants=n_step_cumulants,
            boot_obs=last_obs_transition.next_obs,
            boot_state=last_state,
            terminated=last_obs_transition.terminated,
            truncate=last_obs_transition.truncate,
            state_dp=first_dp,
            next_state_dp=next_dp,
            boot_state_dp=last_dp,
            gamma_exponent=gamma_exp,
            gap=last_obs_transition.gap,
            steps_until_decision=first_step_until_decision,
            next_steps_until_decision=first_step_until_decision - 1,
            boot_steps_until_decision=first_step_until_decision
        )

        return [transition]

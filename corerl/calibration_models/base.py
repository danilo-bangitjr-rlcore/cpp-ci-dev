import numpy as np
import random
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import Optional
from abc import ABC, abstractmethod
from omegaconf import DictConfig

import corerl.calibration_models.utils as utils
from corerl.component.network.utils import to_np
from corerl.data.data import Transition, OldObsTransition
from corerl.data.transition_creator import OldAnytimeTransitionCreator
from corerl.data.data import Trajectory
from corerl.agent.base import BaseAgent


class BaseCalibrationModel(ABC):
    def __init__(self, cfg: DictConfig, train_info: dict):
        self.test_trajectories = train_info['test_trajectories_cm']
        self.train_trajectories = train_info['train_trajectories_cm']
        self.trajectories = self.train_trajectories + self.test_trajectories

        self.reward_func = train_info['reward_func']
        self.normalizer = train_info['normalizer']
        self.transition_creator = train_info['transition_creator']

        self.endo_inds = cfg.endo_inds
        self.exo_inds = cfg.exo_inds

        self.max_rollout_len = cfg.max_rollout_len
        self.steps_per_decision = cfg.steps_per_decision
        self.num_test_rollouts = cfg.num_test_rollouts
        self.gamma = cfg.gamma
        self.allow_learning = cfg.allow_learning

        self.rollout_indices = None  # indices of the trajectories for agent rollouts

    @abstractmethod
    def train(self):
        # You can make this method pass if you do not need to train a model
        raise NotImplementedError

    def do_test_rollouts(self, plot_save_path=None) -> list[list[float]]:
        if plot_save_path is not None:
            plot_save_path.mkdir(parents=True, exist_ok=True)

        losses = []
        for n, test_traj in enumerate(self.test_trajectories):
            last = test_traj.num_transitions - self.max_rollout_len
            increase_idx = last // self.num_test_rollouts
            start_idx = 0
            if test_traj.num_transitions >= self.max_rollout_len:
                for start in range(self.num_test_rollouts):
                    _, traj_losses = self._do_rollout(test_traj,
                                                      start_idx=start_idx,
                                                      plot='test',
                                                      plot_save_path=plot_save_path)
                    start_idx += increase_idx
                    losses.append(traj_losses)

        return losses

    def _get_sample_indices(self, trajectories: list[Trajectory]) -> list[tuple[int, int]]:
        num_transitions = np.array([traj.num_transitions for traj in trajectories])
        sample_probs = num_transitions / np.sum(num_transitions)
        rollout_indices = []
        traj_indices = list(range(len(trajectories)))
        while len(rollout_indices) <= self.num_test_rollouts:
            traj_idx = np.random.choice(traj_indices, p=sample_probs)
            traj = trajectories[traj_idx]
            if traj.num_transitions >= self.max_rollout_len:
                start_idx = random.choice(list(range(traj.num_transitions - self.max_rollout_len)))
                rollout_indices.append((traj_idx, start_idx))
        return rollout_indices

    def do_agent_rollouts(self,
                          agent: BaseAgent,
                          trajectories_agent: list[Trajectory],
                          plot=None,
                          plot_save_path=None,
                          resample=False) -> list[float]:

        if plot_save_path is not None:
            plot_save_path.mkdir(parents=True, exist_ok=True)

        if self.rollout_indices is None:
            # in the first call to do_agent_rollouts, verify that the agent and cm transitions
            # are lined up for the same observations.
            assert len(trajectories_agent) == len(self.trajectories)
            for traj_idx in range(len(trajectories_agent)):
                traj_cm = self.trajectories[traj_idx]
                traj_agent = trajectories_agent[traj_idx]
                assert traj_cm.num_transitions == traj_agent.num_transitions
                for i in range(traj_cm.num_transitions):
                    assert np.allclose(traj_cm.transitions[i].obs, traj_agent.transitions[i].obs)
                    assert np.allclose(traj_cm.transitions[i].next_obs, traj_agent.transitions[i].next_obs)
                    assert np.allclose(traj_cm.transitions[i].action, traj_agent.transitions[i].action)

        if resample or self.rollout_indices is None:
            # this will cache which indices are being sampled, so each call to
            # do_agent_rollouts will use the same starting states
            self.rollout_indices = self._get_sample_indices(self.trajectories)

        returns = []
        for (traj_idx, start_idx) in self.rollout_indices:
            return_, _ = self._do_rollout(self.trajectories[traj_idx],
                                          agent=agent,
                                          traj_agent=trajectories_agent[traj_idx],
                                          start_idx=start_idx,
                                          plot=plot,
                                          plot_save_path=plot_save_path)
            returns.append(return_)

        return returns

    def _get_reward(self, prev_action: np.ndarray, curr_action: np.ndarray, obs: np.ndarray) -> float:
        reward_info = {}
        if prev_action is None:
            reward_info['prev_action'] = curr_action
        else:
            reward_info['prev_action'] = prev_action
        reward_info['curr_action'] = curr_action

        # NOTE: Not sure if this denormalizer should be here.
        denormalized_obs = self.normalizer.obs_normalizer.denormalize(obs)
        r = self.reward_func(denormalized_obs, **reward_info)
        r_norm = self.normalizer.reward_normalizer(r)
        return r_norm

    def _do_rollout(self,
                    traj_cm: Trajectory,
                    agent: Optional[BaseAgent] = None,
                    traj_agent: Optional[Trajectory] = None,
                    start_idx: Optional[int] = None,
                    plot=None,
                    plot_save_path=None,
                    ) -> tuple[float, list[float]]:

        """
        This is a possibly general version of doing a rollout with the calibration model. You
        may use this version by implementing _get_next_obs() in subclasses of BaseCalibrationModel.
        However, you may also wish to override this method in subclasses if the rollout does not fit
        this template.

        This method can be used with or without an agent. If no agent is given, actions are selected
        according to the traj_cm.

        NOTE that the transitions in trajectories here must have the same frequency as the agent expects with its state constructor
        """

        if start_idx is None:
            start_idx = random.randint(0, traj_cm.num_transitions - self.max_rollout_len - 1)

        transitions_cm = traj_cm.transitions[start_idx:]
        sc_cm = traj_cm.get_sc_at_idx(start_idx)  # state constructor for the model
        state_cm = transitions_cm[0].state  # initial state for the model

        state_agent = None  # the state for the agent is not used unless we pass in an agent
        sc_agent = None
        use_agent = False

        if agent is not None:
            assert traj_agent is not None
            transitions_agent = traj_agent.transitions[start_idx:]

            sc_agent = traj_agent.get_sc_at_idx(start_idx)  # state constructor for the agent
            state_agent = transitions_agent[0].state
            use_agent = True

        g = 0  # the return

        losses = []
        endo_obss = []
        predicted_endo_obss = []
        actions = []
        rollout_len = min(len(transitions_cm), self.max_rollout_len)

        steps_until_decision = None
        action = transitions_cm[0].action  # the initial agent's action
        decision_point = transitions_cm[0].state_dp

        # we need the following variables for constructing observation transitions
        prev_action = None
        prev_obs = transitions_cm[0].obs
        prev_steps_until_decision = None
        prev_decision_point = decision_point

        if use_agent:
            # these lists are used to construct transitions.
            curr_decision_obs_transitions = []
            curr_decision_states = [state_agent]  # initialize this list with the first state that the agent sees

        for step in range(rollout_len):
            transition_step = transitions_cm[step]
            if steps_until_decision is None or not use_agent:
                steps_until_decision = transition_step.gamma_exponent
            elif steps_until_decision == 0:
                steps_until_decision = self.steps_per_decision

            if prev_steps_until_decision is None:
                prev_steps_until_decision = steps_until_decision

            action = self._get_action(action, transition_step, decision_point,
                                      use_agent=use_agent, agent=agent, state_agent=state_agent)

            next_obs = transition_step.next_obs  # the TRUE next observation
            next_endo_obs = next_obs[self.endo_inds]  # the endogenous component of the true next observation

            kwargs = {}  # add kwargs to _get_next_endo_obs here
            predicted_next_endo_obs = self._get_next_endo_obs(state_cm, action, kwargs)

            # log the loss. Note this loss is not meaning if we are using an agent
            loss_step = np.mean(np.abs(next_endo_obs - to_np(predicted_next_endo_obs)))
            losses.append(loss_step)

            # construct a fictitious observation using the predicted endogenous variables and the actual
            # exogenous variables
            fictitious_next_obs = utils.new_fictitious_obs(predicted_next_endo_obs, next_obs, self.endo_inds)
            # update the state constructors
            steps_until_decision -= 1
            decision_point = steps_until_decision == 0

            state_cm = sc_cm(fictitious_next_obs, action,
                             decision_point=decision_point,
                             steps_until_decision=steps_until_decision)

            r = self._get_reward(prev_action, action, fictitious_next_obs)
            g += (self.gamma ** step) * r

            if use_agent:
                state_agent = sc_agent(fictitious_next_obs, action,
                                       decision_point=decision_point,
                                       steps_until_decision=steps_until_decision)

                obs_transition = OldObsTransition(
                    prev_action,
                    prev_obs,
                    prev_steps_until_decision,
                    prev_decision_point,
                    action,
                    r,
                    fictitious_next_obs,
                    steps_until_decision,
                    decision_point,
                    False,  # termination false
                    False,  # truncation false
                    gap=False)  # assume no data gap

                curr_decision_states.append(state_agent)
                curr_decision_obs_transitions.append(obs_transition)

                curr_decision_obs_transitions, curr_decision_states = self._make_transitions_and_update(agent,
                                                                                                        decision_point,
                                                                                                        curr_decision_obs_transitions,
                                                                                                        curr_decision_states)

            prev_action = action
            prev_obs = fictitious_next_obs
            prev_steps_until_decision = steps_until_decision
            prev_decision_point = decision_point

            # log stuff
            actions.append(action)
            endo_obss.append(next_obs[0])
            predicted_endo_obss.append(predicted_next_endo_obs)

        if plot is not None:
            self._plot(endo_obss, actions, predicted_endo_obss, plot_save_path, plot, start_idx)

        return g, losses

    def _make_transitions_and_update(self, agent, decision_point, curr_decision_obs_transitions, curr_decision_states):
        if decision_point:
            if self.allow_learning:
                _, _, agent_transitions = self.transition_creator.make_decision_window_transitions(
                    curr_decision_obs_transitions, curr_decision_states)

                for transition in agent_transitions:
                    agent.update_buffer(transition)

            curr_decision_obs_transitions = []
            curr_decision_states = [curr_decision_states[-1]]

        if self.allow_learning:
            agent.update()

        return curr_decision_obs_transitions, curr_decision_states

    def _plot(self, endo_obss, actions, predicted_endo_obss, plot_save_path, plot, start_idx):
        plt.plot(endo_obss, label='endo obs.')
        plt.plot(actions, label='actions')

        predicted_endo_obss = [np.squeeze(to_np(p)) for p in predicted_endo_obss]
        plt.plot(predicted_endo_obss, label='predicted endo obs.')
        plt.legend()

        plt.xlabel("Rollout Step")
        plt.savefig(plot_save_path / f"rollout_{plot}_{start_idx}.png", bbox_inches='tight')
        plt.clf()

    @abstractmethod
    def _get_next_endo_obs(self, state, action, kwargs):
        raise NotImplementedError

    def _get_action(self, prev_action, transition, decision_point, use_agent=False, agent=None, state_agent=None):
        if use_agent:
            assert agent is not None
            assert state_agent is not None
            if decision_point:
                action = agent.get_action(state_agent)
            else:
                action = prev_action
        else:
            action = transition.action
        return action

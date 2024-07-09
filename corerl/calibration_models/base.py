import numpy as np
import random
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import Optional
from abc import ABC, abstractmethod
from omegaconf import DictConfig

import corerl.calibration_models.utils as utils
from corerl.component.network.utils import to_np
from corerl.data.data import Trajectory
from corerl.agent.base import BaseAgent


class BaseCalibrationModel(ABC):
    def __init__(self, cfg: DictConfig, train_info: dict):
        self.test_trajectories = train_info['test_trajectories_cm']
        self.reward_func = train_info['reward_func']
        self.normalizer = train_info['normalizer']

        self.endo_inds = cfg.endo_inds
        self.exo_inds = cfg.exo_inds

        self.max_rollout_len = cfg.max_rollout_len
        self.steps_per_decision = cfg.steps_per_decision
        self.num_test_rollouts = cfg.num_test_rollouts
        self.gamma = cfg.gamma

    @abstractmethod
    def train(self):
        # You can make this method pass if you do not need to train a model
        raise NotImplementedError

    def do_test_rollouts(self, plot_save_path=None):
        for n, test_traj in enumerate(self.test_trajectories):
            last = test_traj.num_transitions - self.max_rollout_len
            increase_idx = last // self.num_test_rollouts
            start_idx = 0
            for start in range(self.num_test_rollouts):
                self._do_rollout(test_traj, start_idx=start_idx, plot='test', plot_save_path=plot_save_path)
                start_idx += increase_idx

    def do_agent_rollouts(self, agent: BaseAgent, trajectories_agent: list[Trajectory], plot=None, plot_save_path=None):
        returns = []
        assert len(trajectories_agent) == len(self.test_trajectories)
        for traj_i, _ in enumerate(self.test_trajectories):
            traj_cm = self.test_trajectories[traj_i]
            traj_agent = trajectories_agent[traj_i]

            assert traj_cm.num_transitions == traj_agent.num_transitions

            last = traj_cm.num_transitions - self.max_rollout_len
            increase_idx = last // self.num_test_rollouts
            start_idx = 0
            for start in range(self.num_test_rollouts):
                return_ = self._do_rollout(traj_cm,
                                           agent=agent,
                                           traj_agent=traj_agent,
                                           start_idx=start_idx,
                                           plot=plot,
                                           plot_save_path=plot_save_path)
                start_idx += increase_idx
                returns.append(return_)
        return returns

    def _do_rollout(self,
                    traj_cm: Trajectory,
                    agent: Optional[BaseAgent] = None,
                    traj_agent: Optional[Trajectory] = None,
                    start_idx: Optional[int] = None,
                    plot=None,
                    plot_save_path=None,
                    ) -> float:

        """
        This is a possibly general version of doing a rollout with the calibration model. You
        may use this version by implementing _get_next_obs() in subclasses of BaseCalibrationModel.
        However, you may also wish to override this method in subclasses if the rollout does not fit
        this template.

        This method can be used with or without an agent. If no agent is given, actions are selected
        according to the traj_cm.
        """

        if start_idx is None:
            start_idx = random.randint(0, traj_cm.num_transitions - self.max_rollout_len - 1)
        transitions_cm = traj_cm.transitions[start_idx:]
        sc_cm = deepcopy(traj_cm.scs[start_idx])  # state constructor for the model
        state_cm = transitions_cm[0].state  # initial state for the model
        state_agent = None
        sc_agent = None
        use_agent = False

        if agent is not None:
            assert traj_agent is not None
            transitions_agent = traj_agent.transitions[start_idx:]
            for i in range(len(transitions_cm)):
                assert np.array_equal(transitions_cm[i].obs, transitions_agent[i].obs)
                assert np.array_equal(transitions_cm[i].action, transitions_agent[i].action)
                assert np.array_equal(transitions_cm[i].next_obs, transitions_agent[i].next_obs)

            sc_agent = deepcopy(traj_agent.scs[start_idx])  # state constructor for the agent
            state_agent = transitions_agent[0].state
            use_agent = True

        g = 0  # the return
        prev_action = None

        losses = []
        endo_obss = []
        predicted_endo_obss = []
        actions = []
        rollout_len = min(len(transitions_cm), self.max_rollout_len)

        steps_until_decision_point = None
        action = transitions_cm[0].action  # the initial agent's action
        decision_point = transitions_cm[0].state_dp

        for step in range(rollout_len):
            transition_step = transitions_cm[step]
            if steps_until_decision_point is None or not use_agent:
                steps_until_decision_point = transition_step.gamma_exponent
            elif steps_until_decision_point == 0:
                steps_until_decision_point = self.steps_per_decision

            action = self._get_action(action, transition_step, decision_point,
                                      use_agent=use_agent, agent=agent, state_agent=state_agent)

            next_obs = transition_step.next_obs  # the true next observation
            next_endo_obs = next_obs[self.endo_inds]  # the endogenous component of the true next observation
            action_duration = steps_until_decision_point / self.steps_per_decision

            kwargs = {'duration': action_duration}
            predicted_next_endo_obs = self._get_next_obs(state_cm, action, kwargs)

            # log the loss
            loss_step = np.mean(np.abs(next_endo_obs - to_np(predicted_next_endo_obs)))
            losses.append(loss_step)

            # construct a fictitious observation using the predicted endogenous variables and the actual
            # exogenous variables
            fictitious_obs = utils.new_fictitious_obs(predicted_next_endo_obs, next_obs, self.endo_inds)

            # update the state constructors
            steps_until_decision_point -= 1
            decision_point = steps_until_decision_point == 0
            state_cm = sc_cm(fictitious_obs, action, decision_point=decision_point)

            if use_agent:
                state_agent = sc_agent(fictitious_obs, action, decision_point=decision_point)

            reward_info = {}
            if prev_action is None:
                reward_info['prev_action'] = action
            else:
                reward_info['prev_action'] = prev_action
            reward_info['curr_action'] = action

            # NOTE: Not sure if this denormalizer should be here.
            # TODO: make this return "regular RL" reward if desired
            denormalized_obs = self.normalizer.obs_normalizer.denormalize(fictitious_obs)
            r = self.reward_func(denormalized_obs, **reward_info)
            r_norm = self.normalizer.reward_normalizer(r)
            g += self.gamma ** step * r_norm
            prev_action = action

            # log stuff
            actions.append(action)
            endo_obss.append(next_obs[0])
            predicted_endo_obss.append(predicted_next_endo_obs)

        if plot is not None:
            plt.plot(endo_obss, label='endo obs.')
            plt.plot(actions, label='actions')

            predicted_endo_obss = [np.squeeze(to_np(p)) for p in predicted_endo_obss]
            plt.plot(predicted_endo_obss, label='predicted endo obs.')
            plt.legend()

            plt.xlabel("Rollout Step")
            plt.savefig(plot_save_path / f"rollout_{plot}_{start_idx}.png", bbox_inches='tight')
            plt.clf()

        return g

    @abstractmethod
    def _get_next_obs(self, state, action, kwargs):
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

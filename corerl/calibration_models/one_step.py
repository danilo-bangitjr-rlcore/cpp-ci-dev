import torch
import torch.nn as nn
import numpy as np
import random

from tqdm import tqdm
from copy import deepcopy
from omegaconf import DictConfig
from typing import Optional

from corerl.agent.base import BaseAgent
from corerl.component.buffer.factory import init_buffer
from corerl.component.network.factory import init_custom_network
from corerl.component.optimizers.factory import init_optimizer
from corerl.component.network.utils import tensor, to_np
from corerl.calibration_models.base import NNCalibrationModel
from corerl.data import Trajectory
from corerl.state_constructor.base import BaseStateConstructor
import corerl.calibration_models.utils as utils

import matplotlib.pyplot as plt


class OneStep(NNCalibrationModel):
    def __init__(self, cfg: DictConfig, train_info):
        self.test_trajectories = train_info['test_trajectories_cm']
        train_transitions = train_info['train_transitions_cm']
        self.reward_func = train_info['reward_func']
        self.interaction = train_info['interaction_cm']

        self.buffer = init_buffer(cfg.buffer)
        self.test_buffer = init_buffer(cfg.buffer)
        self.train_itr = cfg.train_itr
        self.batch_size = cfg.batch_size

        self.buffer.load(train_transitions)
        self.endo_inds = cfg.endo_inds
        self.exo_inds = cfg.exo_inds

        input_dim = len(train_transitions[0].state)
        action_dim = len(train_transitions[0].action)
        output_dim = len(train_transitions[0].obs[self.endo_inds])

        self.model = init_custom_network(cfg.model, input_dim=input_dim + action_dim, output_dim=output_dim)
        self.optimizer = init_optimizer(cfg.optimizer, list(self.model.parameters()))

        self.train_losses = []
        self.test_losses = []

        self.max_rollout_len = cfg.max_rollout_len
        self.steps_per_decision = cfg.steps_per_decision
        self.num_test_rollouts = cfg.num_test_rollouts

    def update(self):
        batch = self.buffer.sample_mini_batch(self.batch_size)
        state_batch, action_batch, next_obs_batch = batch.state, batch.action, batch.next_obs
        # we only predict the next endogenous component of the observation
        endo_next_obs_batch = next_obs_batch[:, self.endo_inds]
        prediction = self.get_prediction(state_batch, action_batch, with_grad=True)
        loss = nn.functional.mse_loss(prediction, endo_next_obs_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_losses.append(loss.detach().numpy())

    def train(self):
        print('Training model...')
        pbar = tqdm(range(self.train_itr))
        for _ in pbar:
            self.update()
            pbar.set_description("train loss: {:7.6f}".format(self.train_losses[-1]))

        self.do_test_rollouts()
        return self.train_losses, self.test_losses

    def get_prediction(self, state: torch.Tensor, action: torch.Tensor, with_grad: bool = False):
        x = torch.concat((state, action), dim=1)
        if with_grad:
            y = self.model(x)
        else:
            with torch.no_grad():
                y = self.model(x)
        return y

    def do_test_rollouts(self):
        for n, test_traj in enumerate(self.test_trajectories):
            last = test_traj.num_transitions - self.max_rollout_len
            increase_idx = last // self.num_test_rollouts
            start_idx = 0
            for start in range(self.num_test_rollouts):
                self.do_test_rollout(test_traj, start_idx=start_idx, plot=True)
                start_idx += increase_idx

    def do_test_rollout(self, traj: Trajectory, start_idx: Optional[int] = None, plot=False) -> list[float]:
        if start_idx is None:
            start_idx = random.randint(0, traj.num_transitions - self.max_rollout_len - 1)

        transitions = traj.transitions[start_idx:]
        sc = deepcopy(traj.scs[start_idx])
        state = transitions[0].state

        losses = []
        endo_obss = []
        predicted_endo_obss = []
        actions = []
        rollout_len = min(len(transitions), self.max_rollout_len)

        for step in range(rollout_len):
            transition_step = transitions[step]
            action = transition_step.action

            next_obs = transition_step.next_obs
            next_endo_obs = next_obs[self.endo_inds]

            state_tensor = tensor(state).reshape((1, -1))
            action_tensor = tensor(action).reshape((1, -1))

            predicted_next_endo_obs = self.get_prediction(state_tensor, action_tensor)

            # log the loss
            loss_step = np.mean(np.abs(next_endo_obs - to_np(predicted_next_endo_obs)))
            losses.append(loss_step)

            # construct a fictitious observation using the predicted endogenous variables and the actual
            # exogenous variables
            new_fictitious_obs = utils.new_fictitious_obs(predicted_next_endo_obs, next_obs, self.endo_inds)

            decision_point = step % self.steps_per_decision == 0
            state = sc(new_fictitious_obs, action, decision_point=decision_point)

            # log stuff
            actions.append(action)
            endo_obss.append(next_obs[0])
            predicted_endo_obss.append(predicted_next_endo_obs)

        if plot:
            plt.plot(endo_obss, label='endo obs.')
            plt.plot(actions, label='actions')

            predicted_endo_obss = [np.squeeze(to_np(p)) for p in predicted_endo_obss]
            plt.plot(predicted_endo_obss, label='predicted endo obs.')
            plt.legend()

            plt.xlabel("Rollout Step")
            plt.savefig(f"test_{start_idx}.png", bbox_inches='tight')
            plt.clf()

        return losses

    def do_agent_rollout(self,
                         traj_cm: Trajectory,
                         traj_agent: Trajectory,
                         agent: BaseAgent,
                         start_idx: Optional[int] = None,
                         plot=False) -> float:
        if start_idx is None:
            start_idx = random.randint(0, traj_cm.num_transitions - self.max_rollout_len - 1)

        transitions_cm = traj_cm.transitions[start_idx:]
        transitions_agent = traj_agent.transitions[start_idx:]
        # we have two different state constructors, one for the agent and one for the model
        sc_cm = deepcopy(traj_cm.scs[start_idx])
        sc_agent = deepcopy(traj_agent.scs[start_idx])

        state_cm = transitions_cm[0].state
        state_agent = transitions_agent[0].state

        gamma = agent.gamma
        g = 0  # the return
        prev_action = None

        losses = []
        endo_obss = []
        predicted_endo_obss = []
        actions = []
        rollout_len = min(len(transitions_cm), self.max_rollout_len)

        remaining_decision_steps = transitions_cm[0].gamma_exponent
        action = transitions_cm[0].action  # the initial agent's action

        for step in range(rollout_len):
            transition_step = transitions_cm[step]

            # if it is time for a decision, sample an action from the agent
            if remaining_decision_steps == 0:
                action = agent.get_action(state_agent)
                remaining_decision_steps = self.steps_per_decision

            next_obs = transition_step.next_obs
            next_endo_obs = next_obs[self.endo_inds]

            state_cm_tensor = tensor(state_cm).reshape((1, -1))
            action_tensor = tensor(action).reshape((1, -1))
            predicted_next_endo_obs = self.get_prediction(state_cm_tensor, action_tensor)

            # log the loss
            loss_step = np.mean(np.abs(next_endo_obs - to_np(predicted_next_endo_obs)))
            losses.append(loss_step)

            # construct a fictitious observation using the predicted endogenous variables and the actual
            # exogenous variables
            new_fictitious_obs = utils.new_fictitious_obs(predicted_next_endo_obs, next_obs, self.endo_inds)

            decision_point = step % self.steps_per_decision == 0
            state_cm = sc_cm(new_fictitious_obs, action, decision_point=decision_point)
            state_agent = sc_agent(new_fictitious_obs, action, decision_point=decision_point)

            reward_info = {}
            if prev_action is None:
                reward_info['prev_action'] = action
            else:
                reward_info['prev_action'] = prev_action
            reward_info['curr_action'] = action

            # Not sure if this denormalizer should be here.
            denormalized_obs = self.interaction.obs_normalizer.denormalize(new_fictitious_obs)
            r = self.reward_func(denormalized_obs, **reward_info)
            r_norm = self.interaction.reward_normalizer(r)
            g += gamma * r_norm
            prev_action = action

            remaining_decision_steps -= 1

            # log stuff
            actions.append(action)
            endo_obss.append(next_obs[0])
            predicted_endo_obss.append(predicted_next_endo_obs)

        if plot:
            plt.plot(endo_obss, label='endo obs.')
            plt.plot(actions, label='actions')

            predicted_endo_obss = [np.squeeze(to_np(p)) for p in predicted_endo_obss]
            plt.plot(predicted_endo_obss, label='predicted endo obs.')
            plt.legend()

            plt.xlabel("Rollout Step")
            plt.savefig(f"test_{start_idx}.png", bbox_inches='tight')
            plt.clf()

        return g

    def do_agent_rollouts(self, agent: BaseAgent, trajectories_agent: list[Trajectory], plot=False):
        returns = []
        assert len(trajectories_agent) == len(self.test_trajectories)
        for traj_i in range(len(trajectories_agent)):
            traj_cm = self.test_trajectories[traj_i]
            traj_agent = self.test_trajectories[traj_i]
            return_ = self.do_agent_rollout(traj_cm, traj_agent, agent, start_idx=1000, plot=True)
            returns.append(return_)
        return returns

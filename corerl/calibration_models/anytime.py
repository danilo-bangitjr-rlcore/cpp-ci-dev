import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from copy import deepcopy
from omegaconf import DictConfig
from typing import Optional

import corerl.calibration_models.utils as utils
from corerl.component.buffer.factory import init_buffer
from corerl.component.network.factory import init_custom_network
from corerl.component.optimizers.factory import init_optimizer
from corerl.calibration_models.base import NNCalibrationModel
from corerl.component.network.utils import tensor, to_np
from corerl.data import Trajectory
from corerl.agent.base import BaseAgent


class AnytimeCalibrationModel(NNCalibrationModel):
    def __init__(self, cfg: DictConfig, train_info):
        self.test_trajectories = train_info['test_trajectories']
        train_transitions = train_info['train_transitions']
        self.reward_func = train_info['reward_func']
        self.interaction = train_info['interaction']

        self.buffer = init_buffer(cfg.buffer)
        self.test_buffer = init_buffer(cfg.buffer)
        self.train_itr = cfg.train_itr
        self.batch_size = cfg.batch_size

        self.buffer.load(train_transitions)

        test_transitions = []
        for t in self.test_trajectories:
            test_transitions += t.transitions

        self.test_buffer.load(test_transitions)

        self.endo_inds = cfg.endo_inds
        self.exo_inds = cfg.exo_inds

        input_dim = len(train_transitions[0].state)
        action_dim = len(train_transitions[0].action)
        output_dim = len(train_transitions[0].obs[self.endo_inds])

        # the plus one is for the duration until the next observation
        self.model = init_custom_network(cfg.model, input_dim=input_dim + action_dim + 1, output_dim=output_dim)
        self.optimizer = init_optimizer(cfg.optimizer, list(self.model.parameters()))

        self.train_losses = []
        self.test_losses = []

        self.max_rollout_len = cfg.max_rollout_len
        self.steps_per_decision = cfg.steps_per_decision
        # this is for duration normalization. Could be different in the future.
        self.max_action_duration = cfg.steps_per_decision
        self.interpolation = cfg.interpolation
        self.num_test_rollouts = cfg.num_test_rollouts

    def eval(self, batch, with_grad):
        # gamma_exponents double as the durations of actions
        state_batch, action_batch, next_obs_batch, duration = batch.state, batch.action, batch.boot_obs, batch.gamma_exponent
        endo_next_obs_batch = next_obs_batch[:, self.endo_inds]

        duration /= self.max_action_duration
        prediction = self.get_prediction(state_batch, action_batch, duration, with_grad=with_grad)
        loss = nn.functional.mse_loss(prediction, endo_next_obs_batch)

        return loss

    def update(self):
        batch = self.buffer.sample_mini_batch(self.batch_size)
        loss = self.eval(batch, with_grad=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_losses.append(loss.detach().numpy())

    def train(self):
        print('Training model...')
        pbar = tqdm(range(self.train_itr))
        train_losses = []
        train_loss = 0
        test_loss = 0
        window_avg = 100
        test_losses = []
        for itr in pbar:
            self.update()

            if len(self.train_losses) >= window_avg:
                train_loss = np.mean(self.train_losses[-window_avg:])

                train_losses.append(train_loss)

            if itr % 100 == 0:
                test_batch = self.test_buffer.sample_mini_batch(self.batch_size)
                test_loss = self.eval(test_batch, with_grad=False).detach().numpy()
                test_losses.append(test_loss)

            pbar.set_description("train loss: {:7.6f}, test_loss: {:7.6f}".format(train_loss, test_loss))

        self.do_test_rollouts()
        return self.train_losses, self.test_losses

    def linear_interpolation(self, inter_step, duration_int, curr_obs, predicted_next_endo_obs):
        w = (inter_step) / duration_int
        fictitious_endo_obs = (1 - w) * curr_obs[self.endo_inds] + w * predicted_next_endo_obs
        return fictitious_endo_obs

    def do_test_rollout(self, traj: Trajectory, start_idx: int, plot: bool = False, agent: BaseAgent=None) -> list[float]:
        # validates the model's accuracy on a test rollout
        transitions = traj.transitions[start_idx:]
        sc = deepcopy(traj.scs[start_idx])
        losses = []
        rollout_len = min(traj.num_transitions, self.max_rollout_len)

        state = transitions[0].state

        endo_obss = []
        predicted_endo_obss = []
        actions = []

        step = 0
        done = False

        num_predictions = 0
        curr_obs = transitions[0].obs
        remaining_decision_steps = 0


        # but the agent could be using a different state constructor from the environment, hey????
        while not done:
            transition_step = transitions[step]

            # this logic decides whether to get the next action from the agent or the environment
            if agent is not None:
                if remaining_decision_steps == 0:
                    action = agent.get_action(state)
                    remaining_decision_steps = self.steps_per_decision
                    duration_int = self.steps_per_decision
                else:
                    duration_int = remaining_decision_steps
            else:
                action = transition_step.action
                duration_int = transition_step.gamma_exponent

            duration = duration_int / self.max_action_duration

            state_tensor = tensor(state).reshape((1, -1))
            action_tensor = tensor(action).reshape((1, -1))
            duration_tensor = tensor(duration).reshape((1, -1))

            predicted_next_endo_obs = to_np(self.get_prediction(state_tensor, action_tensor, duration_tensor))
            for inter_step in range(1, duration_int + 1):
                inter_step_transition = transitions[step + inter_step]
                inter_step_obs = inter_step_transition.obs
                inter_next_obs = inter_step_transition.next_obs

                if self.interpolation == 'linear':
                    predicted_inter_endo_obs = self.linear_interpolation(inter_step, duration_int, curr_obs,
                                                                         predicted_next_endo_obs)
                else:  # use the model to interpolate
                    inter_step_duration = inter_step / self.max_action_duration
                    inter_duration_tensor = tensor(inter_step_duration).reshape((1, -1))

                    predicted_inter_endo_obs = to_np(
                        self.get_prediction(state_tensor, action_tensor, inter_duration_tensor))

                fictitious_obs = utils.new_fictitious_obs(predicted_inter_endo_obs, inter_next_obs, self.endo_inds)
                decision_point = step + inter_step % self.steps_per_decision == 0
                state = sc(fictitious_obs, action, decision_point=decision_point)

                loss_step = np.mean(np.abs(inter_step_obs[self.endo_inds] - to_np(predicted_inter_endo_obs)))
                losses.append(loss_step)

                actions.append(action)
                endo_obss.append(inter_step_obs[0])
                predicted_endo_obss.append(predicted_inter_endo_obs)

            step += duration_int
            num_predictions += 1

            # log the loss
            curr_obs = fictitious_obs

            if step > rollout_len:
                done = True

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

    def do_test_rollouts(self):
        for n, test_traj in enumerate(self.test_trajectories):
            last = test_traj.num_transitions - self.max_rollout_len
            increase_idx = last // self.num_test_rollouts
            start_idx = 0
            for start in range(self.num_test_rollouts):
                self.do_test_rollout(test_traj, start_idx=start_idx, plot=True)
                start_idx += increase_idx

    def get_prediction(self, state: torch.Tensor, action: torch.Tensor, duration: torch.Tensor,
                       with_grad: bool = False):
        x = torch.concat((state, action, duration), dim=1)
        if with_grad:
            y = self.model(x)
        else:
            with torch.no_grad():
                y = self.model(x)
        return y


    def do_agent_rollouts(self, agent, plot=False):
        returns = []
        for test_traj in self.test_trajectories:
            rollout_return = self.do_test_rollout(test_traj, agent=agent, plot=plot, start_idx=1000)
            returns.append(rollout_return)

        return returns


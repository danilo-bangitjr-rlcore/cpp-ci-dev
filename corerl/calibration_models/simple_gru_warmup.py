import numpy as np
import random
import matplotlib.pyplot as plt

from tqdm import tqdm
from omegaconf import DictConfig
from typing import Tuple

from corerl.component.optimizers.factory import init_optimizer
from corerl.component.network.utils import tensor
from corerl.utils.device import device
from corerl.data import Trajectory
from corerl.component.network.networks import GRU

import torch
import torch.nn as nn


class GRUCalibrationModel:
    def __init__(self, cfg: DictConfig, train_info):
        self.train_trajectories = train_info['train_trajectories']
        self.test_trajectories = train_info['test_trajectories']
        self.reward_func = train_info['reward_func']
        self.interaction = train_info['interaction']

        self.train_data = []  # pre-processed version of self.train_trajectories
        self.test_data = []  # pre-processed version of self.train_trajectories

        self.train_itr = cfg.train_itr
        self.batch_size = cfg.batch_size

        self.endo_inds = cfg.endo_inds
        self.exo_inds = cfg.exo_inds

        assert len(self.train_trajectories) > 0, "Must provide at least one train trajectory"

        example_transition = self.train_trajectories[0].transitions[0]
        action_dim = len(example_transition.action.shape)
        endo_dim = len(cfg.endo_inds)
        exo_dim = len(cfg.exo_inds)
        input_dim = endo_dim + exo_dim + action_dim

        # we are only predicting the endogenous variables at the next time step, so output_dim = endo_dim
        self.model = GRU(cfg.model, input_dim, endo_dim)
        self.optimizer = init_optimizer(cfg.optimizer, list(self.model.parameters()))

        self.train_losses = []
        self.test_losses = []

        self.max_rollout_len = cfg.max_rollout_len
        self.steps_per_decision = cfg.steps_per_decision
        self.skip = cfg.skip  # what if we don't predict every observation, but every nth observation?
        self.warmup_len = cfg.warmup_len // self.skip

    def prepare_data(self) -> None:
        for traj in self.train_trajectories:
            inputs, outputs = self.vectorize_trajectory(traj)
            self.train_data.append((inputs, outputs))

        for traj in self.test_trajectories:
            inputs, outputs = self.vectorize_trajectory(traj)
            self.test_data.append((inputs, outputs))

    def get_endo_obs(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) == 2:
            return obs[:, self.endo_inds]
        elif len(obs.shape) == 1:
            return obs[self.endo_inds]
        else:
            raise ValueError("Observation should have 1 or 2 dimensions")

    def get_exo_obs(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) == 2:
            return obs[:, self.exo_inds]
        elif len(obs.shape) == 1:
            return obs[self.exo_inds]
        else:
            raise ValueError("Observation should have 1 or 2 dimensions")

    def vectorize_trajectory(self, traj: Trajectory) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = traj.get_transitions_attr('obs')  # a (seq_len x obs_size) array  of the observations
        next_obs = traj.get_transitions_attr('next_obs')  # a (seq_len x obs_size) array  of the next observations
        action = traj.get_transitions_attr('action')  # a (seq_len x action_dim) array of actions
        endo_obs, exo_obs = self.get_endo_obs(obs), self.get_exo_obs(obs)
        next_endo_obs = self.get_endo_obs(next_obs)

        inputs = np.concatenate((endo_obs, exo_obs, action), axis=1)
        inputs = tensor(np.expand_dims(inputs, axis=0)).to(device)

        # outputs are offset by one from the inputs
        outputs = tensor(np.expand_dims(next_endo_obs, axis=0)).to(device)

        skip_indices = torch.arange(0, inputs.size(1), self.skip)

        inputs = inputs[:, skip_indices, :]
        outputs = outputs[:, skip_indices, :]

        assert inputs.size(1) == outputs.size(1)
        return inputs, outputs

    def sample_mini_batch(self, data, batch_size: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        traj_lens = [sampled_data[0].size(1) for sampled_data in data]
        probs = [traj_len / sum(traj_lens) for traj_len in traj_lens]

        batch_inputs, batch_outputs = [], []
        while len(batch_inputs) < batch_size:
            # sampled data is a list of (input, output) pairs
            sampled_data = random.choices(data, weights=probs, k=1)[0]  # zero-index bc choices returns list
            sampled_traj_inputs, sampled_traj_outputs = sampled_data[0], sampled_data[1]
            num_transitions = sampled_traj_inputs.size(1)

            if num_transitions >= self.max_rollout_len:
                start_index = random.randint(0, num_transitions - self.max_rollout_len - 1)
                sub_traj_inputs = sampled_traj_inputs[:, start_index:start_index + self.max_rollout_len, :]
                sub_traj_outputs = sampled_traj_outputs[:, start_index:start_index + self.max_rollout_len, :]
                assert sub_traj_inputs.size(1) == self.max_rollout_len == sub_traj_outputs.size(1)
                batch_inputs.append(sub_traj_inputs)
                batch_outputs.append(sub_traj_outputs)

        input_tensor = torch.concat(batch_inputs, dim=0).to(device)
        output_tensor = torch.concat(batch_outputs, dim=0).to(device)

        return input_tensor, output_tensor

    def train(self):
        print('Preparing data...')
        self.prepare_data()
        print('Training model...')
        pbar = tqdm(range(self.train_itr))
        test_loss = 0

        for itr in pbar:
            if itr == self.train_itr - 1:
                self.update(plot=True)
            else:
                self.update(plot=False)
            pbar.set_description("train loss: {:7.4f}, test loss: {:7.4f}".format(
                self.train_losses[-1], test_loss))
            # if itr % 10 == 0:
        test_loss = self.test_rollouts()

        return self.train_losses, self.test_losses

    def update(self, plot=False):
        batch = self.sample_mini_batch(self.train_data, batch_size=self.batch_size)
        inputs, outputs = batch
        predicted_out = self.model(inputs, prediction_start=self.warmup_len - 1)
        loss = nn.functional.mse_loss(predicted_out, outputs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if plot:
            for b in range(self.batch_size):
                orps = outputs[b, :, 0].detach().numpy()  # the actual ORP
                actions = inputs[b, :, -1].detach().numpy()  # the preceding action
                predicted_orps = predicted_out[b, :, 0].detach().numpy()

                plt.plot(orps, label='orps')
                plt.plot(actions, label='actions')
                plt.plot(predicted_orps, label='predicted orps')
                plt.legend()

                plt.xlabel("Rollout Step")
                plt.savefig(f"train_{b}.png", bbox_inches='tight')
                plt.clf()

        self.train_losses.append(loss.detach().numpy())

    def test_rollouts(self):
        num_test = 10
        test_batch = self.sample_mini_batch(self.test_data, batch_size=num_test)
        all_losses = []
        for t in range(num_test):
            test_in = torch.unsqueeze(test_batch[0][t, :, :], 0)
            test_out = torch.unsqueeze(test_batch[1][t, :, :], 0)
            losses = self.test_rollout((test_in, test_out), t)
            # all_losses.append(np.mean(losses[self.warmup_len - 1:]))
            # plt.plot(np.array(losses), c='b', alpha=0.2)

        #
        # plt.ylabel("Absolute Error From True ORP")
        # plt.xlabel("Rollout Step")
        # plt.show()
        return np.mean(all_losses)

    def test_rollout(self, traj, num):
        # validates the model's accurary on a test rollout
        inputs, outputs = traj
        losses = []
        with torch.no_grad():
            predicted_out = self.model(inputs, prediction_start=self.warmup_len - 1)

        rollout_len = predicted_out.size(1)
        assert predicted_out.size(1) == outputs.size(1)
        for step in range(self.warmup_len - 1, rollout_len):
            loss = torch.nn.functional.l1_loss(predicted_out[0, step, :], outputs[0, step, :]).detach().numpy()
            losses.append(loss)

        orps = outputs[0, :, 0]  # the actual ORP
        actions = inputs[0, :, -1]  # the preceding action
        predicted_orps = predicted_out[0, :, 0]

        plt.plot(orps, label='orps')
        plt.plot(actions, label='actions')
        plt.plot(predicted_orps, label='predicted orps')
        plt.legend()

        plt.xlabel("Rollout Step")
        plt.savefig(f"test_{num}.png", bbox_inches='tight')
        plt.clf()

        return losses

    # TODO: below this not done!

    def get_prediction(self, state, action, with_grad=False):
        x = torch.concat((state, action), dim=1)
    #     if with_grad:
    #         y = self.model(x)
    #     else:
    #         with torch.no_grad():
    #             y = self.model(x)
    #     return y
    #
    # def _model_step(self, state, action, with_grad=False):
    #     obs = self.get_prediction(tensor(state).reshape((1, -1)),
    #                               tensor(action).reshape((1, -1)),
    #                               with_grad=with_grad)
    #     return obs
    #
    # def do_rollout(self, state, sc, agent, rollout_len=20):
    #     gamma = agent.gamma
    #     g = 0  # the return
    #     prev_action = None
    #     for i in range(rollout_len):
    #         action = agent.get_action(state)
    #         obs = self._model_step(state, action)
    #         state = sc(obs)
    #
    #         reward_info = {}
    #         if prev_action is None:
    #             reward_info['prev_action'] = action
    #         else:
    #             reward_info['prev_action'] = prev_action
    #         reward_info['curr_action'] = action
    #
    #         denormalized_obs = self.interaction.obs_normalizer.denormalize(obs)
    #         g += gamma * self.reward_func(denormalized_obs, **reward_info)
    #         prev_action = action
    #
    #     # This does not factor in truncs, or dones. Should it?
    #     return g
    #
    # def do_n_rollouts(self, agent, num_rollouts=100, rollout_len=20):
    #     returns = []
    #     for rollout in range(num_rollouts):
    #         rand_idx = random.randint(0, len(self.state_constructors))
    #         start_transition = self.test_transitions[rand_idx]
    #         start_state = start_transition[0]
    #         start_sc = self.state_constructors[rand_idx]
    #
    #         return_rollout = self.do_rollout(start_state, start_sc, agent, rollout_len=rollout_len)
    #         returns.append(return_rollout)
    #
    #     return returns

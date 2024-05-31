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
    def __init__(self, cfg: DictConfig, **kwargs):
        self.train_trajectories = kwargs['train_trajectories']
        self.test_trajectories = kwargs['test_trajectories']
        self.train_data = []  # pre-processed version of self.train_trajectories
        self.test_data = []  # pre-processed version of self.train_trajectories

        reward_func = kwargs['reward_func']
        self.interaction = kwargs['interaction']
        self.state_constructors = kwargs['test_scs']

        self.train_itr = cfg.train_itr
        self.batch_size = cfg.batch_size

        self.endo_inds = cfg.endo_inds
        self.exo_inds = cfg.exo_inds

        assert len(self.train_trajectories) > 0

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

        self.reward_func = reward_func
        self.max_rollout_len = cfg.max_rollout_len
        self.steps_per_decision = cfg.steps_per_decision
        self.skip = 1  # what if we don't predict every observation, but every nth observation?
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

    def sample_mini_batch(self, batch_size: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        sampled_data = random.choices(self.train_data, k=batch_size)  # sampled data is a list of (input, output) pairs

        inputs, outputs = list(zip(*sampled_data))
        seq_lens = [input_tensor.shape[1] for input_tensor in inputs]
        min_seq_len = min(seq_lens)

        input_tensors = [input_tensor[:, 0:min_seq_len, :] for input_tensor in inputs]
        input_tensor = torch.concat(input_tensors, dim=0).to(device)

        output_tensors = [output_tensor[:, 0:min_seq_len, :] for output_tensor in outputs]
        output_tensor = torch.concat(output_tensors, dim=0).to(device)

        return input_tensor, output_tensor

    def train(self):
        print('Preparing data...')
        self.prepare_data()
        print('Training model...')
        pbar = tqdm(range(self.train_itr))
        test_loss = 0

        for itr in pbar:
            self.update()
            pbar.set_description("train loss: {:7.4f}, test loss: {:7.4f}".format(
                self.train_losses[-1], test_loss))
            if itr % 10 == 0:
                test_loss = self.test_rollouts()

        return self.train_losses, self.test_losses

    def update(self):
        batch = self.sample_mini_batch(batch_size=self.batch_size)
        inputs, outputs = batch

        predicted_out = self.model(inputs, prediction_start=self.warmup_len - 1)
        # predicted_out = self.model(inputs)
        # loss = nn.functional.mse_loss(predicted_out[:, self.warmup_len-1:, :], outputs[:, self.warmup_len-1:, :])
        loss = nn.functional.mse_loss(predicted_out, outputs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_losses.append(loss.detach().numpy())

    def test_rollouts(self):
        all_losses = []
        for n, test_data in enumerate(self.test_data):
            losses = self.test_rollout(test_data, n)
            all_losses.append(np.mean(losses[self.warmup_len - 1:]))

        #     plt.plot(np.array(losses), c='b', alpha=0.2)
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
        for step in range(self.warmup_len-1, rollout_len):
            loss = torch.nn.functional.l1_loss(predicted_out[0, step, :], outputs[0, step, :]).detach().numpy()
            losses.append(loss)

        orps = outputs[0, :, 0] # the actual ORP
        actions = inputs[0, :, -1] # the preceding action
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

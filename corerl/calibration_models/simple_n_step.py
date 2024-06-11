import numpy as np
import random
import matplotlib.pyplot as plt

from tqdm import tqdm
from omegaconf import DictConfig
from typing import Tuple
from copy import deepcopy

from corerl.component.optimizers.factory import init_optimizer
from corerl.component.network.factory import init_custom_network
from corerl.component.network.utils import tensor
from corerl.utils.device import device
from corerl.data import Trajectory

import torch
import torch.nn as nn


class NStepCalibrationModel:
    def __init__(self, cfg: DictConfig, train_info):
        self.train_trajectories = train_info['train_trajectories']

        self.test_trajectories = train_info['test_trajectories']

        print(len(self.train_trajectories))
        print(len(self.test_trajectories))

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
        action_dim = example_transition.action.shape[0]
        state_dim = example_transition.state.shape[0]
        endo_dim = len(cfg.endo_inds)
        input_dim = state_dim + action_dim

        # we are only predicting the endogenous variables at the next time step, so output_dim = endo_dim
        self.model = init_custom_network(cfg.model, input_dim=input_dim, output_dim=endo_dim)
        self.optimizer = init_optimizer(cfg.optimizer, list(self.model.parameters()))

        self.train_losses = []
        self.test_losses = []

        self.max_rollout_len = cfg.max_rollout_len
        self.steps_per_decision = cfg.steps_per_decision
        self.skip = cfg.skip  # what if we don't predict every observation, but every nth observation?
        self.warmup_len = cfg.warmup_len // self.skip

    def prepare_data(self) -> None:
        for traj in self.train_trajectories:
            state, action, next_obs = self.vectorize_trajectory(traj)
            self.train_data.append((traj.scs, state, action, next_obs))  # TODO: save all SCs with trajectories

        for traj in self.test_trajectories:
            state, action, next_obs = self.vectorize_trajectory(traj)
            self.test_data.append((traj.scs, state, action, next_obs))

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

    def vectorize_trajectory(self, traj: Trajectory) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = tensor(np.expand_dims(traj.get_transitions_attr('state'), axis=0)).to(
            device)  # a (seq_len x obs_size) array  of the states
        action = tensor(np.expand_dims(traj.get_transitions_attr('action'), axis=0)).to(
            device)  # a (seq_len x action_dim) array of actions
        next_obs = tensor(np.expand_dims(traj.get_transitions_attr('next_obs'), axis=0)).to(
            device)  # a (seq_len x obs_size) array  of the next observations

        skip_indices = torch.arange(0, state.size(1), self.skip)

        state = state[:, skip_indices, :]
        action = action[:, skip_indices, :]
        next_obs = next_obs[:, skip_indices, :]

        return state, action, next_obs

    def sample_mini_batch(self, data, batch_size: int = None) -> Tuple[list, torch.Tensor, torch.Tensor, torch.Tensor]:
        traj_lens = [sampled_data[1].size(1) for sampled_data in data]
        probs = [traj_len / sum(traj_lens) for traj_len in traj_lens]

        batch_scs, batch_states, batch_actions, batch_next_obs = [], [], [], []
        while len(batch_states) < batch_size:
            # sampled data is a list of (input, output) pairs
            sampled_data = random.choices(data, weights=probs, k=1)[0]  # zero-index bc choices returns list
            scs, state, action, next_obs = sampled_data
            num_transitions = state.size(1)

            if num_transitions >= self.max_rollout_len:
                start_index = random.randint(0, num_transitions - self.max_rollout_len - 1)

                state = state[:, start_index:start_index + self.max_rollout_len, :]
                action = action[:, start_index:start_index + self.max_rollout_len, :]
                next_obs = next_obs[:, start_index:start_index + self.max_rollout_len, :]

                batch_scs.append(deepcopy(scs[start_index]))
                batch_states.append(state)
                batch_actions.append(action)
                batch_next_obs.append(next_obs)

        batch_states = torch.concat(batch_states, dim=0).to(device)
        batch_actions = torch.concat(batch_actions, dim=0).to(device)
        batch_next_obs = torch.concat(batch_next_obs, dim=0).to(device)

        return batch_scs, batch_states, batch_actions, batch_next_obs

    def train(self):
        print('Preparing data...')
        self.prepare_data()
        print('Training model...')
        pbar = tqdm(range(self.train_itr))
        test_loss = 0

        for itr in pbar:

            if itr == self.train_itr-1:
                self.update(True)
            else:
                self.update(False)
            pbar.set_description("train loss: {:7.6f}, test loss: {:7.6f}".format(
                self.train_losses[-1], test_loss))
            if itr % 10 == 0:
                test_loss = self.test_rollouts()

        return self.train_losses, self.test_losses

    def get_prediction(self, state: torch.Tensor, action: torch.Tensor, with_grad: bool = False):
        x = torch.concat((state, action), dim=1)
        if with_grad:
            y = self.model(x)
        else:
            with torch.no_grad():
                y = self.model(x)
        return y

    def get_next_states(self, scs, predicted_next_obs, actions):
        next_states = []
        for i in range(len(scs)):
            decision_point = 0 % self.steps_per_decision == 0  # not sure if this is entirely right...
            next_state_i = scs[i](predicted_next_obs[i], actions[i])
            next_states.append(torch.unsqueeze(tensor(next_state_i, device), 0))

        next_states = torch.concat(next_states, dim=0)

        return next_states

    def do_rollout(self, batch, with_grad=False, plot=False):
        scs, states, actions, outputs = batch
        scs = deepcopy(scs)
        states_step = states[:, 0, :] # states from the first time step
        loss = 0.0

        predicted_ORPS = []

        # is everything ok here?
        for step in range(0, self.max_rollout_len):
            actions_step = actions[:, step, :]

            predicted_next_endo_obs = self.get_prediction(states_step, actions_step, with_grad=with_grad)

            # maybe we want to look into making these gradients actually work properly!
            next_obs_step = outputs[:, step, :]
            predicted_next_obs = torch.clone(next_obs_step)
            for i, j in enumerate(self.endo_inds):
                predicted_next_obs[:, j] = predicted_next_endo_obs[:, i]
            predicted_ORPS.append(predicted_next_endo_obs.detach().numpy())

            states_step = self.get_next_states(scs, predicted_next_obs, actions_step)
            loss += nn.functional.mse_loss(predicted_next_endo_obs, next_obs_step[:, self.endo_inds])

        predicted_ORPS = np.array(predicted_ORPS)

        if plot:
            batch_size = len(scs)
            predicted_ORPS = np.array(predicted_ORPS)
            for b in range(batch_size):
                plt.plot(actions[b, :, 0], label='actions')
                plt.plot(outputs[b, :, self.endo_inds], label='ORP')
                plt.plot(predicted_ORPS[:, b, 0], label='predicted ORP')
                plt.legend()
                plt.xlabel("Rollout Step")
                plt.savefig(f"test_{b}.png", bbox_inches='tight')
                plt.clf()

        return loss

    def update(self, plot):
        batch = self.sample_mini_batch(self.train_data, batch_size=self.batch_size)
        loss = self.do_rollout(batch, with_grad=True, plot=plot)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_losses.append(loss.detach().numpy())

    def test_rollouts(self):
        num_test = 10
        test_batch = self.sample_mini_batch(self.test_data, batch_size=num_test)
        test_loss = self.do_rollout(test_batch, with_grad=False, plot=True)

        return test_loss

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

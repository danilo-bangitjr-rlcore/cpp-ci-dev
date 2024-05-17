import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from omegaconf import DictConfig

import time
from corerl.component.network.factory import init_custom_network
from corerl.component.optimizers.factory import init_optimizer

import random
from corerl.component.network.utils import tensor, to_np
from corerl.calibration_models.utils import prepare_obs_transition

import torch
import torch.nn as nn


class GRUWithExtraInput(nn.Module):
    def __init__(self, input_dim, extra_input_dim, hidden_dim, output_dim, num_layers=2):
        super(GRUWithExtraInput, self).__init__()

        self.input_dim = input_dim
        self.extra_input_dim = extra_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim + extra_input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        out, _ = self.gru(x, h0)

        out = self.relu(self.fc1(out))
        out = self.fc2(out)

        return out


class GRUWithExtraInput(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GRUWithExtraInput, self).__init__()

        self.input_dim = input_dim
        # self.extra_input_dim = extra_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        out, _ = self.gru(x, h0)

        out = self.relu(self.fc1(out))
        out = self.fc2(out)

        return out

class SimpleGRUCalibrationModel:
    def __init__(self, cfg: DictConfig, **kwargs):
        self.train_trajectories = kwargs['train_trajectories']
        print(len(self.train_trajectories))
        self.test_trajectories = kwargs['test_trajectories']

        reward_func = kwargs['reward_func']
        self.interaction = kwargs['interaction']
        self.state_constructors = kwargs['test_scs']

        self.train_itr = cfg.train_itr
        self.batch_size = cfg.batch_size

        example_transition = self.train_trajectories[0].endo_vars[0]

        state_input_dim = len(example_transition[0])
        action_dim = len(example_transition[1])
        endo_dim = len(example_transition[3])
        exo_dim = len(self.train_trajectories[0].exo_vars[0])
        input_dim = exo_dim + action_dim
        self.model = GRUWithExtraInput(input_dim, 256, endo_dim, num_layers=2)

        self.optimizer = init_optimizer(cfg.optimizer, list(self.model.parameters()))
        self.train_losses = []
        self.test_losses = []

        self.reward_func = reward_func
        self.max_rollout_len = cfg.max_rollout_len
        self.steps_per_decision = cfg.steps_per_decision

    def sample_mini_batch(self, batch_size: int = None) -> dict:
        sampled_data = random.choices(self.train_trajectories, k=batch_size)
        return sampled_data

    def train(self):
        print('Training model...')
        pbar = tqdm(range(self.train_itr))
        for _ in pbar:
            self.update()
            pbar.set_description("train loss: {:7.6f}".format(self.train_losses[-1]))

        self.test_n_rollouts(100)

        return self.train_losses, self.test_losses

    def update(self):
        # is there a way to parallelize rollouts?
        batch = self.sample_mini_batch(batch_size=self.batch_size)
        loss = self.train_rollout(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_losses.append(loss.detach().numpy())

    def prepare_batch(self, batch):
        rollout_len = min([traj.num_transitions() for traj in batch])
        all_transitions = [traj.endo_vars for traj in batch]
        batch_idxs = range(len(batch))

        inputs = list()
        expected_outputs = list()

        initial_states = [all_transitions[traj][0][0].reshape(1, -1) for traj in batch_idxs]
        initial_states = tensor(np.concatenate(initial_states, axis=0)).unsqueeze(1)

        for traj in batch:
            exo_obs = [traj.exo_vars[step].reshape(1, -1) for step in range(rollout_len)]
            exo_obs = np.concatenate(exo_obs, axis=0)

            actions = [traj.transitions[step][1].reshape(1, -1) for step in range(rollout_len)]
            actions = np.concatenate(actions, axis=0)

            exo_action = np.concatenate((exo_obs, actions), axis=1)
            exo_action = np.expand_dims(exo_action, axis=0)
            inputs.append(exo_action)

            endo_obs = [traj.endo_vars[step][3].reshape(1, -1) for step in range(rollout_len)]
            endo_obs = np.concatenate(endo_obs, axis=0)
            endo_obs = np.expand_dims(endo_obs, axis=0)
            expected_outputs.append(endo_obs)

        initial_states = torch.repeat_interleave(initial_states, rollout_len, dim=1)

        inputs = tensor(np.concatenate(inputs, axis=0))
        # inputs = torch.cat((initial_states, inputs), dim=2)
        expected_outputs = tensor(np.concatenate(expected_outputs, axis=0))
        return inputs,  expected_outputs

    def train_rollout(self, batch):
        inputs, expected_out = self.prepare_batch(batch)
        out = self.model(inputs)
        loss = nn.functional.mse_loss(out, expected_out)
        return loss

    def test_rollout(self, traj):
        # validates the model's accurary on a test rollout
        losses = []
        rollout_len = min(traj.num_transitions(), self.max_rollout_len)
        inputs, expected_out = self.prepare_batch([traj])

        with torch.no_grad():
            out = self.model(inputs)

        for step in range(rollout_len):
            loss = torch.nn.functional.l1_loss(out[0, step, :], expected_out[0, step, :]).detach().numpy()
            losses.append(loss)

        return losses

    def test_n_rollouts(self, n):
        import matplotlib.pyplot as plt
        for _ in range(n):
            test_traj = random.choice(self.test_trajectories)
            losses = self.test_rollout(test_traj)
            plt.plot(np.array(losses), c='b', alpha=0.2)  # * (471.20947 - 2.6265918e+02)

        plt.ylabel("Absolute Error From True ORP")
        plt.xlabel("Rollout Step")
        plt.show()

    def get_prediction(self, state, action, with_grad=False):
        x = torch.concat((state, action), dim=1)
        if with_grad:
            y = self.model(x)
        else:
            with torch.no_grad():
                y = self.model(x)
        return y

    def _model_step(self, state, action, with_grad=False):
        obs = self.get_prediction(tensor(state).reshape((1, -1)),
                                  tensor(action).reshape((1, -1)),
                                  with_grad=with_grad)
        return obs

    def do_rollout(self, state, sc, agent, rollout_len=20):
        gamma = agent.gamma
        g = 0  # the return
        prev_action = None
        for i in range(rollout_len):
            action = agent.get_action(state)
            obs = self._model_step(state, action)
            state = sc(obs)

            reward_info = {}
            if prev_action is None:
                reward_info['prev_action'] = action
            else:
                reward_info['prev_action'] = prev_action
            reward_info['curr_action'] = action

            denormalized_obs = self.interaction.obs_normalizer.denormalize(obs)
            g += gamma * self.reward_func(denormalized_obs, **reward_info)
            prev_action = action

        # This does not factor in truncs, or dones. Should it?
        return g

    def do_n_rollouts(self, agent, num_rollouts=100, rollout_len=20):
        returns = []
        for rollout in range(num_rollouts):
            rand_idx = random.randint(0, len(self.state_constructors))
            start_transition = self.test_transitions[rand_idx]
            start_state = start_transition[0]
            start_sc = self.state_constructors[rand_idx]

            return_rollout = self.do_rollout(start_state, start_sc, agent, rollout_len=rollout_len)
            returns.append(return_rollout)

        return returns

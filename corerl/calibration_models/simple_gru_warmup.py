import numpy as np
import random
import matplotlib.pyplot as plt

from tqdm import tqdm
from omegaconf import DictConfig

from corerl.component.optimizers.factory import init_optimizer
from corerl.component.network.utils import tensor
from corerl.utils.device import device
from corerl.calibration_models.utils import Trajectory
from corerl.component.network.networks import GRU

import torch
import torch.nn as nn


class GRUCalibrationModel:
    def __init__(self, cfg: DictConfig, **kwargs):
        self.train_trajectories = kwargs['train_trajectories']
        self.test_trajectories = kwargs['test_trajectories']
        self.train_data = []  # pre-processed version of self.train_trajectories
        self.test_data = [] # pre-processed version of self.train_trajectories

        reward_func = kwargs['reward_func']
        self.interaction = kwargs['interaction']
        self.state_constructors = kwargs['test_scs']

        self.train_itr = cfg.train_itr
        self.batch_size = cfg.batch_size

        example_trajectory = self.train_trajectories[0]
        action_dim = example_trajectory.action_dim
        endo_dim = example_trajectory.endo_dim
        exo_dim = example_trajectory.exo_dim
        input_dim = endo_dim + exo_dim + action_dim

        # we are only predicting the endogenous variables at the next time step, so output_dim = endo_dim
        self.model = GRU(cfg.model, input_dim, endo_dim)
        self.optimizer = init_optimizer(cfg.optimizer, list(self.model.parameters()))
        self.train_losses = []
        self.test_losses = []

        self.reward_func = reward_func
        self.max_rollout_len = cfg.max_rollout_len
        self.steps_per_decision = cfg.steps_per_decision
        self.warmup_len = cfg.warmup_len

    def sample_mini_batch(self, batch_size: int = None) -> tuple[torch.Tensor, torch.Tensor]:
        sampled_data = random.choices(self.train_data, k=batch_size)  # sampled data is a list of (input, output) pairs

        inputs, outputs = list(zip(*sampled_data))
        seq_lens = [input_tensor.shape[1] for input_tensor in inputs]
        min_seq_len = min(seq_lens)

        input_tensors = [input_tensor[:, 0:min_seq_len, :] for input_tensor in inputs]
        input_tensor = torch.concat(input_tensors, dim=0).to(device)

        output_tensors = [output_tensor[:, 0:min_seq_len, :] for output_tensor in outputs]
        output_tensor = torch.concat(output_tensors, dim=0).to(device)

        return input_tensor, output_tensor

    def prepare_data(self):
        for traj in self.train_trajectories:
            inputs, outputs = self.vectorize_trajectory(traj)
            self.train_data.append((inputs, outputs))

        for traj in self.test_trajectories:
            inputs, outputs = self.vectorize_trajectory(traj)
            self.test_data.append((inputs, outputs))

    def vectorize_trajectory(self, traj: Trajectory):
        rollout_len = traj.num_transitions

        exo_obs = [traj.exo_vars[step].reshape(1, -1) for step in range(rollout_len)]
        exo_obs = np.concatenate(exo_obs, axis=0)

        actions = [traj.transitions[step][1].reshape(1, -1) for step in range(rollout_len)]
        actions = np.concatenate(actions, axis=0)

        # there is one additional endo_var, for the next observed endogenous variable
        endo_obs = [traj.endo_vars[step].reshape(1, -1) for step in range(len(traj.endo_vars))]
        endo_obs = np.concatenate(endo_obs, axis=0)
        # we will exclude all but the last element of endo_obs for the inputs
        endo_obs_in = endo_obs[0:-1].copy()
        # after the warm-up period has ended, stop giving the model new information about the endogenous variables
        # all future endogenous observations will be equal to the self.warmup_len-th observation:
        # i.e. at index self.warmup_len-1
        endo_obs_in[self.warmup_len - 1:] = endo_obs_in[self.warmup_len - 1]

        # for the outputs, will take all but the first
        endo_obs_out = endo_obs[1:]

        inputs = np.concatenate((endo_obs_in, exo_obs, actions), axis=1)
        inputs = tensor(np.expand_dims(inputs, axis=0)).to(device)

        # outputs are offset by one from the inputs
        outputs = tensor(np.expand_dims(endo_obs_out, axis=0)).to(device)

        assert inputs.size(1) == outputs.size(1)
        return inputs, outputs

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
        loss = self.train_rollout(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_losses.append(loss.detach().numpy())

    def train_rollout(self, batch):
        inputs, outputs = batch
        predicted_out = self.model(inputs)
        loss = nn.functional.mse_loss(predicted_out, outputs)
        return loss

    def test_rollouts(self):
        all_losses = []
        for test_data in self.test_data:
            losses = self.test_rollout(test_data)
            all_losses.append(np.mean(losses[self.warmup_len - 1:]))

        return np.mean(all_losses)
        #    plt.plot(np.array(losses), c='b', alpha=0.2)
        #
        # plt.ylabel("Absolute Error From True ORP")
        # plt.xlabel("Rollout Step")
        # plt.show()

    def test_rollout(self, traj):
        # validates the model's accurary on a test rollout
        inputs, outputs = traj
        losses = []
        with torch.no_grad():
            predicted_out = self.model(inputs, prediction_start=self.warmup_len - 1)

        rollout_len = predicted_out.size(1)
        assert predicted_out.size(1) == outputs.size(1)
        for step in range(rollout_len):
            loss = torch.nn.functional.l1_loss(predicted_out[0, step, :], outputs[0, step, :]).detach().numpy()
            losses.append(loss)

        return losses

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

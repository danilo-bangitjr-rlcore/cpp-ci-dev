import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from omegaconf import DictConfig
from corerl.component.buffer.factory import init_buffer

from corerl.component.network.factory import init_custom_network
from corerl.component.optimizers.factory import init_optimizer

import random
from corerl.component.network.utils import tensor, to_np
from corerl.calibration_models.utils import prepare_obs_transition


class SimpleCalibrationModel:
    def __init__(self, cfg: DictConfig, **kwargs):
        train_transitions = kwargs['train_obs_transitions']
        self.test_trajectories = kwargs['test_trajectories']
        self.reward_func = kwargs['reward_func']
        self.interaction = kwargs['interaction']
        self.state_constructors = kwargs['test_scs']

        self.buffer = init_buffer(cfg.buffer)
        self.test_buffer = init_buffer(cfg.buffer)
        self.train_itr = cfg.train_itr
        self.batch_size = cfg.batch_size

        self.buffer.load(train_transitions)

        input_dim = len(train_transitions[0][0])
        action_dim = len(train_transitions[0][1])
        output_dim = len(train_transitions[0][3])

        self.model = init_custom_network(cfg.model, input_dim=input_dim + action_dim, output_dim=output_dim)
        self.optimizer = init_optimizer(cfg.optimizer, list(self.model.parameters()))

        self.train_losses = []
        self.test_losses = []

        self.max_rollout_len = cfg.max_rollout_len
        self.steps_per_decision = cfg.steps_per_decision

    def update(self):
        batch = self.buffer.sample_mini_batch(self.batch_size)
        state_batch, action_batch, next_state_batch = batch['states'], batch['actions'], batch['next_states']
        prediction = self.get_prediction(state_batch, action_batch, with_grad=True)
        loss = nn.functional.mse_loss(prediction, next_state_batch)

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

        self.test_n_rollouts(100)

        return self.train_losses, self.test_losses

    def test_rollout(self, traj):
        # validates the model's accurary on a test rollout
        transitions = traj.endo_vars
        sc = deepcopy(traj.start_sc)
        losses = []
        rollout_len = min(traj.num_transitions(), self.max_rollout_len)

        for step in range(rollout_len):
            t = transitions[step]
            exo_obs = traj.exo_vars[step]

            if step == 0:  # on the first iteration, get the state from the
                state, action, endo_obs = prepare_obs_transition(t)
            else:
                _, action, endo_obs = prepare_obs_transition(t)

            if step > 0:
                loss_step = np.mean(np.abs(endo_obs - predicted_endo_obs))
                losses.append(loss_step)

            predicted_endo_obs = self._model_step(state, action).reshape(-1)
            obs = np.concatenate((predicted_endo_obs, exo_obs), axis=0)
            decision_point = step % self.steps_per_decision == 0
            state = sc(obs, action, decision_point=decision_point)

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

    def _model_step(self, state, action):
        obs = self.get_prediction(tensor(state).reshape((1, -1)),
                                  tensor(action).reshape((1, -1)),
                                  with_grad=False)

        obs = to_np(obs)
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

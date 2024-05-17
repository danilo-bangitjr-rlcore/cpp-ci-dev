import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from omegaconf import DictConfig

from corerl.component.network.factory import init_custom_network
from corerl.component.optimizers.factory import init_optimizer

import random
from corerl.component.network.utils import tensor, to_np
from corerl.calibration_models.utils import prepare_obs_transition


class SimpleNStepCalibrationModel:
    def __init__(self, cfg: DictConfig, **kwargs):
        self.train_trajectories = kwargs['train_trajectories']
        self.test_trajectories = kwargs['test_trajectories']

        reward_func = kwargs['reward_func']
        self.interaction = kwargs['interaction']
        self.state_constructors = kwargs['test_scs']

        self.train_itr = cfg.train_itr
        self.batch_size = cfg.batch_size

        example_transition = self.train_trajectories[0].endo_vars[0]
        input_dim = len(example_transition[0])
        action_dim = len(example_transition[1])
        output_dim = len(example_transition[3])

        self.model = init_custom_network(cfg.model, input_dim=input_dim + action_dim, output_dim=output_dim)
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
        scs = [deepcopy(traj.start_sc) for traj in batch]
        states = []
        actions = []
        endo_obs = []
        exo_obs = []

        all_transitions = [traj.endo_vars for traj in batch]
        batch_idxs = range(len(batch))
        for step in range(rollout_len):
            states_step = [all_transitions[traj][step][0].reshape(1, -1) for traj in batch_idxs]
            states.append(tensor(np.concatenate(states_step, axis=0)))

            actions_step = [all_transitions[traj][step][1].reshape(1, -1) for traj in batch_idxs]
            actions.append(tensor(np.concatenate(actions_step, axis=0)))

            endo_obs_step = [all_transitions[traj][step][3].reshape(1, -1) for traj in batch_idxs]
            endo_obs.append(tensor(np.concatenate(endo_obs_step, axis=0)))

            exo_obs_step = [traj.exo_vars[step].reshape(1, -1) for traj in batch]
            exo_obs.append(np.concatenate(exo_obs_step, axis=0))  # doesn't need to be a tensor

        return states, actions, endo_obs, exo_obs, scs

    def train_rollout(self, batch):
        loss = 0.0
        rollout_len = min([traj.num_transitions() for traj in batch])
        states, actions, endo_obs, exo_obs, scs = self.prepare_batch(batch)

        for step in range(rollout_len):
            if step == 0:  # on the first iteration, get the state from the
                states_step, actions_step, endo_obs_step = states[step], actions[step], endo_obs[step]
            else:
                _, actions_step, endo_obs_step = states[step], actions[step], endo_obs[step]

            if step > 0:
                loss += (1/rollout_len)*nn.functional.mse_loss(endo_obs_step, predicted_endo_obs)

            predicted_endo_obs = self.get_prediction(states_step, actions_step, with_grad=True)
            predicted_endo_obs_np = predicted_endo_obs.detach().numpy()

            obs_step = np.concatenate((predicted_endo_obs_np, exo_obs[step]), axis=1)  # array of observations
            decision_point = step % self.steps_per_decision == 0

            # update state constructor
            actions_step_np = actions_step.detach().numpy()
            states_list = [scs[i](obs_step[i, :], actions_step_np[i, :], decision_point=decision_point).reshape(1, -1)
                           for i in range(len(scs))]
            states_step = tensor(np.concatenate(states_list, axis=0))

        return loss

    # def train_rollout(self, traj):
    #     # validates the model's accurary on a test rollout
    #     transitions = traj.endo_transitions
    #     sc = deepcopy(traj.start_sc)
    #     loss = 0.0
    #     rollout_len = min(traj.num_transitions(), self.max_rollout_len)
    #
    #     for step in range(rollout_len):
    #         t = transitions[step]
    #         exo_obs = traj.exo_variables[step]
    #
    #         if step == 0:  # on the first iteration, get the state from the
    #             state, action, endo_obs = prepare_obs_transition(t)
    #         else:
    #             _, action, endo_obs = prepare_obs_transition(t)
    #
    #         if step > 0:
    #             loss += (1 / rollout_len) * nn.functional.mse_loss(tensor(endo_obs), predicted_endo_obs)
    #
    #         predicted_endo_obs = self._model_step(state, action, with_grad=True).reshape(-1)
    #         predicted_endo_obs_np = predicted_endo_obs.detach().numpy()
    #         obs = np.concatenate((predicted_endo_obs_np, exo_obs), axis=0)
    #         decision_point = step % self.steps_per_decision == 0
    #         state = sc(obs, action, decision_point=decision_point)
    #
    #     return loss

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
                loss_step = np.mean(np.abs(endo_obs - predicted_endo_obs.detach().numpy()))
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

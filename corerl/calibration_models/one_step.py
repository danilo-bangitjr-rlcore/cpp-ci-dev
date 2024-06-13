import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from omegaconf import DictConfig
from corerl.component.buffer.factory import init_buffer

from corerl.component.network.factory import init_custom_network
from corerl.component.optimizers.factory import init_optimizer

from corerl.component.network.utils import tensor, to_np
import matplotlib.pyplot as plt


class SimpleCalibrationModel:
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

    def do_test_rollout(self, traj, num, start):
        # validates the model's accuracy on a test rollout
        transitions = traj.transitions[start:]
        sc = deepcopy(traj.scs[start])  # cuz this is completely wrong hey!
        # print("initial trace", sc.sc.parents[1].trace)
        losses = []
        rollout_len = min(traj.num_transitions, self.max_rollout_len)

        state = transitions[0].state

        orps = []
        predicted_orps = []
        actions = []

        for step in range(rollout_len):
            t = transitions[step]
            action = t.action
            next_obs = t.next_obs
            next_endo_obs = next_obs[self.endo_inds]

            state_tensor = tensor(state).reshape((1, -1))
            action_tensor = tensor(action).reshape((1, -1))

            predicted_next_endo_obs = self.get_prediction(state_tensor, action_tensor)

            actions.append(action)
            orps.append(next_obs[0])
            predicted_orps.append(predicted_next_endo_obs)

            # log the loss
            loss_step = np.mean(np.abs(next_endo_obs - to_np(predicted_next_endo_obs)))
            losses.append(loss_step)

            # construct a fictitous observation using the predicted endogenous variables and the actual
            # exogenous variables
            new_obs = next_obs.copy()
            for i, j in enumerate(self.endo_inds):
                new_obs[j] = predicted_next_endo_obs[i]

            decision_point = step % self.steps_per_decision == 0
            state = sc(new_obs, action, decision_point=decision_point) # TODO there is a bug here I think

        plt.plot(orps, label='orps')
        plt.plot(actions, label='actions')

        predicted_orps = [np.squeeze(to_np(p)) for p in predicted_orps]
        plt.plot(predicted_orps, label='predicted orps')
        plt.legend()

        plt.xlabel("Rollout Step")
        plt.savefig(f"test_{num}_{start}.png", bbox_inches='tight')
        plt.clf()

        return losses

    def do_test_rollouts(self):
        import matplotlib.pyplot as plt
        import random


        for n, test_traj in enumerate(self.test_trajectories):
            last = test_traj.num_transitions - self.max_rollout_len
            num_rollouts = 20
            increase_idx = last // num_rollouts
            start_idx = 0
            for start in range(num_rollouts):
                # start_idx = random.choice(range(0, test_traj.num_transitions-self.max_rollout_len))
                losses = self.do_test_rollout(test_traj, n,  start_idx)
                start_idx += increase_idx


        #         plt.plot(np.array(losses), c='b', alpha=0.2)  # * (471.20947 - 2.6265918e+02)
        # plt.ylabel("Absolute Error From True ORP")
        # plt.xlabel("Rollout Step")
        # plt.show()

    def get_prediction(self, state: torch.Tensor, action: torch.Tensor, with_grad: bool = False):
        x = torch.concat((state, action), dim=1)
        if with_grad:
            y = self.model(x)
        else:
            with torch.no_grad():
                y = self.model(x)
        return y

    def do_agent_rollout(self, traj, agent, rollout_len=20):
        gamma = agent.gamma
        g = 0  # the return
        prev_action = None

        transitions = traj.transitions
        sc = deepcopy(traj.start_sc)
        state = transitions[0].state
        for step in range(rollout_len):
            t = transitions[step]

            action = agent.get_action(state)
            next_obs = t.next_obs

            state_tensor = tensor(state).reshape((1, -1))
            action_tensor = tensor(action).reshape((1, -1))

            predicted_next_endo_obs = self.get_prediction(state_tensor, action_tensor)

            new_obs = next_obs.copy()
            for i, j in enumerate(self.endo_inds):
                new_obs[j] = predicted_next_endo_obs[i]

            decision_point = step % self.steps_per_decision == 0
            state = sc(new_obs, action, decision_point=decision_point)

            reward_info = {}
            if prev_action is None:
                reward_info['prev_action'] = action
            else:
                reward_info['prev_action'] = prev_action
            reward_info['curr_action'] = action

            denormalized_obs = self.interaction.obs_normalizer.denormalize(new_obs)
            g += gamma * self.reward_func(denormalized_obs, **reward_info)
            prev_action = action

        return g

    def do_agent_rollouts(self, agent):
        returns = []
        for test_traj in self.test_trajectories:
            rollout_return = self.do_agent_rollout(test_traj, agent)
            returns.append(rollout_return)

        return returns

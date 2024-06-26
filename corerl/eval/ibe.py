"""
Implements Identifiable BE Selection (without the selection) from
https://drive.google.com/drive/u/1/folders/1tJo78FvsWfWaPncJNNyI9IO1f7UbxCFR
"""
import torch
from torch.nn.functional import mse_loss
from omegaconf import DictConfig

from corerl.component.network.factory import init_custom_network
from corerl.component.optimizers.factory import init_optimizer
from corerl.eval.base_eval import BaseEval
from corerl.data.data import TransitionBatch, Transition


class IBE(BaseEval):
    def __init__(self, cfg: DictConfig, **kwargs):
        if 'agent' not in kwargs:
            raise KeyError("Missing required argument: 'agent'")

        self.agent = kwargs['agent']
        state_dim = self.agent.state_dim
        action_dim = self.agent.action_dim
        self.n_updates = cfg.n_updates
        self.gamma = cfg.gamma
        self.model = init_custom_network(cfg.network, state_dim + action_dim, output_dim=1)
        self.optimizer = init_optimizer(cfg.optimizer, param=self.model.parameters(), ensemble=False)
        self.losses = []
        self.bes = []  # the bellman errors

    def get_delta(self, batch: TransitionBatch) -> torch.Tensor:
        # TODO: this does not work with anytime. Need to update
        state_batch = batch.state
        action_batch = batch.action
        reward_batch = batch.reward
        next_state_batch = batch.boot_state
        mask_batch = 1 - batch.terminated

        q = self.agent.q_critic.get_q(state_batch, action_batch, with_grad=False)
        next_actions, _ = self.agent.actor.get_action(next_state_batch, with_grad=False)
        next_q = self.agent.q_critic.get_q(next_state_batch, next_actions,
                                           with_grad=False)  # TODO: what if the agent does not have a q_critic?
        bootstrapped_return = reward_batch + mask_batch * self.gamma * next_q  # the target of the return
        delta = bootstrapped_return - q
        return delta

    def get_loss(self, batch: TransitionBatch) -> torch.Tensor:
        delta = self.get_delta(batch)
        state_batch = batch.state
        action_batch = batch.action
        sa = torch.concatenate((state_batch, action_batch), dim=1)
        predictions = self.model(sa)
        loss = mse_loss(delta, predictions)
        return loss

    def do_eval(self, **kwargs) -> None:
        # train the model
        for _ in range(self.n_updates):
            batch = self.agent.buffer.sample()
            loss = self.get_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss = loss.detach().item()
            self.losses.append(loss)

        # estimate the bellman error on a batch
        batch = self.agent.buffer.sample()
        be = self.estimate_be(batch)
        self.bes.append(be)

    def estimate_be(self, batch: TransitionBatch) -> torch.Tensor:
        state_batch = batch.state
        action_batch = batch.action

        sa = torch.concatenate((state_batch, action_batch), dim=1)
        predictions = self.model(sa)
        delta = self.get_delta(batch)

        be_batch = 2 * predictions * delta - torch.square(delta)
        mean_be = torch.mean(be_batch)
        return mean_be.detach().item()

    def get_stats(self) -> dict:
        stats = {
            'bellman_error': self.bes,
            'last_bellman_error': self.bes[-1]
        }
        return stats

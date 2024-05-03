"""
Implements Identifiable BE Selection (without the selection) from
https://drive.google.com/drive/u/1/folders/1tJo78FvsWfWaPncJNNyI9IO1f7UbxCFR
"""
import torch
from torch.nn.functional import mse_loss

from corerl.component.network.factory import init_custom_network
from corerl.component.optimizers.factory import init_optimizer
from corerl.eval.base_eval import BaseEval


class IBE(BaseEval):
    def __init__(self, cfg, **kwargs):
        if 'agent' not in kwargs:
            raise KeyError("Missing required argument: 'agent'")

        self.agent = kwargs['agent']
        state_dim = self.agent.state_dim
        action_dim = self.agent.action_dim
        self.n_updates = cfg.n_updates
        self.gamma = cfg.gamma
        self.model = init_custom_network(cfg.network, state_dim + action_dim, output_dim=1)
        self.optimizer = init_optimizer(cfg.optimizer, param=self.model.parameters(), ensemble=False)

    def get_loss(self, batch):
        state_batch = batch['states']
        action_batch = batch['actions']
        reward_batch = batch['rewards']
        next_state_batch = batch['next_states']
        mask_batch = 1 - batch['dones']

        next_actions, _ = self.agent.actor.get_action(next_state_batch, with_grad=False)
        # TODO: what if the agent does not have a q_critic?
        next_q = self.agent.q_critic.get_q_target(next_state_batch, next_actions)
        target = reward_batch + mask_batch * self.gamma * next_q

        sa = torch.concatenate((state_batch, action_batch), dim=1)
        predictions = self.model(sa)

        loss = mse_loss(target, predictions)
        return loss

    def do_eval(self, **kwargs):
        losses = []
        for _ in range(self.n_updates):
            batch = self.agent.buffer.sample()
            loss = self.get_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss = loss.detach().item()

    def get_stats(self):
        pass

import torch

from root.component.actor.base_actor import BaseActor
from root.component.network.factory import init_actor_network
from root.component.optimizers.factory import init_optimizer


class NetworkActor(BaseActor):
    def __init__(self, cfg, state_dim, action_dim):
        self.model = init_actor_network(cfg.network, state_dim, action_dim)
        self.optimizer = init_optimizer(cfg.optimizer, self.model.parameters())

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state, with_grad=False):
        if with_grad:
            return self.model.forward(state)
        else:
            with torch.no_grad():
                return self.model.forward(state)

    def get_log_prob(self, states, actions):
        return self.model.log_prob(states, actions)
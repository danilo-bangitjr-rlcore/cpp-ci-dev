import torch
from omegaconf import DictConfig

from root.component.actor.base_actor import BaseActor
from root.component.network.factory import init_actor_network
from root.component.optimizers.factory import init_optimizer


class NetworkActor(BaseActor):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        self.model = init_actor_network(cfg.actor_network, state_dim, action_dim)
        self.optimizer = init_optimizer(cfg.actor_optimizer, self.model.parameters())

    def update(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state: torch.Tensor, with_grad=False) -> (torch.Tensor, dict):

        # if this is discrete control, something will change.
        if with_grad:
            return self.model.forward(state)
        else:
            with torch.no_grad():
                return self.model.forward(state)

    def get_log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> (torch.Tensor, dict):
        return self.model.log_prob(states, actions)
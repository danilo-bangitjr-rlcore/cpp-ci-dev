import torch
from omegaconf import DictConfig

from root.component.critic.base_critic import BaseQ, BaseV
from root.component.optimizers.factory import init_optimizer
from root.component.network.factory import init_critic_network


class EnsembleQCritic(BaseQ):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        self.model = init_critic_network(cfg.network, state_dim, action_dim)
        self.target = init_critic_network(cfg.network, state_dim, action_dim)
        self.optimizer = init_optimizer(cfg.optimizer, self.model.parameters())  # TODO: change this optimizer to True
        self.polyak = cfg.polyak
        self.target_sync_freq = cfg.target_sync_freq
        self.target_sync_counter = 0

    def get_q(self, states: torch.Tensor, actions: torch.Tensor, with_grad: bool = False, get_qs: bool = False) -> torch.Tensor | (torch.Tensor, torch.Tensor):
        state_actions = torch.concat((states, actions), dim=1)
        if with_grad:
            q, qs = self.model(state_actions)
        else:
            with torch.no_grad():
                q, qs = self.model(state_actions)

        if get_qs:
            return q, qs
        else:
            return q

    def get_q_target(self, states: torch.Tensor, actions: torch.Tensor, get_qs: bool = False) -> torch.Tensor | (torch.Tensor, torch.Tensor):
        state_actions = torch.concat((states, actions), dim=1)
        with torch.no_grad():
            q, qs = self.target(state_actions)

        if get_qs:
            return q, qs
        else:
            return q

    def update(self, loss: torch.Tensor) -> None:
        loss.backward()
        self.optimizer.step()
        if self.target_sync_counter % self.target_sync_freq == 0:
            self.sync_target()
            self.target_sync_counter = 0

    def sync_target(self) -> None:
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)


class EnsembleVCritic(BaseV):
    def __init__(self, cfg: DictConfig, state_dim: int):
        self.model = init_critic_network(cfg.network, state_dim, output_dim=1)
        self.target = init_critic_network(cfg.network, state_dim, output_dim=1)
        self.optimizer = init_optimizer(cfg.optimizer, self.model.parameters())
        self.polyak = cfg.polyak
        self.target_sync_freq = cfg.target_sync_freq
        self.target_sync_counter = 0

    def get_v(self, states: torch.Tensor, with_grad: bool = False, get_vs: bool = False) -> torch.Tensor | (torch.Tensor, torch.Tensor):
        if with_grad:
            v, vs = self.model(states)
        else:
            with torch.no_grad():
                v, vs = self.model(states)

        if get_vs:
            return v, vs
        else:
            return v

    def get_v_target(self, states: torch.Tensor, get_vs: bool = False) -> torch.Tensor | (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            v, vs = self.target(states)

        if get_vs:
            return v, vs
        else:
            return v

    def update(self, loss: torch.Tensor) -> None:
        loss.backward()
        self.optimizer.step()
        if self.target_sync_counter % self.target_sync_freq == 0:
            self.sync_target()
            self.target_sync_counter = 0

    def sync_target(self) -> None:
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

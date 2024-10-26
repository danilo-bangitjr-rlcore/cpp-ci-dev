from omegaconf import DictConfig
import torch
import torch.nn as nn
import ctypes
from typing import Optional, Callable
from corerl.component.network.utils import clone_model_0to1
from corerl.component.exploration.base import BaseExploration
from corerl.component.network.factory import init_custom_network
from corerl.component.optimizers.linesearch_optimizer import LineSearchOpt
from corerl.component.optimizers.factory import init_optimizer


class RndNetworkExplore(BaseExploration):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg)
        self.gamma = cfg.gamma
        # Networks for uncertainty measure
        self.ftrue0 = init_custom_network(cfg.exploration_network, state_dim+action_dim, state_dim+action_dim)
        self.ftrue1 = init_custom_network(cfg.exploration_network, state_dim+action_dim, state_dim+action_dim)

        self.fbonus0 = init_custom_network(cfg.exploration_network, state_dim+action_dim, state_dim+action_dim)
        self.fbonus1 = init_custom_network(cfg.exploration_network, state_dim+action_dim, state_dim+action_dim)

        # Ensure the nonlinear net between learning net and target net are the same
        clone_model_0to1(self.ftrue0.random_network, self.fbonus0.random_network)
        clone_model_0to1(self.ftrue1.random_network, self.fbonus1.random_network)

        self.optimizer0 = init_optimizer(cfg.exploration_optimizer, self.fbonus0.parameters())
        self.optimizer1 = init_optimizer(cfg.exploration_optimizer, self.fbonus0.parameters())
        self.random_policy = init_custom_network(cfg.policy_network, state_dim, action_dim)

    def set_parameters(self, buffer_address: int, eval_error_fn: Optional['Callable'] = None) -> None:
        self.buffer = ctypes.cast(buffer_address, ctypes.py_object).value

    def explore_bonus_loss(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
                             next_action: torch.Tensor, mask: torch.Tensor, gamma: float) -> list[torch.Tensor]:
        in_ = torch.concat((state, action), dim=1)
        in_p1 = torch.concat((next_state, next_action), dim=1)
        with torch.no_grad():
            true0_t, _ = self.ftrue0(in_)
            true1_t, _ = self.ftrue1(in_)
            true0_tp1, _ = self.ftrue0(in_p1)
            true1_tp1, _ = self.ftrue1(in_p1)
        reward0 = true0_t - mask * gamma * true0_tp1
        reward1 = true1_t - mask * gamma * true1_tp1

        pred0, _ = self.fbonus0(in_)
        pred1, _ = self.fbonus1(in_)
        with torch.no_grad():
            target0 = reward0.detach() + mask * gamma * self.fbonus0(in_p1)[0] #true0_t #
            target1 = reward1.detach() + mask * gamma * self.fbonus1(in_p1)[0] #true1_t #

        loss0 = nn.functional.mse_loss(pred0, target0)
        loss1 = nn.functional.mse_loss(pred1, target1)
        return [loss0, loss1]

    def get_exploration_bonus(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        in_ = torch.concat((state, action), dim=1)
        with torch.no_grad():
            pred0, _ = self.fbonus0(in_)
            pred1, _ = self.fbonus1(in_)
            true0, _ = self.ftrue0(in_)
            true1, _ = self.ftrue1(in_)
        b, _ = torch.max(torch.concat([torch.abs(pred0 - true0), torch.abs(pred1 - true1)], dim=1),
                         dim=1, keepdim=True)
        return b.detach()

    def update(self) -> None:
        batch = self.buffer.sample()
        state_batch = batch.state
        action_batch = batch.action
        next_state_batch = batch.boot_state
        mask_batch = 1 - batch.terminated
        random_next_action, _ = self.random_policy(next_state_batch)
        loss0, loss1 = self.explore_bonus_loss(state_batch, action_batch, next_state_batch,
                                               random_next_action, mask_batch, self.gamma)
        self.optimizer0.zero_grad()
        loss0.backward()
        self.optimizer0.step(closure=lambda: 0.)
        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step(closure=lambda: 0.)

    def get_networks(self) -> list[torch.nn.Module]:
        return [self.fbonus0, self.fbonus1]


class RndNetworkExploreLineSearch(RndNetworkExplore):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int):
        super().__init__(cfg, state_dim, action_dim)
        self.fbonus0_copy = init_custom_network(cfg.exploration_network, state_dim+action_dim, state_dim+action_dim)
        self.fbonus1_copy = init_custom_network(cfg.exploration_network, state_dim+action_dim, state_dim+action_dim)

        clone_model_0to1(self.fbonus0, self.fbonus0_copy)
        clone_model_0to1(self.fbonus1, self.fbonus1_copy)
        self.optimizer = LineSearchOpt(cfg.exploration_optimizer, [self.fbonus0, self.fbonus1],
                                       cfg.exploration_optimizer.lr, cfg.max_backtracking,
                                       cfg.error_threshold, cfg.lr_lower_bound,
                                       cfg.exploration_optimizer.name)

    def set_parameters(self, buffer_address: int, eval_error_fn: Optional['Callable'] = None) -> None:
        assert eval_error_fn is None # Define the eval function inside the class
        self.optimizer.set_params(buffer_address, [self.fbonus0_copy, self.fbonus1_copy],
                                  self.exploration_eval_error_fn)
        self.buffer = ctypes.cast(buffer_address, ctypes.py_object).value

    def exploration_eval_error_fn(self, args: list[torch.Tensor]) -> torch.Tensor:
        state, action, _, _, _ = args
        return self.get_exploration_bonus(state, action).mean()

    def update(self) -> None:
        batch = self.buffer.sample()
        state_batch = batch.state
        action_batch = batch.action
        next_state_batch = batch.boot_state
        mask_batch = 1 - batch.terminated
        random_next_action, _ = self.random_policy(next_state_batch)
        loss0, loss1 = self.explore_bonus_loss(state_batch, action_batch, next_state_batch,
                                               random_next_action, mask_batch, self.gamma)
        self.optimizer.zero_grad()
        loss0.backward()
        loss1.backward()
        self.optimizer.step()

    def get_networks(self) -> list[torch.nn.Module]:
        return [self.fbonus0, self.fbonus1, self.fbonus0_copy, self.fbonus1_copy]

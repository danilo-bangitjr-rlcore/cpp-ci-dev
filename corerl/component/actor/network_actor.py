import torch
from pathlib import Path
from omegaconf import DictConfig
from typing import Optional, Callable

import corerl.component.policy as policy
from corerl.component.actor.base_actor import BaseActor
from corerl.component.network.factory import init_actor_network
from corerl.component.optimizers.factory import init_optimizer
from corerl.component.optimizers.linesearch_optimizer import LineSearchOpt
from corerl.utils.device import device


DEFAULT_ACTION_MIN = 0
DEFAULT_ACTION_MAX = 1


class NetworkActor(BaseActor):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int, initializer: Optional['NetworkActor'] = None):
        # We always assume actions are normalized in (0, 1) unless otherwise
        # stated
        action_min = cfg.get("action_min", DEFAULT_ACTION_MIN)
        action_max = cfg.get("action_max", DEFAULT_ACTION_MAX)

        self.policy = policy.create(
            cfg.actor_network, state_dim, action_dim, action_min, action_max,
        )

        if initializer:
            self.policy.load_state_dict(initializer.policy.state_dict())

        self.optimizer = init_optimizer(
            cfg.actor_optimizer, self.policy.parameters()
        )
        self.optimizer_name = cfg.actor_optimizer.name

    @property
    def support(self):
        return self.policy.support

    def update(
        self, loss: torch.Tensor, opt_args=tuple(), opt_kwargs=dict(),
    ) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        if self.optimizer_name != "lso":
            self.optimizer.step()
        else:
            self.optimizer.step(*opt_args, **opt_kwargs)

    def get_action(self, state: torch.Tensor, with_grad=False) -> (torch.Tensor, dict):
        if with_grad:
            return self.policy.forward(state)
        else:
            with torch.no_grad():
                return self.policy.forward(state)

    def get_log_prob(self, states: torch.Tensor, actions: torch.Tensor, with_grad=False) -> (torch.Tensor, dict):
        if with_grad:
            return self.policy.log_prob(states, actions)
        else:
            with torch.no_grad():
                return self.policy.log_prob(states, actions)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        net_path = path / "actor_net"
        torch.save(self.policy.state_dict(), net_path)

        opt_path = path / "actor_opt"
        torch.save(self.optimizer.state_dict(), opt_path)

    def load(self, path: Path) -> None:
        net_path = path / 'actor_net'
        self.policy.load_state_dict(torch.load(net_path, map_location=device.device))

        opt_path = path / 'actor_opt'
        self.optimizer.load_state_dict(torch.load(opt_path, map_location=device.device))


class NetworkActorLineSearch(NetworkActor):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int,
                 initializer: Optional['NetworkActor'] = None):
        super().__init__(cfg, state_dim, action_dim, initializer)
        self.optimizer = LineSearchOpt(cfg.actor_optimizer, [self.policy], cfg.actor_optimizer.lr,
                                       cfg.max_backtracking, cfg.error_threshold, cfg.lr_lower_bound,
                                       cfg.actor_optimizer.name)

        action_min, action_max = 0, 1
        self.policy = policy.create(
            cfg.actor_network, state_dim, action_dim, action_min, action_max,
        )
        self.policy_copy = init_actor_network(
            cfg.actor_network, state_dim, action_dim, action_min, action_max
        )

    def set_parameters(self, buffer_address: int, eval_error_fn: Optional['Callable'] = None) -> None:
        self.optimizer.set_params(
            buffer_address, [self.policy_copy.model], eval_error_fn,
        )

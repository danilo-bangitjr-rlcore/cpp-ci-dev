import torch
from pathlib import Path
from omegaconf import DictConfig
from typing import Optional, Callable

from corerl.component.actor.base_actor import BaseActor
from corerl.component.network.factory import init_actor_network
from corerl.component.optimizers.factory import init_optimizer
from corerl.component.optimizers.linesearch_optimizer import LineSearchOpt
from corerl.utils.device import device

class NetworkActor(BaseActor):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int, initializer: Optional['NetworkActor'] = None):
        self.model = init_actor_network(cfg.actor_network, state_dim, action_dim)
        if initializer:
            self.model.load_state_dict(initializer.model.state_dict())
        self.optimizer = init_optimizer(cfg.actor_optimizer, self.model.parameters())
        self.optimizer_name = cfg.actor_optimizer.name

    def distribution_bounds(self):
        return self.model.distribution_bounds()

    def update(
        self, loss: torch.Tensor, opt_args=tuple(), opt_kwargs=dict(),
    ) -> None:
        self.optimizer.zero_grad()
        loss.backward()

        if self.optimizer_name != 'lso':
            self.optimizer.step()
        else:
            self.optimizer.step(*opt_args, **opt_kwargs)

    def get_action(self, state: torch.Tensor, with_grad=False) -> (torch.Tensor, dict):
        if with_grad:
            return self.model.forward(state)
        else:
            with torch.no_grad():
                return self.model.forward(state)

    def get_log_prob(self, states: torch.Tensor, actions: torch.Tensor, with_grad=False) -> (torch.Tensor, dict):
        if with_grad:
            return self.model.log_prob(states, actions)
        else:
            with torch.no_grad():
                return self.model.log_prob(states, actions)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        net_path = path / "actor_net"
        torch.save(self.model.state_dict(), net_path)

        opt_path = path / "actor_opt"
        torch.save(self.optimizer.state_dict(), opt_path)

    def load(self, path: Path) -> None:
        net_path = path / 'actor_net'
        self.model.load_state_dict(torch.load(net_path, map_location=device))

        opt_path = path / 'actor_opt'
        self.optimizer.load_state_dict(torch.load(opt_path, map_location=device))


class NetworkActorLineSearch(NetworkActor):
    def __init__(self, cfg: DictConfig, state_dim: int, action_dim: int,
                 initializer: Optional['NetworkActor'] = None):
        super().__init__(cfg, state_dim, action_dim, initializer)
        self.optimizer = LineSearchOpt(cfg.actor_optimizer, [self.model], cfg.actor_optimizer.lr,
                                       cfg.max_backtracking, cfg.error_threshold, cfg.lr_lower_bound,
                                       cfg.actor_optimizer.name)
        self.model_copy = init_actor_network(cfg.actor_network, state_dim, action_dim)

    def set_parameters(self, buffer_address: int, eval_error_fn: Optional['Callable'] = None) -> None:
        self.optimizer.set_params(buffer_address, [self.model_copy], eval_error_fn)

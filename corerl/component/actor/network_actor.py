import torch
from dataclasses import dataclass, field
from pathlib import Path
from omegaconf import MISSING
from typing import Any, Callable

import corerl.component.policy as policy
import corerl.utils.nullable as nullable
from corerl.component.policy.factory import BaseNNConfig
from corerl.component.actor.base_actor import BaseActor, group
from corerl.component.optimizers.factory import init_optimizer, OptimConfig
from corerl.component.optimizers.linesearch_optimizer import LSOConfig, LineSearchOpt
from corerl.utils.device import device



@dataclass
class NetworkActorConfig:
    name: str = 'network'

    action_min: float = 0
    action_max: float = 1

    actor_network: BaseNNConfig = MISSING
    actor_optimizer: OptimConfig = MISSING
    buffer: Any = MISSING

    defaults: list[Any] = field(default_factory=lambda: [
        {'actor_network': 'beta'},
        {'actor_optimizer': 'adam'},
        {'buffer': 'ensemble_uniform'},
        '_self_',
    ])


class NetworkActor(BaseActor):
    def __init__(
        self,
        cfg: NetworkActorConfig,
        state_dim: int,
        action_dim: int,
        initializer: BaseActor | None = None,
    ):
        # We always assume actions are normalized in (0, 1) unless otherwise
        # stated
        action_min = cfg.action_min
        action_max = cfg.action_max

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
        self,
        loss: torch.Tensor,
        opt_args=tuple(),
        opt_kwargs: dict | None = None,
    ) -> None:
        opt_kwargs = nullable.default(opt_kwargs, dict)

        self.optimizer.zero_grad()
        loss.backward()
        if self.optimizer_name != "lso":
            self.optimizer.step(closure=lambda: 0.)
        else:
            self.optimizer.step(*opt_args, **opt_kwargs)

    def get_action(
        self,
        state: torch.Tensor,
        with_grad=False,
    ) -> tuple[torch.Tensor, dict]:
        if with_grad:
            return self.policy.forward(state)
        else:
            with torch.no_grad():
                return self.policy.forward(state)

    def get_log_prob(self, states: torch.Tensor, actions: torch.Tensor, with_grad=False) -> tuple[torch.Tensor, dict]:
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


group.dispatcher(NetworkActor)


@dataclass
class NetworkActorLineSearchConfig(NetworkActorConfig):
    name: str = 'network_linesearch'

    actor_optimizer: OptimConfig = field(default_factory=LSOConfig)
    error_threshold: float = 1e-4
    lr_lower_bound: float = 1e-6
    max_backtracking: int = 30


class NetworkActorLineSearch(NetworkActor):
    def __init__(
        self,
        cfg: NetworkActorLineSearchConfig,
        state_dim: int,
        action_dim: int,
        initializer: BaseActor | None = None,
    ):
        super().__init__(cfg, state_dim, action_dim, initializer)
        assert isinstance(cfg.actor_optimizer, LSOConfig)
        self.optimizer = LineSearchOpt(
            cfg.actor_optimizer,
            [self.policy.model],
            cfg.actor_optimizer.lr,
            cfg.max_backtracking,
            cfg.error_threshold,
            cfg.lr_lower_bound,
            cfg.actor_optimizer.name,
        )

        action_min, action_max = 0, 1
        self.policy = policy.create(
            cfg.actor_network, state_dim, action_dim, action_min, action_max,
        )
        self.policy_copy = policy.create(
            cfg.actor_network, state_dim, action_dim, action_min, action_max,
        )

    def set_parameters(
        self, buffer_address: int,
        eval_error_fn: Callable[[list[torch.Tensor]], torch.Tensor],
    ) -> None:
        self.optimizer.set_params(
            buffer_address, [self.policy_copy.model], eval_error_fn,
        )

group.dispatcher(NetworkActorLineSearch)

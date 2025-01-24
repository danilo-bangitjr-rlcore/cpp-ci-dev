from dataclasses import field
from pathlib import Path
from typing import Any, Literal, cast

import torch

import corerl.utils.nullable as nullable
from corerl.component.actor.base_actor import BaseActor, group
from corerl.component.buffer.ensemble import EnsembleUniformReplayBufferConfig
from corerl.component.buffer.factory import BufferConfig
from corerl.component.optimizers.factory import OptimizerConfig, init_optimizer
from corerl.component.optimizers.linesearch_optimizer import LineSearchOpt, LSOConfig
from corerl.component.optimizers.torch_opts import AdamConfig
from corerl.component.policy.factory import BaseNNConfig, SquashedGaussianPolicyConfig, create
from corerl.configs.config import MISSING, config
from corerl.utils.device import device


@config(frozen=True)
class _SharedNetworkActorConfig:
    name: Any = MISSING
    action_min: float = 0
    action_max: float = 1

    actor_network: BaseNNConfig = field(default_factory=SquashedGaussianPolicyConfig)
    actor_optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    buffer: BufferConfig = field(default_factory=EnsembleUniformReplayBufferConfig)


@config(frozen=True)
class NetworkActorConfig(_SharedNetworkActorConfig):
    name: Literal['network'] = 'network'


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

        self.policy = create(
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
        opt_args: tuple = tuple(),
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
        with_grad: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        if with_grad:
            return self.policy.forward(state)
        else:
            with torch.no_grad():
                return self.policy.forward(state)

    def get_log_prob(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        with_grad: bool = False,
    ) -> tuple[torch.Tensor, dict]:
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


@config(frozen=True)
class NetworkActorLineSearchConfig(_SharedNetworkActorConfig):
    name: Literal['network_linesearch'] = 'network_linesearch'

    actor_optimizer: LSOConfig = field(default_factory=LSOConfig)
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
        super().__init__(cast(Any, cfg), state_dim, action_dim, initializer)
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
        self.policy = create(
            cfg.actor_network, state_dim, action_dim, action_min, action_max,
        )

group.dispatcher(NetworkActorLineSearch)

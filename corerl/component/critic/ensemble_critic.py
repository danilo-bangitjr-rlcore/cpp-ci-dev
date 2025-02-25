from abc import ABC, abstractmethod
from pathlib import Path

import torch
from pydantic import Field

import corerl.utils.nullable as nullable
from corerl.component.buffer import MixedHistoryBufferConfig
from corerl.component.network.factory import init_target_network
from corerl.component.network.networks import EnsembleNetwork, EnsembleNetworkConfig
from corerl.component.optimizers.factory import OptimizerConfig, init_optimizer
from corerl.component.optimizers.torch_opts import LSOConfig
from corerl.configs.config import config
from corerl.state import AppState
from corerl.utils.device import device


@config()
class CriticConfig:
    critic_network: EnsembleNetworkConfig = Field(default_factory=EnsembleNetworkConfig)
    critic_optimizer: OptimizerConfig = Field(default_factory=LSOConfig)
    buffer: MixedHistoryBufferConfig = Field(default_factory=MixedHistoryBufferConfig)
    polyak: float = 0.995
    target_sync_freq: int = 1


class BaseCritic(ABC):
    @abstractmethod
    def __init__(self, cfg: CriticConfig,  app_state: AppState):
        self.app_state = app_state

    @abstractmethod
    def update(self, loss: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> None:
        raise NotImplementedError

class EnsembleCritic(BaseCritic):
    def __init__(
        self,
        cfg: CriticConfig,
        app_state: AppState,
        state_dim: int,
        action_dim: int,
        output_dim: int = 1
    ):
        input_dim = state_dim + action_dim
        self.model = EnsembleNetwork(
            cfg.critic_network,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        target = EnsembleNetwork(
            cfg.critic_network,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        self.target = init_target_network(target, self.model)

        params = self.model.parameters(independent=True) # type: ignore
        self.optimizer = init_optimizer(
            cfg.critic_optimizer,
            app_state,
            list(params),
            ensemble=True,
        )
        self.polyak = cfg.polyak
        self.target_sync_freq = cfg.target_sync_freq
        self.target_sync_counter = 0
        self.optimizer_name = cfg.critic_optimizer.name

    def _prepare_input(self, batches: list[torch.Tensor]) -> torch.Tensor:
        ensemble = len(batches)
        return torch.stack(batches) if ensemble > 1 else batches[0]

    def _forward(
        self,
        input_tensor: torch.Tensor,
        with_grad: bool = False,
        bootstrap_reduct: bool = True,
    ):
        if with_grad:
            return self.model.forward(input_tensor, bootstrap_reduct)
        else:
            with torch.no_grad():
                return self.model.forward(input_tensor, bootstrap_reduct)

    def _forward_target(
        self,
        input_tensor: torch.Tensor,
        bootstrap_reduct: bool = True,
    ):
        with torch.no_grad():
            return self.target.forward(input_tensor, bootstrap_reduct)

    def ensemble_backward(self, loss: list[torch.Tensor]) -> None:
        for i, loss_item in enumerate(loss):
            params = self.model.parameters(independent=True)[i] # type: ignore
            loss_item.backward(
                inputs=list(params)
            )

    def _update_target(self) -> None:
        self.target_sync_counter += 1
        if self.target_sync_counter % self.target_sync_freq == 0:
            self.sync_target()
            self.target_sync_counter = 0

    def update(
        self,
        loss: torch.Tensor,
        opt_args: tuple = tuple(),
        opt_kwargs: dict | None = None,
    ) -> None:
        opt_kwargs = nullable.default(opt_kwargs, dict)
        self.optimizer.zero_grad()
        loss.backward()

        if self.optimizer_name != "armijo_adam" and self.optimizer_name != "lso":
            self.optimizer.step(closure=lambda: 0.)
        else:
            self.optimizer.step(*opt_args, **opt_kwargs)
        self._update_target()

    def sync_target(self) -> None:
        with torch.no_grad():
            for p, p_targ in zip(
                self.model.parameters(),
                self.target.parameters(),
                strict=True,
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "critic_net")
        torch.save(self.target.state_dict(), path / "critic_target")

    def load(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path / "critic_net", map_location=device.device))
        self.target.load_state_dict(torch.load(path / "critic_target", map_location=device.device))

    def get_qs(
        self,
        state_batches: list[torch.Tensor],
        action_batches: list[torch.Tensor],
        with_grad: bool = False,
        bootstrap_reduct: bool = True,
    ):
        state_action_batches = [
            torch.concat((s, a), dim=1)
            for s, a in zip(state_batches, action_batches, strict=False)
        ]
        input_tensor = self._prepare_input(state_action_batches)
        return self._forward(input_tensor, with_grad=with_grad, bootstrap_reduct=bootstrap_reduct)

    def get_q(
        self,
        state_batches: list[torch.Tensor],
        action_batches: list[torch.Tensor],
        with_grad: bool = False,
        bootstrap_reduct: bool = True,
    ) -> torch.Tensor:
        out = self.get_qs(state_batches, action_batches, with_grad=with_grad, bootstrap_reduct=bootstrap_reduct)
        return out.reduced_value

    def get_qs_target(
        self, state_batches: list[torch.Tensor], action_batches: list[torch.Tensor], bootstrap_reduct: bool = True,
    ):
        state_action_batches = [
            torch.concat((s, a), dim=1)
            for s, a in zip(state_batches, action_batches, strict=False)
        ]
        input_tensor = self._prepare_input(state_action_batches)
        return self._forward_target(input_tensor, bootstrap_reduct=bootstrap_reduct)

    def get_q_target(
        self, state_batches: list[torch.Tensor], action_batches: list[torch.Tensor], bootstrap_reduct: bool = True
    ) -> torch.Tensor:
        out = self.get_qs_target(state_batches, action_batches, bootstrap_reduct=bootstrap_reduct)
        return out.reduced_value

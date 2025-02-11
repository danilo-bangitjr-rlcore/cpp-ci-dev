from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import Field

import corerl.utils.nullable as nullable
from corerl.component.buffer.ensemble import EnsembleUniformReplayBufferConfig
from corerl.component.buffer.factory import BufferConfig
from corerl.component.critic.base_critic import BaseCriticConfig, BaseQ, BaseV
from corerl.component.network.factory import NetworkConfig, init_critic_network, init_critic_target
from corerl.component.network.networks import EnsembleCriticNetworkConfig
from corerl.component.optimizers.factory import OptimizerConfig, init_optimizer
from corerl.component.optimizers.torch_opts import LSOConfig
from corerl.configs.config import MISSING, config
from corerl.state import AppState
from corerl.utils.device import device


@config()
class _SharedEnsembleConfig:
    name: Any = MISSING
    critic_network: NetworkConfig = Field(default_factory=EnsembleCriticNetworkConfig)
    critic_optimizer: OptimizerConfig = Field(default_factory=LSOConfig)
    buffer: BufferConfig = Field(default_factory=EnsembleUniformReplayBufferConfig)
    polyak: float = 0.99
    target_sync_freq: int = 1

@config()
class EnsembleCriticConfig(BaseCriticConfig, _SharedEnsembleConfig):
    name: Literal['ensemble'] = 'ensemble'

class BaseEnsembleCritic:
    def __init__(
        self,
        cfg: EnsembleCriticConfig,
        app_state: AppState,
        state_dim: int,
        action_dim: int,
        output_dim: int = 1
    ):
        input_dim = state_dim + action_dim
        self.model = init_critic_network(
            cfg.critic_network, input_dim=input_dim, output_dim=output_dim,
        )
        self.target = init_critic_target(
            cfg.critic_network, input_dim=input_dim, output_dim=output_dim,
            critic=self.model,
        )
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if with_grad:
            return self.model(input_tensor, bootstrap_reduct)
        else:
            with torch.no_grad():
                return self.model(input_tensor, bootstrap_reduct)

    def _forward_target(
        self,
        input_tensor: torch.Tensor,
        bootstrap_reduct: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.target(input_tensor, bootstrap_reduct)

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
        torch.save(self.optimizer.state_dict(), path / "critic_opt")

    def load(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path / "critic_net", map_location=device.device))
        self.target.load_state_dict(torch.load(path / "critic_target", map_location=device.device))
        self.optimizer.load_state_dict(torch.load(path / "critic_opt", map_location=device.device))

class EnsembleQCritic(BaseQ, BaseEnsembleCritic):
    def __init__(
            self,
            cfg: EnsembleCriticConfig,
            app_state: AppState,
            state_dim: int,
            action_dim: int,
            output_dim: int = 1
        ):
        BaseEnsembleCritic.__init__(self, cfg, app_state, state_dim, action_dim, output_dim)

    def update(
        self,
        loss: torch.Tensor,
        opt_args: tuple = tuple(),
        opt_kwargs: dict | None = None,
    ) -> None:
        BaseEnsembleCritic.update(self, loss, opt_args, opt_kwargs)

    def save(self, path: Path) -> None:
        BaseEnsembleCritic.save(self, path)

    def load(self, path: Path) -> None:
        BaseEnsembleCritic.load(self, path)

    def get_qs(
        self,
        state_batches: list[torch.Tensor],
        action_batches: list[torch.Tensor],
        with_grad: bool = False,
        bootstrap_reduct: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        q, _ = self.get_qs(state_batches, action_batches, with_grad=with_grad, bootstrap_reduct=bootstrap_reduct)
        return q

    def get_qs_target(
        self, state_batches: list[torch.Tensor], action_batches: list[torch.Tensor], bootstrap_reduct: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_action_batches = [
            torch.concat((s, a), dim=1)
            for s, a in zip(state_batches, action_batches, strict=False)
        ]
        input_tensor = self._prepare_input(state_action_batches)
        return self._forward_target(input_tensor, bootstrap_reduct=bootstrap_reduct)

    def get_q_target(
        self, state_batches: list[torch.Tensor], action_batches: list[torch.Tensor], bootstrap_reduct: bool = True
    ) -> torch.Tensor:
        q, _ = self.get_qs_target(state_batches, action_batches, bootstrap_reduct=bootstrap_reduct)
        return q


class EnsembleVCritic(BaseV, BaseEnsembleCritic):
    def __init__(self, cfg: EnsembleCriticConfig, app_state: AppState, state_dim: int, output_dim: int = 1):
        BaseEnsembleCritic.__init__(self, cfg, app_state, state_dim, output_dim)

    def update(
        self,
        loss: torch.Tensor,
        opt_args: tuple = tuple(),
        opt_kwargs: dict | None = None,
    ) -> None:
        BaseEnsembleCritic.update(self, loss, opt_args, opt_kwargs)

    def save(self, path: Path) -> None:
        BaseEnsembleCritic.save(self, path)

    def load(self, path: Path) -> None:
        BaseEnsembleCritic.load(self, path)

    def get_vs(
        self, state_batches: list[torch.Tensor], with_grad: bool = False, bootstrap_reduct: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_tensor = self._prepare_input(state_batches)
        return self._forward(input_tensor, with_grad=with_grad, bootstrap_reduct=bootstrap_reduct)

    def get_v(
        self,
        state_batches: list[torch.Tensor],
        with_grad: bool = False,
        bootstrap_reduct: bool = True,
    ) -> torch.Tensor:
        v, _ = self.get_vs(state_batches, with_grad=with_grad, bootstrap_reduct=bootstrap_reduct)
        return v

    def get_vs_target(
        self, state_batches: list[torch.Tensor], bootstrap_reduct: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_tensor = self._prepare_input(state_batches)
        return self._forward_target(input_tensor, bootstrap_reduct)

    def get_v_target(
        self,
        state_batches: list[torch.Tensor],
        bootstrap_reduct: bool = True,
    ) -> torch.Tensor:
        v, _ = self.get_vs_target(state_batches, bootstrap_reduct)
        return v

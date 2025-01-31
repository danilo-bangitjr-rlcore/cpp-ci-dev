from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import Field

import corerl.utils.nullable as nullable
from corerl.component.buffer.ensemble import EnsembleUniformReplayBufferConfig
from corerl.component.buffer.factory import BufferConfig
from corerl.component.critic.base_critic import BaseQ, BaseQConfig, BaseV
from corerl.component.network.factory import NetworkConfig, init_critic_network, init_critic_target
from corerl.component.network.networks import EnsembleCriticNetworkConfig
from corerl.component.optimizers.factory import OptimizerConfig, init_optimizer
from corerl.component.optimizers.torch_opts import AdamConfig
from corerl.configs.config import MISSING, config
from corerl.utils.device import device


@config()
class _SharedEnsembleConfig:
    name: Any = MISSING
    critic_network: NetworkConfig = Field(default_factory=EnsembleCriticNetworkConfig)
    critic_optimizer: OptimizerConfig = Field(default_factory=AdamConfig)
    buffer: BufferConfig = Field(default_factory=EnsembleUniformReplayBufferConfig)
    polyak: float = 0.99
    target_sync_freq: int = 1


@config()
class EnsembleCriticConfig(BaseQConfig, _SharedEnsembleConfig):
    name: Literal['ensemble'] = 'ensemble'


class EnsembleQCritic(BaseQ):
    def __init__(self, cfg: EnsembleCriticConfig, state_dim: int, action_dim: int, output_dim: int = 1):
        state_action_dim = state_dim + action_dim
        self.model = init_critic_network(
            cfg.critic_network, input_dim=state_action_dim, output_dim=output_dim,
        )
        self.target = init_critic_target(
            cfg.critic_network, input_dim=state_action_dim, output_dim=output_dim,
            critic=self.model,
        )

        params = self.model.parameters(independent=True) # type: ignore
        self.optimizer = init_optimizer(
            cfg.critic_optimizer,
            list(params),
            ensemble=True,
        )

        self.optimizer_name = cfg.critic_optimizer.name
        self.action_dim = action_dim

        self.polyak = cfg.polyak
        self.target_sync_freq = cfg.target_sync_freq
        self.target_sync_counter = 0

    def get_qs(
        self,
        state_batches: list[torch.Tensor],
        action_batches: list[torch.Tensor],
        with_grad: bool = False,
        bootstrap_reduct: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ensemble = len(state_batches)
        state_action_batches = [torch.concat((state_batches[i], action_batches[i]), dim=1) for i in range(ensemble)]
        if ensemble > 1:
            input_tensor = torch.stack(state_action_batches)
        else:
            input_tensor = state_action_batches[0]

        if with_grad:
            q, qs = self.model(input_tensor, bootstrap_reduct)
        else:
            with torch.no_grad():
                q, qs = self.model(input_tensor, bootstrap_reduct)
        return q, qs

    def get_q(
        self,
        state_batches: list[torch.Tensor],
        action_batches: list[torch.Tensor],
        with_grad: bool = False,
        bootstrap_reduct: bool = True,
    ) -> torch.Tensor:
        q, qs = self.get_qs(state_batches, action_batches, with_grad=with_grad, bootstrap_reduct=bootstrap_reduct)
        return q

    def get_qs_target(
        self, state_batches: list[torch.Tensor], action_batches: list[torch.Tensor], bootstrap_reduct: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ensemble = len(state_batches)
        state_action_batches = [torch.concat((state_batches[i], action_batches[i]), dim=1) for i in range(ensemble)]
        if ensemble > 1:
            input_tensor = torch.stack(state_action_batches)
        else:
            input_tensor = state_action_batches[0]

        with torch.no_grad():
            return self.target(input_tensor, bootstrap_reduct)

    def get_q_target(
        self, state_batches: list[torch.Tensor], action_batches: list[torch.Tensor], bootstrap_reduct: bool = True,
    ) -> torch.Tensor:
        q, qs = self.get_qs_target(state_batches, action_batches, bootstrap_reduct=bootstrap_reduct)
        return q

    def update(
        self,
        loss: list[torch.Tensor] | torch.Tensor,
        opt_args: tuple = tuple(),
        opt_kwargs: dict | None = None,
    ) -> None:
        opt_kwargs = nullable.default(opt_kwargs, dict)

        self.optimizer.zero_grad()
        if isinstance(loss, (list, tuple)):
            self.ensemble_backward(loss)
        else:
            loss.backward()

        if self.optimizer_name != "armijo_adam":
            self.optimizer.step(closure=lambda: 0.)
        else:
            self.optimizer.step(*opt_args, **opt_kwargs)

        self.target_sync_counter += 1
        if self.target_sync_counter % self.target_sync_freq == 0:
            self.sync_target()
            self.target_sync_counter = 0


    def ensemble_backward(self, loss: list[torch.Tensor]):
        for i in range(len(loss)):
            params = self.model.parameters(independent=True)[i] # type: ignore
            loss[i].backward(
                inputs=list(params)
            )
        return

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

        net_path = path / "critic_net"
        torch.save(self.model.state_dict(), net_path)

        target_path = path / "critic_target"
        torch.save(self.target.state_dict(), target_path)

        opt_path = path / "critic_opt"
        torch.save(self.optimizer.state_dict(), opt_path)

    def load(self, path: Path) -> None:
        net_path = path / 'critic_net'
        self.model.load_state_dict(torch.load(net_path, map_location=device.device))

        target_path = path / 'critic_target'
        self.target.load_state_dict(
            torch.load(target_path, map_location=device.device),
        )

        opt_path = path / 'critic_opt'
        self.optimizer.load_state_dict(
            torch.load(opt_path, map_location=device.device),
        )


class EnsembleVCritic(BaseV):
    def __init__(self, cfg: EnsembleCriticConfig, state_dim: int, output_dim: int = 1):
        self.model = init_critic_network(
            cfg.critic_network, input_dim=state_dim, output_dim=output_dim,
        )
        self.target = init_critic_target(
            cfg.critic_network, input_dim=state_dim, output_dim=output_dim,
            critic=self.model,
        )

        params = self.model.parameters(independent=True) # type: ignore
        self.optimizer = init_optimizer(
            cfg.critic_optimizer,
            list(params),
            ensemble=True,
        )
        self.optimizer_name = cfg.critic_optimizer.name
        self.polyak = cfg.polyak
        self.target_sync_freq = cfg.target_sync_freq
        self.target_sync_counter = 0

    def get_vs(
        self, state_batches: list[torch.Tensor], with_grad: bool = False, bootstrap_reduct: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ensemble = len(state_batches)
        if ensemble > 1:
            input_tensor = torch.stack(state_batches)
        else:
            input_tensor = state_batches[0]

        if with_grad:
            v, vs = self.model(input_tensor, bootstrap_reduct)
        else:
            with torch.no_grad():
                v, vs = self.model(input_tensor, bootstrap_reduct)
        return v, vs

    def ensemble_backward(self, loss: list[torch.Tensor]):
        for i in range(len(loss)):
            params = self.model.parameters(independent=True)[i] # type: ignore

            loss[i].backward(
                inputs=list(params),
            )
        return

    def get_v(
        self,
        state_batches: list[torch.Tensor],
        with_grad: bool = False,
        bootstrap_reduct: bool = True,
    ) -> torch.Tensor:
        v, vs = self.get_vs(state_batches, with_grad=with_grad, bootstrap_reduct=bootstrap_reduct)
        return v

    def get_vs_target(
        self, state_batches: list[torch.Tensor], bootstrap_reduct: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ensemble = len(state_batches)
        if ensemble > 1:
            input_tensor = torch.stack(state_batches)
        else:
            input_tensor = state_batches[0]
        with torch.no_grad():
            return self.target(input_tensor, bootstrap_reduct)

    def get_v_target(
        self,
        state_batches: list[torch.Tensor],
        bootstrap_reduct: bool = True,
    ) -> torch.Tensor:
        v, vs = self.get_vs_target(state_batches, bootstrap_reduct)
        return v

    def update(
        self,
        loss: list[torch.Tensor] | torch.Tensor,
        opt_args: tuple = tuple(),
        opt_kwargs: dict | None = None,
    ) -> None:
        opt_kwargs = nullable.default(opt_kwargs, dict)
        self.optimizer.zero_grad()
        if isinstance(loss, (list, tuple)):
            self.ensemble_backward(loss)
        else:
            loss.backward()

        if self.optimizer_name != "armijo_adam":
            self.optimizer.step(closure=lambda: 0.)
        else:
            self.optimizer.step(*opt_args, **opt_kwargs)

        self.target_sync_counter += 1
        if self.target_sync_counter % self.target_sync_freq == 0:
            self.sync_target()
            self.target_sync_counter = 0

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

        net_path = path / "critic_net"
        torch.save(self.model.state_dict(), net_path)

        target_path = path / "critic_target"
        torch.save(self.target.state_dict(), target_path)

        opt_path = path / "critic_opt"
        torch.save(self.optimizer.state_dict(), opt_path)

    def load(self, path: Path) -> None:
        net_path = path / 'critic_net'
        self.model.load_state_dict(torch.load(net_path, map_location=device.device))

        target_path = path / 'critic_target'
        self.target.load_state_dict(
            torch.load(target_path, map_location=device.device),
        )

        opt_path = path / 'critic_opt'
        self.optimizer.load_state_dict(
            torch.load(opt_path, map_location=device.device),
        )


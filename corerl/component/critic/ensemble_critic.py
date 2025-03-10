from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import torch
from pydantic import Field

from corerl.component.buffer import MixedHistoryBufferConfig
from corerl.component.network.factory import init_target_network
from corerl.component.network.networks import EnsembleNetwork, EnsembleNetworkConfig
from corerl.component.network.utils import to_np
from corerl.component.optimizers.factory import OptimizerConfig, init_optimizer
from corerl.component.optimizers.torch_opts import LSOConfig
from corerl.configs.config import config
from corerl.eval.torch import get_layers_stable_rank
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
    def update(self, loss: torch.Tensor, closure: Callable[[], float]) -> None:
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
        self._app_state = app_state
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
    ):
        if with_grad:
            return self.model.forward(input_tensor)
        else:
            with torch.no_grad():
                return self.model.forward(input_tensor)

    def _forward_target(
        self,
        input_tensor: torch.Tensor,
    ):
        with torch.no_grad():
            return self.target.forward(input_tensor)

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
        closure: Callable[[], float],
    ) -> None:
        self.optimizer.zero_grad()
        loss.backward()

        # metrics
        log_critic_gradient_norm(self._app_state, self.model)
        log_critic_weight_norm(self._app_state, self.model)
        log_critic_stable_rank(self._app_state, self.model)

        if self.optimizer_name != "armijo_adam" and self.optimizer_name != "lso":
            self.optimizer.step(closure=lambda: 0.)
        else:
            self.optimizer.step(closure=closure)
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

    def get_values(
        self,
        state_batches: list[torch.Tensor],
        action_batches: list[torch.Tensor],
        with_grad: bool = False,
    ):
        state_action_batches = [
            torch.concat((s, a), dim=1)
            for s, a in zip(state_batches, action_batches, strict=False)
        ]
        input_tensor = self._prepare_input(state_action_batches)
        return self._forward(input_tensor, with_grad=with_grad)

    def get_target_values(
        self, state_batches: list[torch.Tensor], action_batches: list[torch.Tensor],
    ):
        state_action_batches = [
            torch.concat((s, a), dim=1)
            for s, a in zip(state_batches, action_batches, strict=False)
        ]
        input_tensor = self._prepare_input(state_action_batches)
        return self._forward_target(input_tensor)

# ---------------------------------------------------------------------------- #
#                                    Metrics                                   #
# ---------------------------------------------------------------------------- #

def log_critic_gradient_norm(app_state: AppState, critic: EnsembleNetwork):
    """
    Logs the gradient norm for each member of the ensemble.
    """
    for ensemble_i, param_i in enumerate(critic.parameters(independent = True)):
        total_norm_i = 0
        for param in param_i:
            if param.requires_grad and param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm_i += param_norm.item() ** 2

        total_norm_i = total_norm_i ** 0.5

        app_state.metrics.write(
                agent_step=app_state.agent_step,
                metric=f"optimizer_critic_{ensemble_i}_grad_norm",
                value=to_np(total_norm_i),
        )

def log_critic_weight_norm(app_state: AppState, critic: EnsembleNetwork):
    """
    Logs the weight norm for each member of the ensemble.
    """
    with torch.no_grad():
        for ensemble_i, param_i in enumerate(critic.parameters(independent = True)):
            total_norm_i = 0
            for param in param_i:
                param_norm = param.norm(2)
                total_norm_i += param_norm.item() ** 2

            total_norm_i = total_norm_i ** 0.5

            app_state.metrics.write(
                    agent_step=app_state.agent_step,
                    metric=f"network_critic_{ensemble_i}_weight_norm",
                    value=to_np(total_norm_i),
            )

def log_critic_stable_rank(app_state: AppState, critic: EnsembleNetwork):
    with torch.no_grad():
        for ensemble_i, ensemble_member in enumerate(critic.subnetworks):
            stable_ranks = get_layers_stable_rank(ensemble_member)

            for i, rank in enumerate(stable_ranks):
                app_state.metrics.write(
                        agent_step=app_state.agent_step,
                        metric=f"network_critic_{ensemble_i}_stable_rank_layer_{i}",
                        value=rank,
                )

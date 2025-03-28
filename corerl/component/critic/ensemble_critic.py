import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import torch
from pydantic import Field

from corerl.component.buffer import MixedHistoryBufferConfig
from corerl.component.network.factory import init_target_network
from corerl.component.network.networks import EnsembleNetwork, EnsembleNetworkConfig, EnsembleNetworkReturn
from corerl.component.optimizers.factory import OptimizerConfig, init_optimizer
from corerl.component.optimizers.torch_opts import LSOConfig
from corerl.configs.config import config
from corerl.state import AppState

logger = logging.getLogger(__name__)

@config()
class CriticConfig:
    """
    Kind: internal

    Critic-specific hyperparameters.
    """
    critic_network: EnsembleNetworkConfig = Field(default_factory=EnsembleNetworkConfig)
    critic_optimizer: OptimizerConfig = Field(default_factory=LSOConfig)
    buffer: MixedHistoryBufferConfig = Field(default_factory=MixedHistoryBufferConfig)
    polyak: float = 0.995
    """
    Retention coefficient for polyak averaged target networks.
    """


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
        self._cfg = cfg
        self._app_state = app_state

        self._state_dim = state_dim
        self._action_dim = action_dim

        self.model = EnsembleNetwork(
            cfg.critic_network,
            input_dims=[state_dim, action_dim],
            output_dim=output_dim,
        )
        target = EnsembleNetwork(
            cfg.critic_network,
             input_dims=[state_dim, action_dim],
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
        self.target_sync_counter = 0
        self.optimizer_name = cfg.critic_optimizer.name


    def _update_target(self) -> None:
        self.target_sync_counter += 1
        self.polyak_avg_target()

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

    def polyak_avg_target(self) -> None:
        if self.polyak == 0:
            return

        with torch.no_grad():
            for p, p_targ in zip(
                self.model.parameters(),
                self.target.parameters(),
                strict=True,
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def save(self, path: Path) -> None:
        self.model.save(path / "critic_net")
        self.target.save(path / "critic_target")

    def load(self, path: Path) -> None:
        try:
            self.model.load(path / "critic_net")
            self.target.load(path / "critic_target")
        except Exception:
            logger.exception('Failed to load critic state from checkpoint. Reinitializing...')
            self.model = EnsembleNetwork(
                self._cfg.critic_network,
                input_dims=[self._state_dim, self._action_dim],
                output_dim=1,
            )
            init_target_network(self.target, self.model)

    def get_values(
        self,
        state_batches: list[torch.Tensor],
        action_batches: list[torch.Tensor],
        with_grad: bool = False,
    ) -> EnsembleNetworkReturn:

        with torch.set_grad_enabled(with_grad):
            return self.model.forward([state_batches, action_batches])

    def get_target_values(
        self,
        state_batches: list[torch.Tensor],
        action_batches: list[torch.Tensor],
    )-> EnsembleNetworkReturn:
        return self.target.forward([state_batches, action_batches])


# ---------------------------------------------------------------------------- #
#                                    Metrics                                   #
# ---------------------------------------------------------------------------- #

def log_critic_gradient_norm(app_state: AppState, critic: EnsembleNetwork):
    """
    Logs the gradient norm for each member of the ensemble.
    """
    for i, norm in enumerate(critic.get_gradient_norms()):
        app_state.metrics.write(
            agent_step=app_state.agent_step,
            metric=f"optimizer_critic_{i}_grad_norm",
            value=norm,
        )

def log_critic_weight_norm(app_state: AppState, critic: EnsembleNetwork):
    """
    Logs the weight norm for each member of the ensemble.
    """
    with torch.no_grad():
        for i, norm in enumerate(critic.get_weight_norms()):
            app_state.metrics.write(
                agent_step=app_state.agent_step,
                metric=f"network_critic_{i}_weight_norm",
                value=norm,
            )

def log_critic_stable_rank(app_state: AppState, critic: EnsembleNetwork):
    with torch.no_grad():
        for i, stable_ranks in enumerate(critic.get_stable_ranks()):
            for j, rank in enumerate(stable_ranks):
                app_state.metrics.write(
                    agent_step=app_state.agent_step,
                    metric=f"network_critic_{i}_stable_rank_layer_{j}",
                    value=rank,
                )

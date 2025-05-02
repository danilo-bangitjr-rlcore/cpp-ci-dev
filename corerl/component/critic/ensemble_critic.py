import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import torch
from pydantic import Field

from corerl.component.buffer import BufferConfig, MixedHistoryBufferConfig
from corerl.component.network.factory import init_target_network
from corerl.component.network.networks import EnsembleNetwork, EnsembleNetworkConfig, EnsembleNetworkReturn
from corerl.component.optimizers.factory import OptimizerConfig, init_optimizer
from corerl.component.optimizers.torch_opts import LSOConfig
from corerl.configs.config import config
from corerl.data_pipeline.datatypes import TransitionBatch
from corerl.messages.events import EventType
from corerl.state import AppState
from corerl.utils.device import device

logger = logging.getLogger(__name__)

@config()
class CriticConfig:
    """
    Kind: internal

    Critic-specific hyperparameters.
    """
    critic_network: EnsembleNetworkConfig = Field(default_factory=EnsembleNetworkConfig)
    critic_optimizer: OptimizerConfig = Field(default_factory=LSOConfig)
    buffer: BufferConfig = Field(
        default_factory=MixedHistoryBufferConfig,
        discriminator='name',
    )
    polyak: float = 0.995
    """
    Retention coefficient for polyak averaged target networks.
    """


class BaseCritic(ABC):
    @abstractmethod
    def __init__(self, cfg: CriticConfig,  app_state: AppState):
        self.app_state = app_state

    @abstractmethod
    def update(
        self,
        batches: list[TransitionBatch],
        next_actions: list[torch.Tensor],
        eval_batches: list[TransitionBatch],
        eval_actions: list[torch.Tensor],
    ) -> float:
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
        self.optimizer_name = cfg.critic_optimizer.name

    # -------------
    # -- Updates --
    # -------------
    def _update(
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

        if self.optimizer_name not in {'armijo_adam', 'lso'}:
            self.optimizer.step(closure=lambda: 0.)
        else:
            self.optimizer.step(closure=closure)

        self.polyak_avg_target()


    def update(
        self,
        batches: list[TransitionBatch],
        next_actions: list[torch.Tensor],
        eval_batches: list[TransitionBatch],
        eval_actions: list[torch.Tensor],
    ):
        self._app_state.event_bus.emit_event(EventType.agent_update_critic)
        q_loss = self.compute_loss(
            batches,
            next_actions,
            log_metrics=True,
        )

        self._update(q_loss, closure=lambda: self.compute_loss(eval_batches, eval_actions).item())
        return q_loss.item()


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

    # ------------
    # -- Losses --
    # ------------
    def compute_loss(
        self,
        ensemble_batch: list[TransitionBatch],
        next_actions: list[torch.Tensor],
        log_metrics: bool=False,
    ) -> torch.Tensor:
        # First, translate ensemble batches in to list for each property
        ensemble_len = len(ensemble_batch)
        state_batches = []
        action_batches = []
        reward_batches = []
        next_state_batches = []
        next_action_batches = []
        gamma_batches = []
        for i, batch in enumerate(ensemble_batch):
            state_batch = batch.prior.state
            direct_action_batch = batch.post.action
            reward_batch = batch.n_step_reward
            next_state_batch = batch.post.state
            gamma_batch = batch.n_step_gamma
            dp_mask = batch.post.dp

            # For the 'Anytime' paradigm, only states at decision points can sample next_actions
            # If a state isn't at a decision point, its next_action is set to the current action
            with torch.no_grad():
                next_action_batches.append((dp_mask * next_actions[i]) + ((1.0 - dp_mask) * direct_action_batch))

            state_batches.append(state_batch)
            action_batches.append(direct_action_batch)
            reward_batches.append(reward_batch)
            next_state_batches.append(next_state_batch)
            gamma_batches.append(gamma_batch)

        loss = self._sarsa_loss(
            state_batches,
            action_batches,
            next_state_batches,
            next_action_batches,
            reward_batches,
            gamma_batches,
        )

        if log_metrics:
            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric="avg_critic_loss",
                value=loss.item() / ensemble_len,
            )

        return loss

    def _sarsa_loss(
        self,
        states: list[torch.Tensor],
        actions: list[torch.Tensor],
        next_states: list[torch.Tensor],
        next_actions: list[torch.Tensor],
        rewards: list[torch.Tensor],
        gammas: list[torch.Tensor],
    ):
        values = self.get_values(states, actions, with_grad=True)
        target_values = self.get_target_values(next_states, next_actions)

        loss = torch.tensor(0.0, device=device.device)
        for i, reward in enumerate(rewards):
            q = values.ensemble_values[i]
            qp = target_values.ensemble_values[i]
            target = reward + gammas[i] * qp
            loss_i = torch.nn.functional.mse_loss(target, q)
            loss += loss_i

            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric=f"critic_loss_{i}",
                value=loss_i.item(),
            )

        if values.ensemble_variance is not None:
            mean_variance = torch.mean(values.ensemble_variance)
            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric="critic_ensemble_variance",
                value=mean_variance.item(),
            )

        return loss

    # ---------------
    # -- Lifecycle --
    # ---------------
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
        with torch.no_grad():
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

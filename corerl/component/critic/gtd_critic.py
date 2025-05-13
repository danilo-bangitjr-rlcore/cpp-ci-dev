import logging
from collections.abc import Callable
from pathlib import Path

import torch

from corerl.component.critic.ensemble_critic import (
    BaseCritic,
    CriticConfig,
    log_critic_gradient_norm,
    log_critic_stable_rank,
    log_critic_weight_norm,
)
from corerl.component.network.networks import EnsembleNetwork, EnsembleNetworkReturn
from corerl.component.optimizers.factory import init_optimizer
from corerl.configs.config import config
from corerl.data_pipeline.datatypes import TransitionBatch
from corerl.messages.events import EventType
from corerl.state import AppState
from corerl.utils.device import device

logger = logging.getLogger(__name__)

@config()
class GTDCriticConfig(CriticConfig):
    beta: float = 1.0


class GTDCritic(BaseCritic):
    def __init__(
        self,
        cfg: GTDCriticConfig,
        app_state: AppState,
        state_dim: int,
        action_dim: int,
        output_dim: int = 1
    ):
        self._cfg = cfg
        self._app_state = app_state

        self._state_dim = state_dim
        self._action_dim = action_dim
        self._output_dim = output_dim

        self.model = EnsembleNetwork(
            cfg.critic_network,
            input_dims=[state_dim, action_dim],
            output_dim=2 * output_dim,
        )

        params = self.model.parameters(independent=True) # type: ignore
        self.optimizer = init_optimizer(
            cfg.critic_optimizer,
            app_state,
            list(params),
            ensemble=True,
        )
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

            next_action_batches.append(next_actions[i])
            state_batches.append(state_batch)
            action_batches.append(direct_action_batch)
            reward_batches.append(reward_batch)
            next_state_batches.append(next_state_batch)
            gamma_batches.append(gamma_batch)

        loss = self._qrc_loss(
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

    def _qrc_loss(
        self,
        states: list[torch.Tensor],
        actions: list[torch.Tensor],
        next_states: list[torch.Tensor],
        next_actions: list[torch.Tensor],
        rewards: list[torch.Tensor],
        gammas: list[torch.Tensor],
    ):
        values = self.get_hv_values(states, actions, with_grad=True)
        target_values = self.get_values_sampled_actions(next_states, next_actions)

        loss = torch.tensor(0.0, device=device.device)
        for i, reward in enumerate(rewards):
            out = values.ensemble_values[i]
            q = out[:, 0].unsqueeze(-1)
            h = out[:, 1].unsqueeze(-1)

            qp = target_values.ensemble_values[i].mean(dim=1)[:, 0].unsqueeze(-1)
            target = reward + gammas[i] * qp

            delta_l = target.detach() - q
            delta_r = target - q.detach()

            q_loss_i = (0.5 * delta_l**2 + torch.tanh(h).detach() * delta_r).mean()
            h_loss_i = torch.nn.functional.mse_loss(delta_l.detach(), h)
            h_reg_i = self._cfg.beta * torch.mean(h**2)
            loss_i = q_loss_i + h_loss_i + h_reg_i
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

    def load(self, path: Path) -> None:
        try:
            self.model.load(path / "critic_net")
        except Exception:
            logger.exception('Failed to load critic state from checkpoint. Reinitializing...')
            self.model = EnsembleNetwork(
                self._cfg.critic_network,
                input_dims=[self._state_dim, self._action_dim],
                output_dim=2,
            )

    def get_values(
        self,
        state_batches: list[torch.Tensor],
        action_batches: list[torch.Tensor],
        with_grad: bool = False,
    ) -> EnsembleNetworkReturn:
        with torch.set_grad_enabled(with_grad):
            out = self.model.forward([state_batches, action_batches])

        return EnsembleNetworkReturn(
            reduced_value=out.reduced_value[:, 0],
            ensemble_values=out.ensemble_values[:, :, 0],
            ensemble_variance=out.ensemble_variance,
        )

    def get_hv_values(
        self,
        state_batches: list[torch.Tensor],
        action_batches: list[torch.Tensor],
        with_grad: bool = False,
    ):
        with torch.set_grad_enabled(with_grad):
            return self.model.forward([state_batches, action_batches])

    def get_values_sampled_actions(
        self,
        state_batches: list[torch.Tensor],
        action_batches: list[torch.Tensor],
    ):
        assert action_batches[0].dim() == 3
        ensemble = len(action_batches)
        batch_size = action_batches[0].size(0)
        n_samples = action_batches[0].size(1)
        states = [
            state_batch.repeat_interleave(n_samples, dim=0)
            for state_batch in state_batches
        ]

        actions = [
            action_batch.reshape(batch_size * n_samples, -1)
            for action_batch in action_batches
        ]

        out = self.model.forward([states, actions])
        return EnsembleNetworkReturn(
            reduced_value=out.reduced_value.reshape(batch_size, n_samples, 2),
            ensemble_values=out.ensemble_values.reshape(ensemble, batch_size, n_samples, 2),
            ensemble_variance=out.ensemble_variance,
        )

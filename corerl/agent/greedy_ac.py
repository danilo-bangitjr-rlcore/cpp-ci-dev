import logging
import pickle as pkl
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import Field

from corerl.agent.base import BaseAgent, BaseAgentConfig
from corerl.component.buffer import buffer_group
from corerl.component.critic.ensemble_critic import CriticConfig, EnsembleCritic
from corerl.component.network.utils import tensor, to_np
from corerl.component.policy_manager import ActionReturn, GACPolicyManager, GACPolicyManagerConfig
from corerl.configs.config import config
from corerl.data_pipeline.datatypes import TransitionBatch
from corerl.data_pipeline.pipeline import ColumnDescriptions, PipelineReturn
from corerl.messages.events import EventType
from corerl.state import AppState
from corerl.utils.device import device
from corerl.utils.math import exp_moving_avg

logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)


@config()
class GreedyACConfig(BaseAgentConfig):
    """
    Kind: internal

    Agent hyperparameters. For internal use only.
    These should never be modified for production unless
    for debugging. These may be modified in tests and
    research to illicit particular behaviors.
    """
    name: Literal["greedy_ac"] = "greedy_ac"

    critic: CriticConfig = Field(default_factory=CriticConfig)
    policy: GACPolicyManagerConfig = Field(default_factory=GACPolicyManagerConfig)

    loss_threshold: float = 0.0001
    """
    Kind: internal

    Minimum desired change in loss between updates. If the loss value changes
    by more than this magnitude, then continue performing updates.
    """

    loss_ema_factor: float = 0.75
    """
    Kind: internal

    Exponential moving average factor for early stopping based on loss.
    Closer to 1 means slower update to avg, closer to 0 means less averaging.
    """

    max_internal_actor_updates: int = 1
    """
    Number of actor updates per critic update. Early stopping is done
    using the loss_threshold. A minimum of 1 update will always be performed.
    """

    max_critic_updates: int = 1
    """
    Number of critic updates. Early stopping is done using the loss_threshold.
    A minimum of 1 update will always be performed.
    """

    eval_batch : bool = False
    """
    Toggle for using a separate batch for evaluation of the
    linesearch stopping condition.
    """

    # metrics
    ingress_loss : bool = True


class GreedyAC(BaseAgent):
    def __init__(self, cfg: GreedyACConfig, app_state: AppState, col_desc: ColumnDescriptions):
        super().__init__(cfg, app_state, col_desc)
        self.cfg = cfg
        self._col_desc = col_desc
        self.eval_batch = cfg.eval_batch

        self._policy_manager = GACPolicyManager(cfg.policy, app_state, self.state_dim, self.action_dim)

        # Critic can train on all transitions whereas the policy only trains on transitions that are at decision points
        self.critic = EnsembleCritic(cfg.critic, app_state, self.state_dim, self.action_dim)
        self.critic_buffer = buffer_group.dispatch(cfg.critic.buffer, app_state)

        self.ensemble = self.cfg.critic.buffer.ensemble

        # for early stopping
        self._last_critic_loss = 0.
        self._avg_critic_delta: float | None = None
        self._last_actor_loss = 0.
        self._avg_actor_delta: float | None = None


    @property
    def actor_percentile(self) -> float:
        return self._policy_manager.cfg.actor_percentile

    @property
    def is_policy_buffer_sampleable(self)-> bool:
        return self._policy_manager.buffer.is_sampleable

    def sample_policy_buffer(self) -> list[TransitionBatch]:
        return self._policy_manager.buffer.sample()

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, dict]:
        with torch.no_grad():
            return self._policy_manager.actor.log_prob(states, actions)

    def prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_prob(states, actions)[0])

    def get_action_interaction(
        self,
        state: np.ndarray,
        action_lo: np.ndarray,
        action_hi: np.ndarray,
    ) -> np.ndarray:
        """
        Samples a single action during interaction.
        """
        self._app_state.event_bus.emit_event(EventType.agent_get_action)

        tensor_state = tensor(state, device.device)
        tensor_state = tensor_state.unsqueeze(0)
        tensor_action_lo = tensor(action_lo, device.device).unsqueeze(0)
        tensor_action_hi = tensor(action_hi, device.device).unsqueeze(0)

        ar = self._policy_manager.get_actor_actions(
            1,
            tensor_state,
            tensor_action_lo,
            tensor_action_hi,
            self.critic
        )
        direct_action = ar.direct_actions

        return to_np(direct_action)[0]

    def policy_to_direct_action(
        self,
        policy_actions: torch.Tensor,
        action_lo: torch.Tensor,
        action_hi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Converts policy actions to direct actions.
        """
        return self._policy_manager.ensure_direct_action(action_lo, action_hi, policy_actions)

    def get_actor_actions(
        self,
        n_samples: int,
        states: torch.Tensor,
        action_lo: torch.Tensor,
        action_hi: torch.Tensor,
    ) -> ActionReturn:
        """
        Sample actions from actor and return both direct and policy actions.
        """
        return self._policy_manager.get_actor_actions(
            n_samples,
            states,
            action_lo,
            action_hi,
            self.critic,
        )

    def get_sampler_actions(
        self,
        n_samples: int,
        states: torch.Tensor,
        action_lo: torch.Tensor,
        action_hi: torch.Tensor,
    ) -> ActionReturn:
        """
        Sample actions from sampler and return both direct and policy actions.
        """
        return self._policy_manager.get_sampler_actions(
            n_samples,
            states,
            action_lo,
            action_hi
        )

    def get_uniform_actions(
        self,
        n_samples: int,
        states: torch.Tensor,
        action_lo: torch.Tensor,
        action_hi: torch.Tensor,
    ) -> ActionReturn:
        """
        Sample actions UAR and return both direct and policy actions.
        """
        return self._policy_manager.get_uniform_actions(
            n_samples,
            states,
            action_lo,
            action_hi
        )

    def update_buffer(self, pr: PipelineReturn) -> None:
        if pr.transitions is None:
            return

        self._app_state.event_bus.emit_event(EventType.agent_update_buffer)
        recent_critic_idxs = self.critic_buffer.feed(pr.transitions, pr.data_mode)
        self._policy_manager.update_buffer(pr)

        # ---------------------------------- ingress loss metic --------------------------------- #

        if self.cfg.ingress_loss and len(recent_critic_idxs) > 0:
            recent_critic_batch = self.critic_buffer.get_batch(recent_critic_idxs)
            duplicated_critic_batch = [recent_critic_batch for i in range(self.ensemble)]
            bootstrap_actions = self._get_bootstrap_actions(duplicated_critic_batch)
            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric=f"ingress_critic_loss_{pr.data_mode.name}",
                value=self.critic.compute_loss(duplicated_critic_batch, bootstrap_actions).item(),
            )

        # ------------------------- transition length metric ------------------------- #

        for t in pr.transitions:
            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric="pipeline_transition_len",
                value=len(t),
            )

    def load_buffer(self, pr: PipelineReturn) -> None:
        if pr.transitions is None:
            return

        self._policy_manager.buffer.reset()
        self.critic_buffer.reset()
        self._policy_manager.buffer.feed(pr.transitions, pr.data_mode)
        self.critic_buffer.feed(pr.transitions, pr.data_mode)
        self._policy_manager.buffer.app_state = self._app_state
        self.critic_buffer.app_state = self._app_state

    # --------------------------- critic updating-------------------------- #

    def _get_bootstrap_actions(
        self,
        batches: list[TransitionBatch]
    ):
        next_actions: list[torch.Tensor] = []

        for batch in batches:
            with torch.no_grad():
                cur_action = batch.post.action
                dp_mask = batch.post.dp
                ar = self._policy_manager.get_actor_actions(
                    1,
                    batch.post.state,
                    batch.post.action_lo,
                    batch.post.action_hi,
                    self.critic,
                )
                next_direct_actions = ar.direct_actions
                next_direct_actions = (dp_mask * next_direct_actions) + ((1.0 - dp_mask) * cur_action)

            next_actions.append(next_direct_actions)

        return next_actions


    def update_critic(self) -> list[float]:
        if not self.critic_buffer.is_sampleable:
            return [0 for _ in range(self.ensemble)]

        batches = self.critic_buffer.sample()
        bootstrap_actions = self._get_bootstrap_actions(batches)
        eval_actions = bootstrap_actions

        if self.eval_batch:
            eval_batches = self.critic_buffer.sample()
            eval_actions = self._get_bootstrap_actions(eval_batches)
        else:
            eval_batches = batches

        q_loss = self.critic.update(batches, bootstrap_actions, eval_batches, eval_actions)
        return [float(q_loss)]


    def update(self) -> list[float]:
        q_losses = []

        alpha = self.cfg.loss_ema_factor
        for _ in range(self.cfg.max_critic_updates):
            losses = self.update_critic()
            q_losses += losses
            avg_critic_loss = np.mean(losses)

            for _ in range(self.cfg.max_internal_actor_updates):
                actor_loss = self._policy_manager.update(self.critic)

                last = self._last_actor_loss
                self._last_actor_loss = actor_loss
                delta = actor_loss - last
                self._avg_actor_delta = exp_moving_avg(alpha, self._avg_actor_delta, delta)

                if np.abs(self._avg_actor_delta) < self.cfg.loss_threshold:
                    break

            last = self._last_critic_loss
            self._last_critic_loss = avg_critic_loss
            delta = avg_critic_loss - last
            self._avg_critic_delta = exp_moving_avg(alpha, self._avg_critic_delta, delta)

            if np.abs(self._avg_critic_delta) < self.cfg.loss_threshold:
                break

        return q_losses

    # ---------------------------- saving and loading ---------------------------- #

    def save(self, path: Path) -> None:
        self._app_state.event_bus.emit_event(EventType.agent_save)

        path.mkdir(parents=True, exist_ok=True)
        actor_path = path / "actor"
        self._policy_manager.save(actor_path)

        q_critic_path = path / "q_critic"
        self.critic.save(q_critic_path)

        critic_buffer_path = path / "critic_buffer.pkl"
        with open(critic_buffer_path, "wb") as f:
            pkl.dump(self.critic_buffer, f)

    def load(self, path: Path) -> None:
        self._app_state.event_bus.emit_event(EventType.agent_load)

        actor_path = path / "actor"
        self._policy_manager.load(actor_path)

        q_critic_path = path / "q_critic"
        self.critic.load(q_critic_path)

        critic_buffer_path = path / "critic_buffer.pkl"
        with open(critic_buffer_path, "rb") as f:
            self.critic_buffer = pkl.load(f)

    def get_buffer_sizes(self) -> dict[str, list[int]]:
        return {
            "critic": self.critic_buffer.size,
            "policy": self._policy_manager.buffer.size,
        }

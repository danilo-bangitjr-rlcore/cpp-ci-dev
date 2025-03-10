import logging
import pickle as pkl
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import Field

from corerl.agent.base import BaseAgent, BaseAgentConfig
from corerl.component.buffer import MixedHistoryBuffer
from corerl.component.critic.ensemble_critic import CriticConfig, EnsembleCritic
from corerl.component.network.utils import tensor, to_np
from corerl.component.policy_manager import ActionReturn, GACPolicyManager, GACPolicyManagerConfig
from corerl.configs.config import config
from corerl.data_pipeline.datatypes import TransitionBatch
from corerl.data_pipeline.pipeline import ColumnDescriptions, PipelineReturn
from corerl.messages.events import EventType
from corerl.state import AppState
from corerl.utils.device import device

logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)

EPSILON = 1e-6


@config()
class GreedyACConfig(BaseAgentConfig):
    name: Literal["greedy_ac"] = "greedy_ac"

    critic: CriticConfig = Field(default_factory=CriticConfig)
    policy: GACPolicyManagerConfig = Field(default_factory=GACPolicyManagerConfig)

    n_actor_updates: int = 1
    n_critic_updates: int = 1

    ensemble_targets: bool = False
    eval_batch : bool = False

    # metrics
    ingress_loss : bool = True
    most_recent_batch_loss : bool = True


class GreedyAC(BaseAgent):
    def __init__(self, cfg: GreedyACConfig, app_state: AppState, col_desc: ColumnDescriptions):
        super().__init__(cfg, app_state, col_desc)
        self.cfg = cfg
        self._col_desc = col_desc
        self.ensemble_targets = cfg.ensemble_targets
        self.eval_batch = cfg.eval_batch

        self.n_actor_updates = cfg.n_actor_updates
        self.n_critic_updates = cfg.n_critic_updates

        self._policy_manager = GACPolicyManager(cfg.policy, app_state, self.state_dim, self.action_dim)

        # Critic can train on all transitions whereas the policy only trains on transitions that are at decision points
        self.critic = EnsembleCritic(cfg.critic, app_state, self.state_dim, self.action_dim)
        self.critic_buffer = MixedHistoryBuffer(cfg.critic.buffer, app_state)

        self.ensemble = self.cfg.critic.buffer.ensemble

    @property
    def delta_actions(self) -> bool:
        return self._policy_manager.cfg.delta_actions

    @property
    def actor_percentile(self) -> float:
        return self._policy_manager.cfg.actor_percentile

    @property
    def is_policy_buffer_sampleable(self)-> bool:
        return self._policy_manager.buffer.is_sampleable

    def sample_policy_buffer(self) -> list[TransitionBatch]:
        return self._policy_manager.buffer.sample()

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return self._policy_manager.actor.log_prob(states, actions)

    def get_action_interaction(self, state: np.ndarray, prev_direct_action: np.ndarray) -> np.ndarray:
        """
        Samples a single action during interaction.
        """
        self._app_state.event_bus.emit_event(EventType.agent_get_action)

        tensor_state = tensor(state, device.device)
        tensor_state = tensor_state.unsqueeze(0)
        tensor_prev_direct_action = tensor(prev_direct_action, device.device)
        tensor_prev_direct_action = tensor_prev_direct_action.unsqueeze(0)

        ar = self._policy_manager.get_actor_actions(
            tensor_state,
            tensor_prev_direct_action,
        )
        direct_action = ar.direct_actions

        return to_np(direct_action)[0]

    def get_actor_actions(self, states: torch.Tensor, prev_direct_actions: torch.Tensor) -> ActionReturn:
        """
        Sample actions from actor and return both direct and policy actions.
        """
        return self._policy_manager.get_actor_actions(
            states,
            prev_direct_actions,
        )

    def get_sampler_actions(self, states: torch.Tensor, prev_direct_actions: torch.Tensor) -> ActionReturn:
        """
        Sample actions from sampler and return both direct and policy actions.
        """
        return self._policy_manager.get_sampler_actions(
            states,
            prev_direct_actions,
        )

    def get_uniform_actions(self, states: torch.Tensor, prev_direct_actions: torch.Tensor) -> ActionReturn:
        """
        Sample actions UAR and return both direct and policy actions.
        """
        return self._policy_manager.get_uniform_actions(
            states,
            prev_direct_actions,
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
            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric=f"ingress_critic_loss_{pr.data_mode.name}",
                value=self._compute_critic_loss(duplicated_critic_batch),
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

    def _compute_critic_loss(
            self,
            ensemble_batch: list[TransitionBatch],
            with_grad: bool=False,
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
        for batch in ensemble_batch:
            state_batch = batch.prior.state
            direct_action_batch = batch.post.action
            reward_batch = batch.n_step_reward
            next_state_batch = batch.post.state
            gamma_batch = batch.n_step_gamma
            dp_mask = batch.post.dp

            ar = self._policy_manager.get_actor_actions(next_state_batch,  direct_action_batch)
            next_direct_actions = ar.direct_actions

            # For the 'Anytime' paradigm, only states at decision points can sample next_actions
            # If a state isn't at a decision point, its next_action is set to the current action
            with torch.no_grad():
                next_direct_actions = (dp_mask * next_direct_actions) + ((1.0 - dp_mask) * direct_action_batch)

            state_batches.append(state_batch)
            action_batches.append(direct_action_batch)
            reward_batches.append(reward_batch)
            next_state_batches.append(next_state_batch)
            next_action_batches.append(next_direct_actions)
            gamma_batches.append(gamma_batch)

        # Second, use this information to compute the targets
        target_values = self.critic.get_target_values(next_state_batches, next_action_batches)

        # Third, compute losses
        values = self.critic.get_values(state_batches, action_batches, with_grad=with_grad)
        loss = torch.tensor(0.0, device=device.device)
        for i in range(ensemble_len):
            target =  reward_batches[i] + gamma_batches[i] * target_values.ensemble_values[i]
            loss_i = torch.nn.functional.mse_loss(target, values.ensemble_values[i])
            loss += loss_i

            if log_metrics:
                self._app_state.metrics.write(
                    agent_step=self._app_state.agent_step,
                    metric=f"critic_loss_{i}",
                    value=to_np(loss_i),
                )

        if log_metrics and values.ensemble_variance is not None:
            mean_variance = torch.mean(values.ensemble_variance)
            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric="critic_ensemble_variance",
                value=to_np(mean_variance),
            )

        if log_metrics:
            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric="avg_critic_loss",
                value=to_np(loss)/ensemble_len,
            )

        return loss

    def update_critic(self) -> list[float]:
        if not self.critic_buffer.is_sampleable:
            return []

        self._app_state.event_bus.emit_event(EventType.agent_update_critic)
        batches = self.critic_buffer.sample()
        q_loss = self._compute_critic_loss(batches, with_grad=True, log_metrics=True)

        log_most_recent_batch_loss = self.cfg.most_recent_batch_loss and self.critic_buffer.n_most_recent > 0
        if log_most_recent_batch_loss:
            # grab the most recent samples from the batch and log the loss on only these samples
            batch_slices = [b[:self.critic_buffer.n_most_recent] for b in batches]
            n_most_recent_loss = self._compute_critic_loss(batch_slices)

            self._app_state.metrics.write(
                    agent_step=self._app_state.agent_step,
                    metric=f"critic_loss_{self.critic_buffer.n_most_recent}_most_recent",
                    value=n_most_recent_loss,
            )

        if self.eval_batch:
            eval_batches = self.critic_buffer.sample()
        else:
            eval_batches = batches

        self.critic.update(q_loss, closure=lambda: self._compute_critic_loss(eval_batches).item())
        return [float(q_loss)]

    def update(self) -> list[float]:
        q_losses = []
        for _ in range(self.n_actor_updates):
            for _ in range(self.n_critic_updates):
                q_loss = self.update_critic()
                q_losses += q_loss

            self._policy_manager.update(self.critic)

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

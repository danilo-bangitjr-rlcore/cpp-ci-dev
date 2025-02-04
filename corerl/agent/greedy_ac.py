import logging
import pickle as pkl
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import numpy
import numpy as np
import torch
from jaxtyping import Float
from pydantic import Field

from corerl.agent.base import BaseAC, BaseACConfig
from corerl.component.actor.factory import init_actor
from corerl.component.actor.network_actor import NetworkActorConfig
from corerl.component.buffer.factory import init_buffer
from corerl.component.critic.ensemble_critic import EnsembleCriticConfig
from corerl.component.network.utils import state_to_tensor, to_np
from corerl.configs.config import config
from corerl.data_pipeline.datatypes import TransitionBatch
from corerl.data_pipeline.pipeline import ColumnDescriptions, PipelineReturn
from corerl.data_pipeline.transforms.delta import Delta
from corerl.messages.events import EventType
from corerl.state import AppState
from corerl.utils.device import device

logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)

EPSILON = 1e-6


@dataclass
class PolicyUpdateInfo:
    state_batch: Float[torch.Tensor, "batch_size state_dim"]
    repeated_states: Float[torch.Tensor, "batch_size*num_samples state_dim"]
    sample_actions: Float[torch.Tensor, "batch_size num_samples action_dim"]
    sorted_q_inds: Float[torch.Tensor, "batch_size num_samples"]
    stacked_s_batch: Float[torch.Tensor, "batch_size*top_actions state_dim"]
    best_actions: Float[torch.Tensor, "batch_size*num_samples action_dim"]
    batch_size: int


@config(frozen=True)
class GreedyACConfig(BaseACConfig):
    name: Literal["greedy_ac"] = "greedy_ac"

    average_entropy: bool = True
    ensemble_targets: bool = False
    interleave_updates: bool = True
    n_sampler_updates: int = 1
    num_samples: int = 500
    prop_rho_mult: float = 2.0
    rho: float = 0.1
    share_batch: bool = True
    tau: float = 0.0
    uniform_sampling_percentage: float = 0.5

    actor: NetworkActorConfig = Field(default_factory=NetworkActorConfig)
    critic: EnsembleCriticConfig = Field(default_factory=EnsembleCriticConfig)


class GreedyAC(BaseAC):
    def __init__(self, cfg: GreedyACConfig, app_state: AppState, col_desc: ColumnDescriptions):
        super().__init__(cfg, app_state, col_desc)
        self.cfg = cfg
        self._col_desc = col_desc
        self.ensemble_targets = cfg.ensemble_targets

        # Whether to average the proposal policy's entropy over all the sampled actions
        self.average_entropy = cfg.average_entropy
        # Entropy constant used in the entropy version of the proposal policy update
        self.tau = cfg.tau
        # percentage of sampled actions used in actor update
        self.rho = cfg.rho
        # percentage of sampled actions used in the non-entropy version of the proposal policy update
        self.rho_proposal = self.rho * cfg.prop_rho_mult

        # number of actions sampled from the proposal policy
        self.num_samples = cfg.num_samples
        # whether updates to proposal and actor should share a batch
        self.share_batch = cfg.share_batch

        self.uniform_sampling_percentage = cfg.uniform_sampling_percentage
        self.learned_proposal_percent = 1 - self.uniform_sampling_percentage
        self.uniform_proposal = self.uniform_sampling_percentage == 1

        self._interleave_updates = cfg.interleave_updates

        self.n_sampler_updates = cfg.n_sampler_updates
        if self.share_batch and not self.uniform_proposal:
            assert self.n_actor_updates == self.n_sampler_updates, "Actor and proposal must use same number of updates"

        self.top_actions = int(self.rho * self.num_samples)  # Number of actions used to update actor
        self.top_actions_proposal = int(
            self.rho_proposal * self.num_samples
        )  # Number of actions used to update proposal policy

        self.sampler = init_actor(cfg.actor, self.state_dim, self.action_dim, initializer=self.actor)
        # Critic can train on all transitions whereas the policy only trains on transitions that are at decision points
        self.critic_buffer = init_buffer(cfg.critic.buffer)
        self.policy_buffer = init_buffer(cfg.actor.buffer)

        if self.discrete_control:
            self.top_actions = 1
            if self.num_samples >= self.action_dim:
                self.num_samples = self.action_dim
                self.top_actions_proposal = self.action_dim

    def get_action(self, state: numpy.ndarray) -> numpy.ndarray:
        self._app_state.event_bus.emit_event(EventType.agent_get_action)

        tensor_state = state_to_tensor(state, device.device)

        action, _action_info = self.actor.get_action(
            tensor_state,
            with_grad=False,
        )
        return to_np(action)[0]

    def update_buffer(self, pr: PipelineReturn) -> None:
        if pr.transitions is None:
            return

        self._app_state.event_bus.emit_event(EventType.agent_update_buffer)

        self.critic_buffer.feed(pr.transitions, pr.data_mode)
        self.policy_buffer.feed([t for t in pr.transitions if t.prior.dp], pr.data_mode)

    def load_buffer(self, pr: PipelineReturn) -> None:
        if pr.transitions is None:
            return

        policy_transitions = []
        for transition in pr.transitions:
            if transition.prior.dp:
                policy_transitions.append(transition)

        self.policy_buffer.load(policy_transitions, pr.data_mode)
        self.critic_buffer.load(pr.transitions, pr.data_mode)

    def get_sorted_q_values(
        self,
        state_batch: Float[torch.Tensor, "batch_size state_dim"],
        sample_actions: Float[torch.Tensor, "batch_size num_samples action_dim"],
    ) -> Float[torch.Tensor, "batch_size num_samples 1"]:
        # https://github.com/samuelfneumann/GreedyAC/blob/master/agent/nonlinear/GreedyAC.py

        batch_size = state_batch.shape[0]
        num_samples = sample_actions.shape[1]
        action_dim = sample_actions.shape[2]

        repeated_states = state_batch.repeat_interleave(num_samples, dim=0)
        flattened_actions = sample_actions.view(batch_size * num_samples, action_dim)

        q_values: Float[torch.Tensor, "batch_size*num_samples 1"]
        q_values = self.q_critic.get_q([repeated_states], [flattened_actions], with_grad=False, bootstrap_reduct=False)
        q_values = q_values.view(batch_size, num_samples, 1)
        sorted_q_inds: Float[torch.Tensor, "batch_size num_samples 1"]
        sorted_q_inds = torch.argsort(q_values, dim=1, descending=True)

        self._app_state.metrics.write(
            agent_step=self._app_state.agent_step,
            metric="top_q_value",
            value=q_values.max().item(),
        )

        return sorted_q_inds

    def get_uniform_action_samples(
        self, batch_size: int, n_rand: int
    ) -> Float[torch.Tensor, "batch_size n_rand action_dim"]:
        if self.discrete_control:
            action_indices = torch.randint(high=self.action_dim, size=(batch_size, n_rand, 1))
            empty_action_onehot = torch.zeros(size=(batch_size, n_rand, self.action_dim))
            uniform_sample_actions = empty_action_onehot.scatter_(dim=-1, index=action_indices, value=1)
        else:
            uniform_sample_actions = torch.rand(batch_size, n_rand, self.action_dim)
            uniform_sample_actions = torch.clip(uniform_sample_actions, EPSILON, 1)

        return uniform_sample_actions.to(device.device)

    def get_sampled_actions(
        self, state_batch: Float[torch.Tensor, "batch_size state_dim"]
    ) -> Float[torch.Tensor, "batch_size num_samples action_dim"]:
        batch_size = state_batch.shape[0]

        if self.uniform_proposal:
            self.uniform_sampling_percentage = 1.0

        n_proposal = int(
            np.floor(self.num_samples * self.learned_proposal_percent),
        )
        if n_proposal > 0:
            repeated_states: Float[torch.Tensor, "batch_size*n_proposal state_dim"]
            repeated_states = state_batch.repeat_interleave(n_proposal, dim=0)

            proposed_actions: Float[torch.Tensor, "batch_size*n_proposal action_dim"]
            proposed_actions, _ = self.sampler.get_action(
                repeated_states,
                with_grad=False,
            )
            proposed_actions = proposed_actions.reshape(batch_size, n_proposal, self.action_dim)

        else:
            proposed_actions = torch.empty(batch_size, 0, self.action_dim)

        if self.uniform_sampling_percentage > 0:
            n_rand = int(np.ceil(self.num_samples * self.uniform_sampling_percentage))
            uniform_sample_actions = self.get_uniform_action_samples(batch_size=batch_size, n_rand=n_rand)

        else:
            uniform_sample_actions = torch.empty(batch_size, 0, self.action_dim)

        SAMPLE_DIM = 1
        sample_actions = torch.cat([proposed_actions, uniform_sample_actions], dim=SAMPLE_DIM)
        logger.debug(f"{proposed_actions.shape=}")
        logger.debug(f"{uniform_sample_actions.shape=}")

        return sample_actions

    def get_top_actions(
        self,
        proposed_actions: Float[torch.Tensor, "batch_size num_samples action_dim"],
        sorted_q_inds: Float[torch.Tensor, "batch_size num_samples 1"],
        n_top_actions: int,
    ) -> Float[torch.Tensor, "batch_size*n_top_actions action_dim"]:
        best_ind: Float[torch.Tensor, "batch_size n_top_actions action_dim"]
        best_ind = sorted_q_inds[:, :n_top_actions].repeat_interleave(self.action_dim, -1)

        # recall that if self.sample_all_discrete_actions then self.num_samples = self.action_dim
        best_actions = torch.gather(proposed_actions, dim=1, index=best_ind)
        batch_size = proposed_actions.shape[0]
        best_actions = torch.reshape(best_actions, (batch_size * n_top_actions, self.action_dim))

        return best_actions

    def get_policy_update_info(
        self,
        state_batch: Float[torch.Tensor, "batch_size state_dim"],
        action_batch: Float[torch.Tensor, "batch_size action_dim"],
    ) -> PolicyUpdateInfo:
        batch_size = state_batch.shape[0]
        # recall that if self.sample_all_discrete_actions then self.num_samples = self.action_dim
        repeated_states: Float[torch.Tensor, "batch_size*num_samples state_dim"]
        repeated_states = state_batch.repeat_interleave(self.num_samples, dim=0)

        # if we are in direct action mode, then the sampled
        # actions are direct and not deltas. So zero out the
        # offset
        if not self.cfg.delta_action:
            action_batch = torch.zeros_like(action_batch)

        sample_actions = self.get_sampled_actions(state_batch)
        sample_actions = self.ensure_direct_action(action_batch, sample_actions)
        sorted_q_inds = self.get_sorted_q_values(state_batch, sample_actions)
        best_actions = self.get_top_actions(sample_actions, sorted_q_inds, n_top_actions=self.top_actions)

        # Reshape samples for calculating the loss
        stacked_s_batch: Float[torch.Tensor, "batch_size*top_actions state_dim"]
        stacked_s_batch = state_batch.repeat_interleave(self.top_actions, dim=0)

        update_info = PolicyUpdateInfo(
            state_batch,
            repeated_states,
            sample_actions,
            sorted_q_inds,
            stacked_s_batch,
            best_actions,
            batch_size,
        )

        return update_info

    def filter_only_direct_actions(self, actions: torch.Tensor):
        if not self.cfg.delta_action:
            return actions

        direct_idxs = [i for i, col in enumerate(self._col_desc.action_cols) if not Delta.is_delta_transformed(col)]
        return actions[:, direct_idxs]

    def filter_only_delta_actions(self, actions: torch.Tensor):
        if not self.cfg.delta_action:
            return actions

        delta_idxs = [i for i, col in enumerate(self._col_desc.action_cols) if Delta.is_delta_transformed(col)]
        return actions[:, delta_idxs]

    def ensure_direct_action(self, action: torch.Tensor, next_action: torch.Tensor):
        if not self.cfg.delta_action:
            return next_action

        bounds = self.cfg.delta_bounds
        assert bounds is not None, "Delta actions are enabled, however the agent has no delta bounds"
        scale = bounds[1] - bounds[0]
        bias = bounds[0]

        # when updating the policy, our next action is a tensor
        # (batch_size, num_samples from proposal, action_dim)
        # however, our direct action offset is only (batch_size, action_dim)
        if len(next_action.shape) == 3:
            action = action.unsqueeze(1).expand(next_action.shape)

        delta = scale * next_action + bias
        direct_action = action + delta

        # because we are always operating in normalized space,
        # we can hardcode the spatial constraints
        return torch.clip(direct_action, 0, 1)


    def compute_critic_loss(self, ensemble_batch: list[TransitionBatch]) -> list[torch.Tensor]:
        # First, translate ensemble batches in to list for each property
        ensemble_len = len(ensemble_batch)
        state_batches = []
        action_batches = []
        reward_batches = []
        next_state_batches = []
        next_action_batches = []
        gamma_batches = []
        next_qs = []
        for batch in ensemble_batch:
            state_batch = batch.prior.state
            action_batch = batch.post.action
            reward_batch = batch.n_step_reward
            next_state_batch = batch.post.state
            gamma_batch = batch.n_step_gamma
            dp_mask = batch.post.dp

            # put actions into direct form
            next_actions, _ = self.actor.get_action(next_state_batch, with_grad=False)
            action_batch = self.filter_only_direct_actions(action_batch)
            next_actions = self.ensure_direct_action(action_batch, next_actions)
            # For the 'Anytime' paradigm, only states at decision points can sample next_actions
            # If a state isn't at a decision point, its next_action is set to the current action
            with torch.no_grad():
                next_actions = (dp_mask * next_actions) + ((1.0 - dp_mask) * action_batch)

            state_batches.append(state_batch)
            action_batches.append(action_batch)
            reward_batches.append(reward_batch)
            next_state_batches.append(next_state_batch)
            next_action_batches.append(next_actions)
            gamma_batches.append(gamma_batch)

        # Second, use this information to compute the targets
            # Option 1: Using the corresponding target function in the ensemble in the update target:
        if self.ensemble_targets:
            _, next_qs = self.q_critic.get_qs_target(next_state_batches, next_action_batches, bootstrap_reduct=True)

            # Option 2: Using the reduction of the ensemble in the update target:
        else:
            next_qs = []
            for i in range(ensemble_len):
                next_q = self.q_critic.get_q_target([next_state_batches[i]], [next_action_batches[i]], bootstrap_reduct=True)
                next_q = torch.unsqueeze(next_q, 0)
                next_qs.append(next_q)

            next_qs = torch.cat(next_qs, dim=0)

        targets = [reward_batches[i] + gamma_batches[i] * next_qs[i] for i in range(ensemble_len)]

        # Third, compute losses
        _, qs = self.q_critic.get_qs(state_batches, action_batches, with_grad=True, bootstrap_reduct=True)
        losses = []

        for i in range(ensemble_len):
            target = targets[i]
            losses.append(torch.nn.functional.mse_loss(target, qs[i]))

        # Fourth, log metrics
        self._app_state.metrics.write(
            agent_step=self._app_state.agent_step,
            metric="critic_loss",
            value=np.mean([loss.detach().numpy() for loss in losses]),
        )

        return losses

    def compute_sampler_entropy_loss(self, update_info: PolicyUpdateInfo) -> torch.Tensor:
        sample_actions = update_info.sample_actions.reshape(-1, self.action_dim)
        sampler_entropy, _ = self.sampler.get_log_prob(update_info.repeated_states, sample_actions)

        with torch.no_grad():
            sampler_entropy *= sampler_entropy
        sampler_entropy = sampler_entropy.reshape(update_info.batch_size, self.num_samples, 1)

        if self.average_entropy:
            sampler_entropy = -sampler_entropy.mean(dim=1)
        else:
            sampler_entropy = -sampler_entropy[:, 0, :]

        logp, _ = self.sampler.get_log_prob(update_info.stacked_s_batch, update_info.best_actions, with_grad=True)
        sampler_loss = logp.reshape(update_info.batch_size, self.top_actions, 1)
        sampler_loss = -1 * (sampler_loss.mean(dim=1) + self.tau * sampler_entropy).mean()

        return sampler_loss

    def compute_sampler_no_entropy_loss(self, update_info: PolicyUpdateInfo) -> torch.Tensor:
        # gets the best actions
        best_ind_proposal = update_info.sorted_q_inds[:, : self.top_actions_proposal]
        best_ind_proposal = best_ind_proposal.repeat_interleave(self.action_dim, -1)
        best_actions_proposal = torch.gather(update_info.sample_actions, 1, best_ind_proposal)
        stacked_s_batch_proposal = update_info.state_batch.repeat_interleave(self.top_actions_proposal, dim=0)
        best_actions_proposal = torch.reshape(best_actions_proposal, (-1, self.action_dim))

        logp, _ = self.sampler.get_log_prob(stacked_s_batch_proposal, best_actions_proposal, with_grad=True)
        sampler_loss = logp.reshape(update_info.batch_size, self.top_actions_proposal, 1)
        sampler_loss = -1 * (sampler_loss.mean(dim=1)).mean()

        return sampler_loss

    def compute_actor_loss(
        self,
        update_info: PolicyUpdateInfo,
    ) -> torch.Tensor:
        logp, _ = self.actor.get_log_prob(
            update_info.stacked_s_batch,
            update_info.best_actions,
            with_grad=True,
        )

        actor_loss = -logp.mean()

        self._app_state.metrics.write(
            agent_step=self._app_state.agent_step,
            metric="actor_loss",
            value=actor_loss,
        )

        return actor_loss

    def compute_sampler_loss(
        self,
        update_info: PolicyUpdateInfo,
    ) -> torch.Tensor:
        if self.tau != 0:  # Entropy version of proposal policy update
            sampler_loss = self.compute_sampler_entropy_loss(update_info)
        else:
            # Non-entropy version of proposal policy update.
            # A greater percentage of actions are used to update the proposal policy than the actor policy
            sampler_loss = self.compute_sampler_no_entropy_loss(update_info)

        self._app_state.metrics.write(
            agent_step=self._app_state.agent_step,
            metric="sampler_loss",
            value=sampler_loss,
        )

        return sampler_loss

    def update_critic(self) -> list[float]:
        if min(self.critic_buffer.size) <= 0:
            return []

        self._app_state.event_bus.emit_event(EventType.agent_update_critic)

        batches = self.critic_buffer.sample()

        def closure():
            losses = self.compute_critic_loss(batches)  # noqa: B023
            return torch.stack(losses, dim=-1).sum(dim=-1)

        q_loss = closure()
        self.q_critic.update(q_loss, opt_kwargs={"closure": closure})

        return [float(q_loss)]

    def update_actor(self) -> PolicyUpdateInfo | None:
        self._app_state.event_bus.emit_event(EventType.agent_update_actor)

        if min(self.policy_buffer.size) <= 0:
            return None

        batches = self.policy_buffer.sample()
        # Assuming we don't have an ensemble of policies
        assert len(batches) == 1
        batch = batches[0]

        # if in direct action mode, this is a no-op
        action = self.filter_only_delta_actions(batch.post.action)
        update_info = self.get_policy_update_info(batch.prior.state, action)
        actor_loss = self.compute_actor_loss(update_info)

        self.actor.update(
            actor_loss,
            opt_kwargs={
                "closure": partial(self.actor_err, update_info.stacked_s_batch, update_info.best_actions),
            },
        )

        return update_info

    def update_sampler(self, update_infos: list[PolicyUpdateInfo] | None) -> None:
        if self.uniform_proposal:
            return

        if update_infos is not None:
            for update_info in update_infos:
                sampler_loss = self.compute_sampler_loss(update_info)

                self.sampler.update(
                    sampler_loss,
                    opt_kwargs={"closure": partial(
                        self.sampler_err,
                        update_info.stacked_s_batch,
                        update_info.best_actions)},
                )

        else:
            if min(self.policy_buffer.size) <= 0:
                return

            batches = self.policy_buffer.sample()
            assert len(batches) == 1
            batch = batches[0]

            # if in direct action mode, this is a no-op
            action = self.filter_only_direct_actions(batch.post.action)
            update_info = self.get_policy_update_info(batch.prior.state, action)

            sampler_loss = self.compute_sampler_loss(update_info)

            self.sampler.update(
                sampler_loss,
                opt_kwargs={
                    "closure": partial(self.sampler_err,
                                       update_info.stacked_s_batch,
                                       update_info.best_actions),
                },
            )

    def actor_err(self, stacked_s_batch: torch.Tensor, best_actions: torch.Tensor) -> torch.Tensor:
        logp, _ = self.actor.get_log_prob(
            stacked_s_batch,
            best_actions,
            with_grad=True,
        )
        return -logp.mean()

    def sampler_err(self, stacked_s_batch: torch.Tensor, best_actions: torch.Tensor) -> torch.Tensor:
        logp, _ = self.sampler.get_log_prob(
            stacked_s_batch,
            best_actions,
            with_grad=True,
        )
        return -logp.mean()

    def update(self) -> list[float]:
        if self._interleave_updates:
            q_losses = self._update_interleave()
            return q_losses

        q_losses = self._update_sequential()

        return q_losses

    def _update_interleave(self) -> list[float]:
        n_sampler_updates = 0
        q_losses = []
        for _ in range(self.n_actor_updates):
            for _ in range(self.n_critic_updates):
                q_loss = self.update_critic()
                q_losses += q_loss

            update_info = self.update_actor()

            # if update_info is None, the sampler will sample its own states, actions, etc to use in updating
            # this happens with self.share_batch is False, and when update_actor returns None
            update_info = [update_info] if update_info is not None and self.share_batch else None

            if n_sampler_updates <= self.n_sampler_updates:
                self.update_sampler(update_infos=update_info)
                n_sampler_updates += 1

        return q_losses

    def _update_sequential(self) -> list[float]:
        q_losses = []
        for _ in range(self.n_critic_updates):
            q_loss = self.update_critic()
            q_losses += q_loss

        update_infos = []
        for _ in range(self.n_actor_updates):
            update_info = self.update_actor()
            if update_info is not None:
                update_infos.append(update_info)

        if self.share_batch:
            self.update_sampler(update_infos=update_infos)
        else:
            self.update_sampler(update_infos=None)

        return q_losses

    def save(self, path: Path) -> None:
        self._app_state.event_bus.emit_event(EventType.agent_save)

        path.mkdir(parents=True, exist_ok=True)
        actor_path = path / "actor"
        self.actor.save(actor_path)

        sampler_path = path / "sampler"
        self.sampler.save(sampler_path)

        q_critic_path = path / "q_critic"
        self.q_critic.save(q_critic_path)

        critic_buffer_path = path / "critic_buffer.pkl"
        with open(critic_buffer_path, "wb") as f:
            pkl.dump(self.critic_buffer, f)

        policy_buffer_path = path / "policy_buffer.pkl"
        with open(policy_buffer_path, "wb") as f:
            pkl.dump(self.policy_buffer, f)

    def load(self, path: Path) -> None:
        self._app_state.event_bus.emit_event(EventType.agent_load)

        actor_path = path / "actor"
        self.actor.load(actor_path)

        sampler_path = path / "sampler"
        self.sampler.load(sampler_path)

        q_critic_path = path / "q_critic"
        self.q_critic.load(q_critic_path)

        critic_buffer_path = path / "critic_buffer.pkl"
        with open(critic_buffer_path, "rb") as f:
            self.critic_buffer = pkl.load(f)

        policy_buffer_path = path / "policy_buffer.pkl"
        with open(policy_buffer_path, "rb") as f:
            self.policy_buffer = pkl.load(f)

    def get_buffer_sizes(self) -> dict[str, list[int]]:
        return {
            "critic": self.critic_buffer.size,
            "policy": self.policy_buffer.size,
        }

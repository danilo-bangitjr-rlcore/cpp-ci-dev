import logging
import pickle as pkl
from dataclasses import dataclass
from functools import partial
from math import floor
from pathlib import Path
from typing import Literal

import numpy
import numpy as np
import torch
from jaxtyping import Float
from pydantic import Field

from corerl.agent.base import BaseAC, BaseACConfig
from corerl.component.actor.base_actor import BaseActor
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


# --------------------------------- Utilities -------------------------------- #

def sample_actions(
    state_batch: Float[torch.Tensor, "batch_size state_dim"],
    policy : BaseActor,
    n_samples : int,
    action_dim: int,
    uniform_weight : float = 0.0
) -> Float[torch.Tensor, "batch_size num_samples action_dim"]:
    """
    For each state in the state_batch, sample n actions according to policy.

    Returns a tensor with dimensions (batch_size, num_samples, action_dim)
    """
    batch_size = state_batch.shape[0]

    policy_weight = 1 - uniform_weight
    n_samples_policy = floor(policy_weight * n_samples) # number of samples from the policy
    n_samples_uniform = n_samples - n_samples_policy

    # sample n_samples_policy actions from the policy
    if n_samples_policy > 0:
        repeated_states: Float[torch.Tensor, "batch_size*n_samples_policy state_dim"]
        repeated_states = state_batch.repeat_interleave(n_samples_policy, dim=0)

        proposed_actions: Float[torch.Tensor, "batch_size*n_samples_policy action_dim"]
        proposed_actions, _ = policy.get_action(
            repeated_states,
            with_grad=False,
        )
        proposed_actions = proposed_actions.reshape(batch_size, n_samples_policy, action_dim)
    else:
        proposed_actions = torch.empty(batch_size, 0, action_dim)

    # sample remaining n_samples_uniform actions uniformly
    uniform_sample_actions = torch.rand(batch_size, n_samples_uniform, action_dim)
    uniform_sample_actions = torch.clip(uniform_sample_actions, EPSILON, 1)

    SAMPLE_DIM = 1
    sample_actions = torch.cat([proposed_actions, uniform_sample_actions], dim=SAMPLE_DIM)

    logger.debug(f"{proposed_actions.shape=}")
    logger.debug(f"{uniform_sample_actions.shape=}")

    sample_actions.to(device.device)

    return sample_actions


@config(frozen=True)
class GreedyACConfig(BaseACConfig):
    name: Literal["greedy_ac"] = "greedy_ac"

    ensemble_targets: bool = False
    n_sampler_updates: int = 1
    num_samples: int = 500
    prop_rho_mult: float = 2.0
    rho: float = 0.1
    share_batch: bool = True
    uniform_sampling_percentage: float = 0.5

    actor: NetworkActorConfig = Field(default_factory=NetworkActorConfig)
    critic: EnsembleCriticConfig = Field(default_factory=EnsembleCriticConfig)


class GreedyAC(BaseAC):
    def __init__(self, cfg: GreedyACConfig, app_state: AppState, col_desc: ColumnDescriptions):
        super().__init__(cfg, app_state, col_desc)
        self.cfg = cfg
        self._col_desc = col_desc
        self.ensemble_targets = cfg.ensemble_targets

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

        self.n_sampler_updates = cfg.n_sampler_updates
        if self.share_batch and not self.uniform_proposal:
            assert self.n_actor_updates == self.n_sampler_updates, "Actor and proposal must use same number of updates"

        self.sampler = init_actor(cfg.actor, self.state_dim, self.action_dim, initializer=self.actor)
        # Critic can train on all transitions whereas the policy only trains on transitions that are at decision points
        self.critic_buffer = init_buffer(cfg.critic.buffer)
        self.policy_buffer = init_buffer(cfg.actor.buffer)


    def get_action(self, state: numpy.ndarray) -> numpy.ndarray:
        self._app_state.event_bus.emit_event(EventType.agent_get_action)

        tensor_state = state_to_tensor(state, device.device)

        action, _ = self.actor.get_action(
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
                next_q = self.q_critic.get_q_target(
                    [next_state_batches[i]],
                    [next_action_batches[i]],
                    bootstrap_reduct=True)
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

    def _get_top_n_sampled_actions(
        self,
        state_batch: Float[torch.Tensor, "batch_size state_dim"],
        action_batch: Float[torch.Tensor, "batch_size action_dim"],
        n_samples: int,
        percentile: float,
        uniform_weight: float
    ) -> tuple[
            Float[torch.Tensor, "batch_size*top_actions state_dim"],
            Float[torch.Tensor, "batch_size*top_actions action_dim"]
    ]:

        # first, sample actions
        sampled_actions : Float[torch.Tensor, "batch_size num_samples action_dim"]
        sampled_actions = sample_actions(
            state_batch,
            self.sampler,
            n_samples,
            self.action_dim,
            uniform_weight)

        # Next, send the sampled actions though the critic to get a q value for each (state, action)
        batch_size = state_batch.shape[0]
        action_dim = sampled_actions.shape[2]

        repeated_states: Float[torch.Tensor, "batch_size*num_samples state_dim"]
        repeated_states = state_batch.repeat_interleave(n_samples, dim=0)
        flattened_actions = sampled_actions.view(batch_size * n_samples, action_dim)

        q_values: Float[torch.Tensor, "batch_size*num_samples 1"]
        q_values = self.q_critic.get_q([repeated_states], [flattened_actions], with_grad=False, bootstrap_reduct=False)
        q_values = q_values.view(batch_size, n_samples, 1)

        # Next, sort these q values
        sorted_q_inds: Float[torch.Tensor, "batch_size num_samples 1"]
        sorted_q_inds = torch.argsort(q_values, dim=1, descending=True)

        # Take the top percentile
        top_n = floor(percentile*n_samples)
        best_inds: Float[torch.Tensor, "batch_size n_top_actions action_dim"]
        best_inds = sorted_q_inds[:, :top_n].repeat_interleave(self.action_dim, -1)

        # grab the top_n best actions from sampled_actions
        best_actions = torch.gather(sampled_actions, dim=1, index=best_inds)
        batch_size = sampled_actions.shape[0]
        best_actions = torch.reshape(best_actions, (batch_size *top_n, self.action_dim))

        # also return the corresponding state for each of the top actions
        states_for_best_actions: Float[torch.Tensor, "batch_size*top_actions state_dim"]
        states_for_best_actions = state_batch.repeat_interleave(top_n, dim=0)

        return states_for_best_actions, best_actions

    def update_actor(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        self._app_state.event_bus.emit_event(EventType.agent_update_actor)

        if min(self.policy_buffer.size) <= 0:
            return None

        # Assuming we don't have an ensemble of policies
        batches = self.policy_buffer.sample()
        assert len(batches) == 1
        batch = batches[0]

        state_batch = batch.prior.state
        action_batch = self.filter_only_delta_actions(batch.post.action)
        # if we are in direct action mode, then the sampled
        # actions are direct and not deltas. So zero out the
        # offset
        if not self.cfg.delta_action:
            action_batch = torch.zeros_like(action_batch)

        states_for_best_actions, best_actions = self._get_top_n_sampled_actions(
            state_batch=state_batch,
            action_batch=action_batch,
            n_samples=self.num_samples,
            percentile=self.rho,
            uniform_weight=self.uniform_sampling_percentage
        )

        actor_loss = self.policy_err(states_for_best_actions, best_actions)

        self._app_state.metrics.write(
            agent_step=self._app_state.agent_step,
            metric="actor_loss",
            value=actor_loss,
        )

        self.actor.update(
            actor_loss,
            opt_kwargs={
                "closure": partial(self.policy_err, states_for_best_actions, best_actions),
            },
        )

        return batch.prior.state, action_batch

    def update_sampler(
            self,
            actor_update_return: tuple[torch.Tensor, torch.Tensor] | None = None,
            ) -> None:

        # return as no update necessary
        if self.uniform_proposal:
            return

        if min(self.policy_buffer.size) <= 0:
            return

        # no state_batch/action_batch passed in, must compute them
        if actor_update_return is None:
            # Assuming we don't have an ensemble of policies
            batches = self.policy_buffer.sample()
            assert len(batches) == 1
            batch = batches[0]

            state_batch = batch.prior.state
            action_batch = self.filter_only_delta_actions(batch.post.action)
            # if we are in direct action mode, then the sampled
            # actions are direct and not deltas. So zero out the
            # offset
            if not self.cfg.delta_action:
                action_batch = torch.zeros_like(action_batch)
        else:
            state_batch, action_batch = actor_update_return

        assert state_batch is None == action_batch is None
        assert state_batch is not None
        assert action_batch is not None

        states_for_best_actions, best_actions = self._get_top_n_sampled_actions(
            state_batch=state_batch,
            action_batch=action_batch,
            n_samples=self.num_samples,
            percentile=self.rho_proposal, #NOTE: using rho_proposal here
            uniform_weight=self.uniform_sampling_percentage
        )

        sampler_loss = self.policy_err(states_for_best_actions, best_actions)

        self._app_state.metrics.write(
            agent_step=self._app_state.agent_step,
            metric="sampler_loss",
            value=sampler_loss,
        )

        self.actor.update(
            sampler_loss,
            opt_kwargs={
                "closure": partial(self.policy_err, states_for_best_actions, best_actions),
            },
        )

    def policy_err(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logp, _ = self.actor.get_log_prob(
            states,
            actions,
            with_grad=True,
        )
        return -logp.mean()

    def update(self) -> list[float]:
        n_sampler_updates = 0
        q_losses = []
        for _ in range(self.n_actor_updates):
            for _ in range(self.n_critic_updates):
                q_loss = self.update_critic()
                q_losses += q_loss

            actor_update_return = self.update_actor()

            if not self.share_batch:
                actor_update_return = None # signals to update_sampler to sample a batch anew

            if n_sampler_updates <= self.n_sampler_updates:
                self.update_sampler(actor_update_return)
                n_sampler_updates += 1

        return q_losses

# ---------------------------- saving and loading ---------------------------- #

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

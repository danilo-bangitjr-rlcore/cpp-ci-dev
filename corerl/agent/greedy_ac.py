import logging
import pickle as pkl
from functools import partial
from math import floor
from pathlib import Path
from typing import Literal

import numpy
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
def unsqueeze_repeat(tensor: torch.Tensor, dim: int, repeats: int) -> torch.Tensor:
    tensor = tensor.unsqueeze(dim)
    tensor = tensor.repeat_interleave(repeats, dim=dim)
    return tensor


def sample_actions(
    state_batch: Float[torch.Tensor, "batch_size state_dim"],
    policy: BaseActor,
    n_samples: int,
    action_dim: int,
    uniform_weight: float = 0.0,
) -> tuple[
    Float[torch.Tensor, "batch_size num_samples action_dim"], Float[torch.Tensor, "batch_size num_samples state_dim"]
]:
    """
    For each state in the state_batch, sample n actions according to policy.

    Returns a tensor with dimensions (batch_size, num_samples, action_dim)
    """
    batch_size = state_batch.shape[0]

    policy_weight = 1 - uniform_weight
    n_samples_policy = floor(policy_weight * n_samples)  # number of samples from the policy
    n_samples_uniform = n_samples - n_samples_policy

    SAMPLE_DIM = 1
    # sample n_samples_policy actions from the policy
    repeated_states: Float[torch.Tensor, "batch_size n_samples_policy state_dim"]
    repeated_states = unsqueeze_repeat(state_batch, SAMPLE_DIM, n_samples_policy)
    proposed_actions: Float[torch.Tensor, "batch_size n_samples_policy action_dim"]
    proposed_actions, _ = policy.get_action(repeated_states, with_grad=False)

    # sample remaining n_samples_uniform actions uniformly
    uniform_sample_actions = torch.rand(batch_size, n_samples_uniform, action_dim)
    uniform_sample_actions = torch.clip(uniform_sample_actions, EPSILON, 1)

    sample_actions = torch.cat([proposed_actions, uniform_sample_actions], dim=SAMPLE_DIM)

    repeated_states: Float[torch.Tensor, "batch_size n_samples state_dim"]
    repeated_states = unsqueeze_repeat(state_batch, SAMPLE_DIM, n_samples)

    logger.debug(f"{proposed_actions.shape=}")
    logger.debug(f"{uniform_sample_actions.shape=}")

    sample_actions.to(device.device)

    return sample_actions, repeated_states


def grab_percentile(
        values: torch.Tensor,
        keys: torch.Tensor,
        percentile: float,
    ) -> torch.Tensor:
    assert keys.dim() == 3
    assert values.dim() == 2
    assert values.size(0) == keys.size(0)
    assert values.size(1) == keys.size(1)
    key_dim = keys.size(2)

    n_samples = values.size(1)
    top_n = floor(percentile * n_samples)

    values = values.squeeze(dim=-1)
    sorted_inds = torch.argsort(values, dim=1, descending=True)
    top_n_indices = sorted_inds[:, :top_n]
    top_n_indices = top_n_indices.unsqueeze(-1)
    top_n_indices = top_n_indices.repeat_interleave(key_dim, -1)
    return top_n_indices


@config(frozen=True)
class GreedyACConfig(BaseACConfig):
    name: Literal["greedy_ac"] = "greedy_ac"

    ensemble_targets: bool = False
    num_samples: int = 500
    prop_rho_mult: float = 2.0
    rho: float = 0.1
    share_batch: bool = True
    uniform_sampling_percentage: float = 0.5
    eval_batch : bool = True

    # metrics
    ingress_loss : bool = True
    most_recent_batch_loss : bool = True

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
        # whether the closure function in line search uses a separate batch for evaluation.
        self.eval_batch = cfg.eval_batch

        self.uniform_sampling_percentage = cfg.uniform_sampling_percentage
        self.learned_proposal_percent = 1 - self.uniform_sampling_percentage
        self.uniform_proposal = self.uniform_sampling_percentage == 1

        self.sampler = init_actor(cfg.actor, app_state, self.state_dim, self.action_dim, initializer=self.actor)
        # Critic can train on all transitions whereas the policy only trains on transitions that are at decision points
        self.critic_buffer = init_buffer(cfg.critic.buffer, app_state)
        self.policy_buffer = init_buffer(cfg.actor.buffer, app_state)

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

        recent_critic_idxs = self.critic_buffer.feed(pr.transitions, pr.data_mode)
        recent_policy_idxs = self.policy_buffer.feed([t for t in pr.transitions if t.prior.dp], pr.data_mode)

        # ---------------------------------- ingress loss metic --------------------------------- #
        if self.cfg.ingress_loss and len(recent_policy_idxs) > 0:
            recent_policy_batch = self.policy_buffer.get_batch(recent_policy_idxs)
            if len(recent_policy_batch):
                assert len(recent_policy_batch) == 1
                recent_policy_batch = recent_policy_batch[0]
                self._app_state.metrics.write(
                    agent_step=self._app_state.agent_step,
                    metric=f"ingress_policy_loss_{pr.data_mode.name}",
                    value=self._policy_err(
                        self.actor,
                        recent_policy_batch.prior.state,
                        recent_policy_batch.post.action),
                )

                self._app_state.metrics.write(
                    agent_step=self._app_state.agent_step,
                    metric=f"ingress_sampler_loss_{pr.data_mode.name}",
                    value=self._policy_err(
                        self.sampler,
                        recent_policy_batch.prior.state,
                        recent_policy_batch.post.action),
                )

        if self.cfg.ingress_loss and len(recent_critic_idxs) > 0:
            recent_critic_batch = self.critic_buffer.get_batch(recent_critic_idxs)
            if len(recent_critic_batch):
                self._app_state.metrics.write(
                    agent_step=self._app_state.agent_step,
                    metric=f"ingress_critic_loss_{pr.data_mode.name}",
                    value=self._compute_critic_loss(recent_critic_batch),
                )

        # ------------------------- transition length metric ------------------------- #

        for t in pr.transitions:
            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric="transition_len",
                value=len(t),
            )



    def load_buffer(self, pr: PipelineReturn) -> None:
        if pr.transitions is None:
            return

        policy_transitions = []
        for transition in pr.transitions:
            if transition.prior.dp:
                policy_transitions.append(transition)

        self.policy_buffer.load(policy_transitions, pr.data_mode)
        self.critic_buffer.load(pr.transitions, pr.data_mode)
        self.policy_buffer.app_state = self._app_state
        self.critic_buffer.app_state = self._app_state

    def _filter_only_direct_actions(self, actions: torch.Tensor):
        if not self.cfg.delta_action:
            return actions

        direct_idxs = [i for i, col in enumerate(self._col_desc.action_cols) if not Delta.is_delta_transformed(col)]
        return actions[:, direct_idxs]

    def _filter_only_delta_actions(self, actions: torch.Tensor):
        if not self.cfg.delta_action:
            return actions

        delta_idxs = [i for i, col in enumerate(self._col_desc.action_cols) if Delta.is_delta_transformed(col)]
        return actions[:, delta_idxs]

    def _ensure_direct_action(self, direct_action: torch.Tensor, next_action: torch.Tensor):
        if not self.cfg.delta_action:
            return next_action

        bounds = self.cfg.delta_bounds
        assert bounds is not None, "Delta actions are enabled, however the agent has no delta bounds"
        scale = bounds[1] - bounds[0]
        bias = bounds[0]

        delta = scale * next_action + bias
        direct_action = direct_action + delta

        # because we are always operating in normalized space,
        # we can hardcode the spatial constraints
        return torch.clip(direct_action, 0, 1)

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
        next_qs = []
        for batch in ensemble_batch:
            state_batch = batch.prior.state
            direct_action_batch = batch.post.action
            reward_batch = batch.n_step_reward
            next_state_batch = batch.post.state
            gamma_batch = batch.n_step_gamma
            dp_mask = batch.post.dp

            # put actions into direct form
            next_actions, _ = self.actor.get_action(next_state_batch, with_grad=False)
            direct_action_batch = self._filter_only_direct_actions(direct_action_batch)
            next_direct_actions = self._ensure_direct_action(direct_action_batch, next_actions)
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
        if self.ensemble_targets:
            # Option 1: Using the corresponding target function in the ensemble in the update target:
            _, next_qs = self.q_critic.get_qs_target(next_state_batches, next_action_batches, bootstrap_reduct=True)

        else:
            # Option 2: Using the reduction of the ensemble in the update target:
            next_qs = []
            for i in range(ensemble_len):
                next_q = self.q_critic.get_q_target(
                    [next_state_batches[i]], [next_action_batches[i]], bootstrap_reduct=True
                )
                next_q = torch.unsqueeze(next_q, 0)
                next_qs.append(next_q)

            next_qs = torch.cat(next_qs, dim=0)

        # Third, compute losses
        _, qs = self.q_critic.get_qs(state_batches, action_batches, with_grad=with_grad, bootstrap_reduct=True)
        loss = torch.tensor(0.0, device=device.device)
        for i in range(ensemble_len):
            target =  reward_batches[i] + gamma_batches[i] * next_qs[i]
            loss_i = torch.nn.functional.mse_loss(target, qs[i])
            loss += loss_i

            if log_metrics:
                self._app_state.metrics.write(
                    agent_step=self._app_state.agent_step,
                    metric=f"critic_loss_{i}",
                    value=to_np(loss_i),
                )

        if log_metrics:
            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric="avg_critic_loss",
                value=to_np(loss)/ensemble_len,
            )

        return loss

    def update_critic(self) -> list[float]:
        if min(self.critic_buffer.size) <= 0:
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
                    metric=f"critic_loss_{self.policy_buffer.n_most_recent}_most_recent",
                    value=n_most_recent_loss,
            )

        if self.eval_batch:
            eval_batches = self.critic_buffer.sample()
        else:
            eval_batches = batches

        self.q_critic.update(q_loss, opt_kwargs={"closure": partial(self._compute_critic_loss, eval_batches)})
        return [float(q_loss)]

    # --------------------------- actor and sampler updating-------------------------- #

    def _get_top_n_sampled_actions(
        self,
        state_batch: Float[torch.Tensor, "batch_size state_dim"],
        direct_action_batch: Float[torch.Tensor, "batch_size action_dim"],
        n_samples: int,
        percentile: float,
        uniform_weight: float,  # proportion of samples coming from uniform dist
        sampler: BaseActor,  # which network to sample non-uniform actions from,
    ) -> tuple[
        Float[torch.Tensor, "batch_size*top_actions state_dim"],
        Float[torch.Tensor, "batch_size*top_actions action_dim"],
        Float[torch.Tensor, "batch_size*n_samples action_dim"],
        Float[torch.Tensor, "batch_size*n_samples action_dim"],
        Float[torch.Tensor, "batch_size*n_samples action_dim"]
    ]:
        """
        Returns the top n_sampled actions for states within state_batch. The samples can either
        be in delta action space (if self.cfg.delta_action) or direct action space. Also returns
        the direct action samples regardless (for testing).
        """
        BATCH_DIM = 0
        SAMPLE_DIM = 1

        ACTION_DIM = direct_action_batch.size(1)
        STATE_DIM = state_batch.size(1)

        # FIRST, sample actions. These can be direct or delta, depending on self.cfg.delta_action
        sampled_actions: Float[torch.Tensor, "batch_size num_samples action_dim"]
        # states that each of the sampled actions correspond to:
        repeated_states: Float[torch.Tensor, "batch_size num_samples state_dim"]
        sampled_actions, repeated_states = sample_actions(
            state_batch, sampler, n_samples, self.action_dim, uniform_weight
        )

        batch_size = state_batch.size(BATCH_DIM)
        assert sampled_actions.dim() == repeated_states.dim()
        assert sampled_actions.size(BATCH_DIM) == repeated_states.size(BATCH_DIM) == batch_size
        assert sampled_actions.size(SAMPLE_DIM) == repeated_states.size(SAMPLE_DIM) == n_samples

        # the next few lines produce direct actions, which is what the critic takes in
        direct_action_batch = unsqueeze_repeat(direct_action_batch, SAMPLE_DIM, n_samples)
        assert direct_action_batch.size(BATCH_DIM) == batch_size
        assert direct_action_batch.size(SAMPLE_DIM) == n_samples
        assert direct_action_batch.size(-1) == ACTION_DIM

        # after this line, sampled sampled_direct_actions are guaranteed to be direct
        sampled_direct_actions = self._ensure_direct_action(direct_action_batch, sampled_actions)

        # NEXT we will query the critic for q_values
        # however, we first need to reshape the repeated_states and actions
        # to be the right shape to feed into the critic (i.e. 2 dimensional)
        repeated_states_2d = repeated_states.reshape(batch_size * n_samples, STATE_DIM)
        sampled_direct_actions_2d = sampled_direct_actions.reshape(batch_size * n_samples, ACTION_DIM)

        q_values: Float[torch.Tensor, "batch_size*num_samples 1"]
        q_values = self.q_critic.get_q([repeated_states_2d], [sampled_direct_actions_2d],
                                       with_grad=False, bootstrap_reduct=False)
        # now, we need to reshape the q_values to have size (batch_size, num_samples)
        q_values: Float[torch.Tensor, "batch_size num_samples"]
        q_values = q_values.reshape(batch_size, n_samples)

        # NEXT, we will grab the top percentile of direct_actions according to the q_values
        top_actions: Float[torch.Tensor, "batch_size top_n action_dim"]
        top_action_inds = grab_percentile(q_values, sampled_direct_actions, percentile)
        top_n = top_action_inds.size(SAMPLE_DIM)

        # grab the top actions. Can be direct OR delta
        top_actions = torch.gather(sampled_actions, dim=1, index=top_action_inds)

        # grab the top states for those actions
        states_for_best_actions = repeated_states[:, :top_n, :]

        # FINALLY, reshape returned states and actions to be two dimensional, since this is what
        # the loss function for the policy expects
        states_for_best_actions_2d = states_for_best_actions.reshape(batch_size * top_n, STATE_DIM)
        top_actions_2d = top_actions.reshape(batch_size * top_n, ACTION_DIM)
        sampled_actions_2d = sampled_actions.reshape(batch_size * n_samples, ACTION_DIM)

        # also return the sampled direct actions
        direct_top_actions = torch.gather(sampled_direct_actions, dim=1, index=top_action_inds)
        direct_top_actions_2d = direct_top_actions.reshape(batch_size * top_n, ACTION_DIM)
        direct_sampled_actions_2d = sampled_direct_actions.reshape(batch_size * n_samples, ACTION_DIM)

        return states_for_best_actions_2d,\
            top_actions_2d, sampled_actions_2d, \
            direct_top_actions_2d, direct_sampled_actions_2d


    def _compute_policy_loss(
            self,
            policy: BaseActor,
            batch: TransitionBatch,
            percentile: float,
            with_grad:bool=False
        ) -> torch.Tensor:
        state_batch = batch.prior.state
        direct_action_batch = self._filter_only_direct_actions(batch.post.action)

        # get the top percentile of actions
        states_for_best_actions, best_actions, _, _, _ = self._get_top_n_sampled_actions(
            state_batch=state_batch,
            direct_action_batch=direct_action_batch,
            n_samples=self.num_samples,
            percentile=percentile,
            uniform_weight=self.uniform_sampling_percentage,
            sampler=self.sampler,
        )
        loss = self._policy_err(policy, states_for_best_actions, best_actions, with_grad=with_grad)
        return loss


    def _update_policy(
        self,
        policy: BaseActor,
        policy_name: str,
        percentile: float,
        update_batch : TransitionBatch | None = None,
    ) -> TransitionBatch:

        assert policy_name == "actor" or policy_name == "sampler"
        update_batch = self._ensure_policy_batch(update_batch)
        loss = self._compute_policy_loss(policy, update_batch, percentile, with_grad=True)

        self._app_state.metrics.write(
            agent_step=self._app_state.agent_step,
            metric=policy_name + "_loss",
            value=to_np(loss),
        )

        log_most_recent_batch_loss = self.cfg.most_recent_batch_loss and self.policy_buffer.n_most_recent > 0
        if log_most_recent_batch_loss:
            # grab the most recent samples from the batch and log the loss on only these samples
            n_most_recent_loss = self._compute_policy_loss(
                policy,
                update_batch[:self.policy_buffer.n_most_recent],
                percentile,
            )

            self._app_state.metrics.write(
                    agent_step=self._app_state.agent_step,
                    metric=f"{policy_name}_loss_{self.policy_buffer.n_most_recent}_most_recent",
                    value=n_most_recent_loss,
            )

        if self.eval_batch:
            # sample another batch for the evaluation of updates when using line search.
            eval_batch = self._ensure_policy_batch()
        else:
            eval_batch = update_batch

        # apply the update
        policy.update(
            loss,
            opt_kwargs={
                "closure": partial(self._compute_policy_loss, policy, eval_batch, percentile),
            },
        )

        return update_batch

    def _ensure_policy_batch(self, update_batch: TransitionBatch | None = None) -> TransitionBatch:
        if update_batch is None:
            # Assuming we don't have an ensemble of policies
            batches = self.policy_buffer.sample()
            assert len(batches) == 1
            update_batch = batches[0]

        assert update_batch is not None
        return update_batch

    def update_actor(self) -> TransitionBatch | None:
        self._app_state.event_bus.emit_event(EventType.agent_update_actor)
        if min(self.policy_buffer.size) <= 0:
            return None

        batch = self._update_policy(self.actor, "actor", self.rho)
        return batch

    def update_sampler(self, update_batch: TransitionBatch | None = None) -> None:
        self._app_state.event_bus.emit_event(EventType.agent_update_sampler)
        if min(self.policy_buffer.size) <= 0:
            return None

        self._update_policy(self.sampler, "sampler", self.rho_proposal, update_batch)

    def _policy_err(
            self,
            policy: BaseActor,
            states: torch.Tensor,
            actions: torch.Tensor,
            with_grad:bool=False,
        ) -> torch.Tensor:
        logp, _ = policy.get_log_prob(
            states,
            actions,
            with_grad=with_grad,
        )
        return -logp.mean()

    def update(self) -> list[float]:
        q_losses = []
        for _ in range(self.n_actor_updates):
            for _ in range(self.n_critic_updates):
                q_loss = self.update_critic()
                q_losses += q_loss

            actor_update_return = self.update_actor()

            # signals to update_sampler to sample a batch anew
            if not self.share_batch:
                actor_update_return = None

            self.update_sampler(actor_update_return)

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

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Callable, Literal, NamedTuple

import torch
from pydantic import Field

from corerl.agent.utils import SampledQReturn, get_sampled_qs, grab_percentile, mix_uniform_actions
from corerl.component.buffer import MixedHistoryBuffer, MixedHistoryBufferConfig
from corerl.component.critic.ensemble_critic import EnsembleCritic
from corerl.component.network.utils import tensor, to_np
from corerl.component.optimizers.ensemble_optimizer import EnsembleOptimizer
from corerl.component.optimizers.factory import OptimizerConfig, init_optimizer
from corerl.component.optimizers.torch_opts import AdamConfig
from corerl.component.policy.factory import BaseNNConfig, SquashedGaussianPolicyConfig, create
from corerl.component.policy.policy import Policy
from corerl.configs.config import config
from corerl.data_pipeline.pipeline import PipelineReturn
from corerl.eval.agent import get_layers_stable_rank
from corerl.messages.events import EventType
from corerl.state import AppState
from corerl.utils.device import device

logger = logging.getLogger(__name__)

OUTPUT_MIN = 0
OUTPUT_MAX = 1

class ActionReturn(NamedTuple):
    # the direct actions to be taken in the env
    direct_actions: torch.Tensor
    # the actions emitted from the policy. in [0, 1].
    # i.e. they are either normalized delta or direct actions.
    policy_actions: torch.Tensor


Optimizer = torch.optim.Optimizer | EnsembleOptimizer

@config()
class GACPolicyManagerConfig:
    name: Literal["network"] = "network"
    delta_actions: bool = False
    delta_bounds: list[tuple[float, float]] = Field(default_factory=list)
    delta_rejection_sample: bool = True

    # hyperparameters
    num_samples: int = 128
    actor_percentile: float = 0.1
    sampler_percentile: float = 0.2
    uniform_weight: float = 1.0
    init_sampler_with_actor_weights: bool = True
    resample_for_sampler_update: bool = True

    # metrics
    ingress_loss: bool = True

    # components
    network: BaseNNConfig = Field(default_factory=SquashedGaussianPolicyConfig)
    optimizer: OptimizerConfig = Field(default_factory=AdamConfig)
    buffer: MixedHistoryBufferConfig = Field(
        default_factory=lambda: MixedHistoryBufferConfig(
            ensemble=1,
            ensemble_probability=1.0,
        ),
    )

class GACPolicyManager:
    def __init__(
        self,
        cfg: GACPolicyManagerConfig,
        app_state: AppState,
        state_dim: int,
        action_dim: int,
    ):
        self._app_state = app_state
        self.cfg = cfg

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.is_uniform_sampler = self.cfg.uniform_weight== 1.

        self.actor = create(
            cfg.network,
            state_dim,
            action_dim,
            OUTPUT_MIN,
            OUTPUT_MAX,
        )

        self.sampler = create(
            cfg.network,
            state_dim,
            action_dim,
            OUTPUT_MIN,
            OUTPUT_MAX,
        )

        if cfg.init_sampler_with_actor_weights:
            self.sampler.load_state_dict(self.actor.state_dict())

        self.buffer = MixedHistoryBuffer(cfg.buffer, app_state)

        self.optimizer_name = cfg.optimizer.name
        self.actor_optimizer = init_optimizer(cfg.optimizer, app_state, self.actor.parameters())
        self.sampler_optimizer = init_optimizer(cfg.optimizer, app_state, self.sampler.parameters())
        assert isinstance(self.actor_optimizer, Optimizer)
        assert isinstance(self.sampler_optimizer, Optimizer)

        if self.cfg.delta_actions:
            self.delta_low = tensor([db[0] for db in cfg.delta_bounds], device.device)
            self.delta_high = tensor([db[1] for db in cfg.delta_bounds], device.device)

            self.delta_scale = (self.delta_high - self.delta_low) / (OUTPUT_MAX - OUTPUT_MIN)
            self.delta_bias = self.delta_low

    @property
    def support(self):
        return self.actor.support

    # ---------------------------------------------------------------------------- #
    #                      Helper methods for sampling actions                     #
    # ---------------------------------------------------------------------------- #

    def _ensure_direct_action(
            self,
            prev_direct_actions: torch.Tensor,
            policy_actions: torch.Tensor
        ) -> torch.Tensor:
        """
        Ensures that the output of this function is a direct action
        """
        if self.cfg.delta_actions:
            delta_actions = policy_actions * self.delta_scale + self.delta_bias
            direct_actions = prev_direct_actions + delta_actions
        else:
            direct_actions = policy_actions
        return direct_actions

    def _sample_actor(self, states: torch.Tensor) -> torch.Tensor:
        """
        Samples actions from the actor
        """
        with torch.no_grad():
            policy_actions, _ = self.actor.forward(states)  # actions in [0, 1]
        return policy_actions

    def _sample_sampler(self, states: torch.Tensor) -> torch.Tensor:
        """
        Samples a mixture between the sampler and a uniform distribution
        """
        if self.is_uniform_sampler:
            policy_actions = torch.rand(states.size(0), self.action_dim, device=device.device)
            return policy_actions

        with torch.no_grad():
            policy_actions, _ = self.sampler.forward(states)  # actions in [0, 1]
        policy_actions, _ = mix_uniform_actions(policy_actions, self.cfg.uniform_weight)
        return policy_actions

    def _sample_unform(self, states: torch.Tensor):
        return torch.rand(states.size(0), self.action_dim, device=device.device)

    def _rejection_sample(
            self,
            sampler: Callable[[torch.Tensor], torch.Tensor],
            states: torch.Tensor,
            prev_direct_actions: torch.Tensor,
            direct_actions: torch.Tensor,
            policy_actions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform rejection sampling on actions to ensure they are within valid bounds

        This method checks if the provided `direct_actions` are within the valid range [0, 1]
        If any actions are out of bounds, it resamples them using a mixture policy until all actions are valid.
        """
        max_itr = 100
        for itr in range(max_itr):
            # check if in valid range for normalized direct actions
            if torch.all((direct_actions>=OUTPUT_MIN) & (direct_actions<=OUTPUT_MAX)):
                break
            # set up mask which rows were invalid
            invalid_mask = (direct_actions<OUTPUT_MIN) | (direct_actions>OUTPUT_MAX)
            invalid_mask = invalid_mask.any(dim=1)
            # resample invalid actions
            policy_actions[invalid_mask] = sampler(states[invalid_mask])
            direct_actions = self._ensure_direct_action(prev_direct_actions, policy_actions)

            if itr == max_itr-1:
                logging.warning(f"Maximum iterations ({max_itr}) in rejection sampling reached..."
                                 +"defaulting to sampling uniform")
                policy_actions[invalid_mask] = self._sample_unform(states[invalid_mask])
                direct_actions = self._ensure_direct_action(prev_direct_actions, policy_actions)

        # Clip the direct actions. This should be unnecessary if rejection sampling is successful
        direct_actions = torch.clip(direct_actions, OUTPUT_MIN, OUTPUT_MAX)
        return direct_actions, policy_actions

    def _get_actions(
        self,
        sampler: Callable[[torch.Tensor], torch.Tensor],
        states: torch.Tensor,
        prev_direct_actions: torch.Tensor,
    ) -> ActionReturn:
        policy_actions = sampler(states)
        direct_actions = self._ensure_direct_action(prev_direct_actions, policy_actions)

        if self.cfg.delta_actions and self.cfg.delta_rejection_sample:
            direct_actions, policy_actions = self._rejection_sample(
                sampler, states, prev_direct_actions, direct_actions, policy_actions
            )
        else:
            direct_actions = torch.clip(direct_actions, OUTPUT_MIN, OUTPUT_MAX)

        return ActionReturn(direct_actions, policy_actions)

    # ---------------------------------------------------------------------------- #
    #                                      API                                     #
    # ---------------------------------------------------------------------------- #

    def get_actor_actions(
        self,
        states: torch.Tensor,
        prev_direct_actions: torch.Tensor,
    ) -> ActionReturn:
        """
        Samples direct actions for states from the actor.
        """
        return self._get_actions(self._sample_actor, states, prev_direct_actions)

    def get_sampler_actions(
        self,
        states: torch.Tensor,
        prev_direct_actions: torch.Tensor,
    ) -> ActionReturn:
        """
        Samples direct actions for states.
        If uniform_weight is greater than 0, will sample actions from a mixture between the policy and uniform.
        """
        return self._get_actions(self._sample_sampler, states, prev_direct_actions)

    def get_uniform_actions(
        self,
        states: torch.Tensor,
        prev_direct_actions: torch.Tensor,
    ) -> ActionReturn:
        """
        Samples direct actions for states UAR
        """
        return self._get_actions(self._sample_unform, states, prev_direct_actions)

    def update_buffer(self, pr: PipelineReturn) -> None:
        """
        Adds transitions to the buffer.
        """
        if pr.transitions is None:
            return

        valid_transitions = [t for t in pr.transitions if t.prior.dp]
        recent_idxs = self.buffer.feed(valid_transitions, pr.data_mode)
        # ---------------------------------- ingress loss metic --------------------------------- #
        if self.cfg.ingress_loss and len(recent_idxs) > 0:
            recent_batch = self.buffer.get_batch(recent_idxs)

            if self.cfg.delta_actions:
                recent_actions = recent_batch.post.action - recent_batch.prior.action
            else:
                recent_actions = recent_batch.post.action

            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric=f"ingress_actor_loss_{pr.data_mode.name}",
                value=self._policy_err(self.actor, recent_batch.prior.state, recent_actions),
            )

            if not self.is_uniform_sampler:
                self._app_state.metrics.write(
                    agent_step=self._app_state.agent_step,
                    metric=f"ingress_sampler_loss_{pr.data_mode.name}",
                    value=self._policy_err(self.sampler, recent_batch.prior.state, recent_actions),
                )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        actor_net_path = path / "actor_net"
        torch.save(self.actor.state_dict(), actor_net_path)

        actor_opt_path = path / "actor_opt"
        torch.save(self.actor_optimizer.state_dict(), actor_opt_path)

        sampler_net_path = path / "sampler_net"
        torch.save(self.sampler.state_dict(), sampler_net_path)

        sampler_opt_path = path / "sampler_opt"
        torch.save(self.sampler_optimizer.state_dict(), sampler_opt_path)

        buffer_path = path / "buffer.pkl"
        with open(buffer_path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, path: Path) -> None:
        actor_net_path = path / "actor_net"
        self.actor.load_state_dict(torch.load(actor_net_path, map_location=device.device))

        actor_opt_path = path / "actor_opt"
        self.actor_optimizer.load_state_dict(torch.load(actor_opt_path, map_location=device.device))

        sampler_net_path = path / "sampler_net"
        self.sampler.load_state_dict(torch.load(sampler_net_path, map_location=device.device))

        sampler_opt_path = path / "sampler_opt"
        self.sampler_optimizer.load_state_dict(torch.load(sampler_opt_path, map_location=device.device))

        buffer_path = path / "buffer.pkl"
        with open(buffer_path, "rb") as f:
            self.buffer = pickle.load(f)

    def update(self, critic: EnsembleCritic) -> None:
        """
        Performs a percentile-based update to the policy.
        """
        self._app_state.event_bus.emit_event(EventType.agent_update_actor)
        if min(self.buffer.size) <= 0:
            return None

        # Assuming we don't have an ensemble of policies
        batches = self.buffer.sample()
        assert len(batches) == 1
        update_batch = batches[0]

        # sample self.cfg.num_samples according to sampler, then rank them by critic
        # top_states has size (batch_size, n, state_dim) and top_policy_actions has size
        # (batch_size, n, action_dim), where n = floor(self.percentile*batch_size)
        qr = get_sampled_qs(
            states=update_batch.prior.state,
            prev_actions=update_batch.prior.action,
            n_samples=self.cfg.num_samples,
            sampler=self.get_sampler_actions,
            critic=critic,
        )

        assert isinstance(self.actor_optimizer, Optimizer)
        actor_closure = self._get_closure(self.actor, critic, self.cfg.actor_percentile)
        self._regress_towards_percentile(qr, self.actor, self.actor_optimizer,
                                         self.cfg.actor_percentile , 'actor', actor_closure)
        if not self.is_uniform_sampler:
            if self.cfg.resample_for_sampler_update:
                qr = get_sampled_qs(
                    states=update_batch.prior.state,
                    prev_actions=update_batch.prior.action,
                    n_samples=self.cfg.num_samples,
                    sampler=self.get_sampler_actions,
                    critic=critic,
                )
            assert isinstance(self.sampler_optimizer, Optimizer)
            sampler_closure = self._get_closure(self.sampler, critic, self.cfg.sampler_percentile)
            self._regress_towards_percentile(qr, self.sampler,  self.sampler_optimizer,
                                            self.cfg.sampler_percentile, 'sampler', sampler_closure)

    # ---------------------------------------------------------------------------- #
    #                            updating helper methods                           #
    # ---------------------------------------------------------------------------- #

    def _get_closure(
            self,
            policy: Policy,
            critic: EnsembleCritic,
            percentile: float,
            ) -> Callable[[], float]:

        if self.optimizer_name != 'lso':
            return lambda: 0.

        batches = self.buffer.sample()
        assert len(batches) == 1
        batch = batches[0]

        sampler = self.get_sampler_actions
        qr = get_sampled_qs(
            states=batch.prior.state,
            prev_actions=batch.prior.action,
            n_samples=self.cfg.num_samples,
            sampler=sampler,
            critic=critic,
        )

        top_states, top_policy_actions = grab_percentile(
            values=qr.q_values,
            keys=[qr.states, qr.policy_actions],
            percentile=percentile,
        )

        def closure():
            loss = self._policy_err(
                policy,
                states=top_states.reshape(-1, self.state_dim),
                policy_actions=top_policy_actions.reshape(-1, self.action_dim),
            )
            return to_np(loss).max()

        return closure

    def _regress_towards_percentile(
            self,
            qr: SampledQReturn,
            policy: Policy,
            optimizer: Optimizer,
            percentile: float,
            metric_id: str,
            closure: Callable[[], float]
        ):
        top_states, top_policy_actions = grab_percentile(
            values=qr.q_values,
            keys=[qr.states, qr.policy_actions],
            percentile=percentile,
        )

        # reshape, then take loss w.r.t. these states and actions
        loss = self._policy_err(
            policy=policy,
            states=top_states.reshape(-1, self.state_dim),
            policy_actions=top_policy_actions.reshape(-1, self.action_dim),
            with_grad=True,
        )

        self._app_state.metrics.write(
            agent_step=self._app_state.agent_step,
            metric=metric_id + "_loss",
            value=to_np(loss),
        )

        # take a step with the optimizer
        optimizer.zero_grad()
        loss.backward()

        # log to metrics table
        log_policy_gradient_norm(self._app_state, policy, prefix=metric_id)
        log_policy_weight_norm(self._app_state, policy, prefix=metric_id)

        opt_args = tuple()
        opt_kwargs = {"closure": closure}
        optimizer.step(*opt_args, **opt_kwargs)

    def _policy_err(
        self,
        policy: Policy,
        states: torch.Tensor,
        policy_actions: torch.Tensor,
        with_grad: bool = False,
    ) -> torch.Tensor:
        with torch.set_grad_enabled(with_grad):
            logp, _ = policy.log_prob(
                states,
                policy_actions,
            )
        return -logp.mean()

# ---------------------------------------------------------------------------- #
#                                    Metrics                                   #
# ---------------------------------------------------------------------------- #

def log_policy_gradient_norm(app_state: AppState, policy: Policy, prefix: str):
    """
    Logs the gradient norm.
    """
    total_norm = 0
    for param in policy.parameters():
        if param.requires_grad and param.grad is not None:
            param_norm = param.grad.data.norm(2)

            total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5

    app_state.metrics.write(
            agent_step=app_state.agent_step,
            metric="optimizer_" + prefix + "_grad_norm",
            value=to_np(total_norm),
        )

def log_policy_weight_norm(app_state: AppState, policy: Policy, prefix: str):
    """
    Logs the weight norm.
    """
    with torch.no_grad():
        total_norm = 0
        for param in policy.parameters():
            param_norm = param.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5

        app_state.metrics.write(
                agent_step=app_state.agent_step,
                metric="optimizer_" + prefix + "_weight_norm",
                value=to_np(total_norm),
            )

def log_policy_stable_rank(app_state: AppState, policy: Policy, prefix: str):
    with torch.no_grad():
        stable_ranks = get_layers_stable_rank(policy.model)

        for i, rank in enumerate(stable_ranks):
            app_state.metrics.write(
                    agent_step=app_state.agent_step,
                    metric="optimizer_" + prefix + f"_stable_rank_layer_{i}",
                    value=rank,
            )

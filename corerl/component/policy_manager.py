from __future__ import annotations

import functools
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, NamedTuple

import torch
from pydantic import Field, TypeAdapter

from corerl.agent.utils import (
    SampledQReturn,
    ValueEstimator,
    get_sampled_qs,
    grab_percentile,
    grab_top_n,
)
from corerl.component.buffer import BufferConfig, MixedHistoryBufferConfig, RecencyBiasBufferConfig, buffer_group
from corerl.component.network.utils import to_np
from corerl.component.optimizers.ensemble_optimizer import EnsembleOptimizer
from corerl.component.optimizers.factory import OptimizerConfig, init_optimizer
from corerl.component.optimizers.torch_opts import AdamConfig
from corerl.component.policy.factory import NormalPolicyConfig, PolicyConfig, create
from corerl.component.policy.policy import Policy
from corerl.configs.config import MISSING, computed, config, post_processor
from corerl.data_pipeline.pipeline import PipelineReturn
from corerl.eval.torch import get_layers_stable_rank
from corerl.messages.events import EventType
from corerl.state import AppState
from corerl.utils.device import device
from corerl.utils.torch import clip_gradients

if TYPE_CHECKING:
    from corerl.config import MainConfig


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
    action_bounds: bool = MISSING
    greedy: bool = False

    # hyperparameters
    num_samples: int = 128
    actor_percentile: float = 0.1
    sampler_percentile: float = 0.2
    prop_percentile_learned: float = 0.9
    init_sampler_with_actor_weights: bool = True
    resample_for_sampler_update: bool = True
    grad_clip: float = 50_000
    sort_noise: float = 0.0

    # metrics
    ingress_loss: bool = True

    # components
    network: PolicyConfig = Field(default_factory=NormalPolicyConfig)
    optimizer: OptimizerConfig = Field(default_factory=AdamConfig)
    buffer: BufferConfig = MISSING

    @computed("action_bounds")
    @classmethod
    def _action_bounds(cls, cfg: "MainConfig"):
        return cfg.feature_flags.action_bounds

    @computed('buffer')
    @classmethod
    def _buffer(cls, cfg: 'MainConfig'):
        default_buffer_type = (
            RecencyBiasBufferConfig
            if cfg.feature_flags.recency_bias_buffer else
            MixedHistoryBufferConfig
        )

        ta = TypeAdapter(default_buffer_type)
        default_buffer = default_buffer_type(id='critic')
        default_buffer_dict = ta.dump_python(default_buffer, warnings=False)
        main_cfg: Any = cfg
        out = ta.validate_python(default_buffer_dict, context=main_cfg)
        return out

    @post_processor
    def _default_stepsize(self, cfg: 'MainConfig'):
        if isinstance(self.optimizer, AdamConfig):
            self.optimizer.lr = 0.001

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

        self.is_uniform_sampler = self.cfg.prop_percentile_learned == 0.
        self._uniform_weight = 1 - (self.cfg.prop_percentile_learned * self.cfg.actor_percentile)

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

        self.buffer = buffer_group.dispatch(cfg.buffer, app_state)

        self.optimizer_name = cfg.optimizer.name
        self.actor_optimizer = init_optimizer(cfg.optimizer, app_state, self.actor.parameters())
        self.sampler_optimizer = init_optimizer(cfg.optimizer, app_state, self.sampler.parameters())
        assert isinstance(self.actor_optimizer, Optimizer)
        assert isinstance(self.sampler_optimizer, Optimizer)


    @property
    def support(self):
        return self.actor.support

    # ---------------------------------------------------------------------------- #
    #                      Helper methods for sampling actions                     #
    # ---------------------------------------------------------------------------- #

    def ensure_direct_action(
            self,
            action_lo: torch.Tensor,
            action_hi: torch.Tensor,
            policy_actions: torch.Tensor
        ) -> torch.Tensor:
        """
        Ensures that the output of this function is a direct action
        """
        assert policy_actions.dim() == 3, 'Expected policy_actions to be (batch_size, n_samples, action_dim)'
        assert action_lo.dim() == 2, 'Expected action_lo to be (batch_size, action_dim)'

        if self.cfg.action_bounds:
            # add an n_samples dim to control automatic broadcasting
            action_lo = action_lo.unsqueeze(1)
            action_hi = action_hi.unsqueeze(1)
            direct_actions = torch.clip(policy_actions, min=action_lo, max=action_hi)
        else:
            direct_actions = policy_actions
        return direct_actions

    def _sample_actor(
        self,
        states: torch.Tensor,
        action_lo: torch.Tensor,
        action_hi: torch.Tensor,
        n_samples: int
    ) -> torch.Tensor:
        """
        Samples actions from the actor
        """
        batch_size = states.size(0)

        with torch.no_grad():
            dist, _ = self.actor.get_dist(states)  # actions normalized with respect to the operating range

        policy_actions = dist.sample((n_samples,))
        assert policy_actions.shape == (n_samples, batch_size, self.action_dim)

        policy_actions = policy_actions.permute(1, 0, 2)
        return policy_actions

    def _sample_sampler(
        self,
        states: torch.Tensor,
        action_lo: torch.Tensor,
        action_hi: torch.Tensor,
        n_samples: int
    ) -> torch.Tensor:
        """
        Samples a mixture between the sampler and a uniform distribution
        """
        batch_size = states.size(0)

        if self.is_uniform_sampler:
            return self._sample_uniform(batch_size, n_samples, action_lo, action_hi)

        with torch.no_grad():
            dist, _ = self.sampler.get_dist(states)

        uniform_samples = int(self._uniform_weight * n_samples)
        learned_samples = n_samples - uniform_samples
        policy_actions = dist.sample((learned_samples,))
        assert policy_actions.shape == (learned_samples, batch_size, self.action_dim)

        policy_actions = policy_actions.permute(1, 0, 2)
        rand_actions = self._sample_uniform(batch_size, uniform_samples, action_lo, action_hi)
        out = torch.concatenate([policy_actions, rand_actions], dim=1)
        assert out.shape == (batch_size, n_samples, self.action_dim)
        return out

    def _sample_uniform(self, batch_size: int, n_samples: int, action_lo: torch.Tensor, action_hi: torch.Tensor):
        uniform_actions = torch.rand(batch_size, n_samples, self.action_dim, device=device.device)
        action_lo = action_lo.unsqueeze(1)
        action_hi = action_hi.unsqueeze(1)
        bounded_uniform_actions = (action_hi - action_lo) * uniform_actions + action_lo

        return bounded_uniform_actions

    def _get_actions(
        self,
        sampler: Callable[[torch.Tensor], torch.Tensor],
        states: torch.Tensor,
        action_lo: torch.Tensor,
        action_hi: torch.Tensor,
    ) -> ActionReturn:
        policy_actions = sampler(states)
        assert policy_actions.dim() == 3
        assert policy_actions.size(0) == states.size(0)

        direct_actions = self.ensure_direct_action(action_lo, action_hi, policy_actions)
        direct_actions = torch.clip(direct_actions, OUTPUT_MIN, OUTPUT_MAX)

        return ActionReturn(direct_actions, policy_actions)

    # ---------------------------------------------------------------------------- #
    #                                      API                                     #
    # ---------------------------------------------------------------------------- #

    def _sample_greedy(
        self,
        states: torch.Tensor,
        action_lo: torch.Tensor,
        action_hi: torch.Tensor,
        critic: ValueEstimator,
    ) -> torch.Tensor:
        qr = get_sampled_qs(
            states=states,
            action_lo=action_lo,
            action_hi=action_hi,
            n_samples=self.cfg.num_samples,
            sampler=self.get_sampler_actions,
            critic=critic,
        )

        _, policy_actions = grab_top_n(values=qr.q_values, keys=[qr.states, qr.policy_actions], n=1)
        assert policy_actions.shape == (states.size(0), 1, self.action_dim)
        return policy_actions

    def get_greedy_actions(
        self,
        states: torch.Tensor,
        action_lo: torch.Tensor,
        action_hi: torch.Tensor,
        critic: ValueEstimator,
    ) -> ActionReturn:
        """
        For each state, performs random search (with samples from proposal)
        in action space of approximate action-value function (critic)
        """
        sampler = functools.partial(
            self._sample_greedy,
            action_lo=action_lo,
            action_hi=action_hi,
            critic=critic,
        )
        policy_actions = sampler(states)
        direct_actions = self.ensure_direct_action(action_lo, action_hi, policy_actions)
        direct_actions = torch.clip(direct_actions, OUTPUT_MIN, OUTPUT_MAX)

        return ActionReturn(direct_actions, policy_actions)

    def get_actor_actions(
        self,
        n_samples: int,
        states: torch.Tensor,
        action_lo: torch.Tensor,
        action_hi: torch.Tensor,
        critic: ValueEstimator | None = None,
    ) -> ActionReturn:
        """
        Samples direct actions for states from the actor.
        """
        if not self.cfg.greedy:
            return self._get_actions(
                lambda x: self._sample_actor(x, action_lo, action_hi, n_samples),
                states,
                action_lo,
                action_hi,
            )

        assert critic is not None
        return self.get_greedy_actions(states, action_lo, action_hi, critic)

    def get_sampler_actions(
        self,
        n_samples: int,
        states: torch.Tensor,
        action_lo: torch.Tensor,
        action_hi: torch.Tensor,
    ) -> ActionReturn:
        """
        Samples direct actions for states.
        If uniform_weight is greater than 0, will sample actions from a mixture between the policy and uniform.
        """
        return self._get_actions(
            lambda x: self._sample_sampler(x, action_lo, action_hi, n_samples),
            states,
            action_lo,
            action_hi,
        )

    def get_uniform_actions(
        self,
        n_samples: int,
        states: torch.Tensor,
        action_lo: torch.Tensor,
        action_hi: torch.Tensor,
    ) -> ActionReturn:
        """
        Samples direct actions for states UAR
        """
        return self._get_actions(
            lambda x: self._sample_uniform(x.size(0), n_samples, action_lo, action_hi),
            states,
            action_lo,
            action_hi,
        )

    def update_buffer(self, pr: PipelineReturn) -> None:
        """
        Adds transitions to the buffer.
        """
        if pr.transitions is None:
            return

        recent_idxs = self.buffer.feed(pr.transitions, pr.data_mode)
        # ---------------------------------- ingress loss metic --------------------------------- #
        if self.cfg.ingress_loss and len(recent_idxs) > 0:
            recent_batch = self.buffer.get_batch(recent_idxs)
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
        try:
            actor_net_path = path / "actor_net"
            self.actor.load_state_dict(torch.load(actor_net_path, map_location=device.device))
            actor_opt_path = path / "actor_opt"
            self.actor_optimizer.load_state_dict(torch.load(actor_opt_path, map_location=device.device))
        except Exception:
            logger.exception('Failed to load actor state from checkpoint. Reinitializing...')
            self.actor = create(
                self.cfg.network,
                self.state_dim,
                self.action_dim,
                OUTPUT_MIN,
                OUTPUT_MAX,
            )

        try:
            sampler_net_path = path / "sampler_net"
            self.sampler.load_state_dict(torch.load(sampler_net_path, map_location=device.device))
            sampler_opt_path = path / "sampler_opt"
            self.sampler_optimizer.load_state_dict(torch.load(sampler_opt_path, map_location=device.device))
        except Exception:
            logger.exception('Failed to load sampler state from checkpoint. Reinitializing...')
            self.sampler = create(
                self.cfg.network,
                self.state_dim,
                self.action_dim,
                OUTPUT_MIN,
                OUTPUT_MAX,
            )

        try:
            buffer_path = path / "buffer.pkl"
            with open(buffer_path, "rb") as f:
                self.buffer = pickle.load(f)
        except Exception:
            logger.exception('Failed to load buffer from checkpoint. Reinitializing...')


    def update(self, critic: ValueEstimator):
        """
        Performs a percentile-based update to the policy.
        """
        self._app_state.event_bus.emit_event(EventType.agent_update_actor)
        if min(self.buffer.size) <= 0:
            return 0

        # Assuming we don't have an ensemble of policies
        batches = self.buffer.sample()
        assert len(batches) == 1
        update_batch = batches[0]

        # sample self.cfg.num_samples according to sampler, then rank them by critic
        # top_states has size (batch_size, n, state_dim) and top_policy_actions has size
        # (batch_size, n, action_dim), where n = floor(self.percentile*batch_size)
        qr = get_sampled_qs(
            states=update_batch.prior.state,
            action_lo=update_batch.prior.action_lo,
            action_hi=update_batch.prior.action_hi,
            n_samples=self.cfg.num_samples,
            sampler=self.get_sampler_actions,
            critic=critic,
        )

        assert isinstance(self.actor_optimizer, Optimizer)
        actor_loss = self._regress_towards_percentile(
            qr,
            self.actor,
            self.actor_optimizer,
            self.cfg.actor_percentile ,
            'actor',
        )
        if not self.is_uniform_sampler:
            if self.cfg.resample_for_sampler_update:
                qr = get_sampled_qs(
                    states=update_batch.prior.state,
                    action_lo=update_batch.prior.action_lo,
                    action_hi=update_batch.prior.action_hi,
                    n_samples=self.cfg.num_samples,
                    sampler=self.get_sampler_actions,
                    critic=critic,
                )
            assert isinstance(self.sampler_optimizer, Optimizer)
            self._regress_towards_percentile(qr, self.sampler,  self.sampler_optimizer,
                                            self.cfg.sampler_percentile, 'sampler')

        return actor_loss

    # ---------------------------------------------------------------------------- #
    #                            updating helper methods                           #
    # ---------------------------------------------------------------------------- #

    def _regress_towards_percentile(
            self,
            qr: SampledQReturn,
            policy: Policy,
            optimizer: Optimizer,
            percentile: float,
            metric_id: str,
        ):

        sort_noise = 0
        if self.cfg.sort_noise > 0:
            sort_noise = torch.normal(0, self.cfg.sort_noise, size=qr.q_values.shape)

        top_states, top_policy_actions = grab_percentile(
            values=qr.q_values + sort_noise,
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

        grad_clip_delta = clip_gradients(policy.model, self.cfg.grad_clip)
        self._app_state.metrics.write(
            agent_step=self._app_state.agent_step,
            metric=metric_id + "_max_grad_clip",
            value=grad_clip_delta,
        )

        # log to metrics table
        log_policy_gradient_norm(self._app_state, policy, prefix=metric_id)
        log_policy_weight_norm(self._app_state, policy, prefix=metric_id)
        log_policy_stable_rank(self._app_state, policy, prefix=metric_id)

        optimizer.step()

        return float(loss.item())

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
                metric="network_" + prefix + "_weight_norm",
                value=to_np(total_norm),
            )

def log_policy_stable_rank(app_state: AppState, policy: Policy, prefix: str):
    with torch.no_grad():
        stable_ranks = get_layers_stable_rank(policy.model)

        for i, rank in enumerate(stable_ranks):
            app_state.metrics.write(
                    agent_step=app_state.agent_step,
                    metric="network_" + prefix + f"_stable_rank_layer_{i}",
                    value=rank,
            )

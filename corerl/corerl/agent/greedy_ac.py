import logging
import pickle as pkl
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import chex
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u
import numpy as np
from lib_agent.actor.percentile_actor import PAConfig, PercentileActor
from lib_agent.buffer.buffer import State
from lib_agent.critic.qrc_critic import QRCConfig, QRCCritic, get_stable_rank
from pydantic import Field, TypeAdapter

from corerl.agent.base import BaseAgent, BaseAgentConfig
from corerl.component.buffer import (
    BufferConfig,
    JaxTransition,
    MixedHistoryBufferConfig,
    RecencyBiasBufferConfig,
    buffer_group,
)
from corerl.configs.config import MISSING, computed, config
from corerl.data_pipeline.datatypes import AbsTransition
from corerl.data_pipeline.pipeline import ColumnDescriptions, PipelineReturn
from corerl.messages.events import EventType
from corerl.state import AppState
from corerl.utils.math import exp_moving_avg

if TYPE_CHECKING:
    from corerl.config import MainConfig

logger = logging.getLogger(__name__)

@config()
class CriticNetworkConfig:
    ensemble: int = 1

@config()
class GTDCriticConfig:
    action_regularization: float = 0.0
    action_regularization_epsilon: float = 0.1
    buffer: BufferConfig = MISSING
    stepsize: float = 0.0001
    critic_network: CriticNetworkConfig = Field(default_factory=CriticNetworkConfig)

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
        return ta.validate_python(default_buffer_dict, context=main_cfg)

@config()
class PercentileActorConfig:
    num_samples: int = 128
    actor_percentile: float = 0.05
    sampler_percentile: float = 0.2
    prop_percentile_learned: float = 0.9
    sort_noise: float = 0.0
    actor_stepsize: float = 0.0001
    sampler_stepsize: float = 0.0001

    # components
    buffer: BufferConfig = MISSING

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
        return ta.validate_python(default_buffer_dict, context=main_cfg)


class EnsembleNetworkReturn(NamedTuple):
    # some reduction over ensemble members, producing a single
    # value function
    reduced_value: jax.Array

    # the value function for every member of the ensemble
    ensemble_values: jax.Array

    # the variance of the ensemble values
    ensemble_variance: jax.Array


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

    critic: GTDCriticConfig = Field(default_factory=GTDCriticConfig)
    policy: PercentileActorConfig = Field(default_factory=PercentileActorConfig)

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

    max_internal_actor_updates: int = 3
    """
    Number of actor updates per critic update. Early stopping is done
    using the loss_threshold. A minimum of 1 update will always be performed.
    """

    max_critic_updates: int = 10
    """
    Number of critic updates. Early stopping is done using the loss_threshold.
    A minimum of 1 update will always be performed.
    """

    bootstrap_action_samples: int = 10
    """
    Number of action samples to use for bootstrapping,
    producing an Expected Sarsa-like update.
    """

    max_action_stddev: float = 3
    """
    Maximum number of stddevs from the mean for the action
    taken during an interaction step. Forcefully prevents
    very long-tailed events from occurring.
    """


class GreedyAC(BaseAgent):
    def __init__(self, cfg: GreedyACConfig, app_state: AppState, col_desc: ColumnDescriptions):
        super().__init__(cfg, app_state, col_desc)
        self.cfg = cfg
        self._col_desc = col_desc

        actor_cfg = PAConfig(
            name='percentile',
            num_samples=cfg.policy.num_samples,
            actor_percentile=cfg.policy.actor_percentile,
            proposal_percentile=cfg.policy.sampler_percentile,
            uniform_weight=1-cfg.policy.prop_percentile_learned*cfg.policy.sampler_percentile,
            actor_lr=cfg.policy.actor_stepsize,
            proposal_lr=cfg.policy.sampler_stepsize,
            max_action_stddev=cfg.max_action_stddev,
            sort_noise=cfg.policy.sort_noise,
        )

        self._actor = PercentileActor(
            actor_cfg,
            app_state.cfg.seed,
            col_desc.state_dim,
            col_desc.action_dim,
        )

        critic_cfg = QRCConfig(
            name='qrc',
            stepsize=cfg.critic.stepsize,
            ensemble=cfg.critic.critic_network.ensemble,
            ensemble_prob=cfg.critic.buffer.ensemble_probability,
            num_rand_actions=cfg.bootstrap_action_samples,
            action_regularization=cfg.critic.action_regularization,
            action_regularization_epsilon=cfg.critic.action_regularization_epsilon,
            l2_regularization=1.0,
        )

        self.critic = QRCCritic(
            critic_cfg,
            app_state.cfg.seed,
            col_desc.state_dim,
            col_desc.action_dim,
        )

        self._actor_buffer = buffer_group.dispatch(cfg.policy.buffer, app_state)
        self.critic_buffer = buffer_group.dispatch(cfg.critic.buffer, app_state)

        self.ensemble = self.cfg.critic.buffer.ensemble

        # for early stopping
        self._last_critic_loss = 0.
        self._avg_critic_delta: float | None = None
        self._last_actor_loss = 0.
        self._avg_actor_delta: float | None = None

        # actor and critic state
        dummy_x = jnp.zeros(self.state_dim)
        dummy_a = jnp.zeros(self.action_dim)

        self._jax_rng = jax.random.PRNGKey(0)
        self._jax_rng, c_rng = jax.random.split(self._jax_rng)
        self._critic_state = self.critic.init_state(c_rng, dummy_x, dummy_a)
        self._jax_rng, a_rng = jax.random.split(self._jax_rng)
        self._actor_state = self._actor.init_state(a_rng, dummy_x)

        self._has_preinitialized = False
        self._nominal_setpoints = jnp.array([
            cfg.nominal_setpoint
            if cfg.nominal_setpoint is not None else 0.5
            for cfg in self._col_desc.action_tags
        ])


    @property
    def actor_percentile(self) -> float:
        return self.cfg.policy.actor_percentile

    @property
    def is_policy_buffer_sampleable(self)-> bool:
        return self._actor_buffer.is_sampleable

    def sample_policy_buffer(self) -> JaxTransition:
        return self._actor_buffer.sample()


    def get_dist(self, states: jax.Array):
        dummy_jaxtions = jnp.zeros(self.action_dim)
        state_ = State(
            states,
            a_lo=dummy_jaxtions,
            a_hi=dummy_jaxtions,
            dp=jnp.ones((states.shape[0], 1), dtype=jnp.bool),
            last_a=dummy_jaxtions,
        )
        return self._actor.get_dist(
            self._actor_state.actor.params,
            state_,
        )

    def prob(self, states: jax.Array, actions: jax.Array) -> jax.Array:
        state_ = State(
            states,
            a_lo=actions,
            a_hi=actions,
            dp=jnp.ones_like(states),
            last_a=actions,
        )

        return self._actor.get_probs(
            self._actor_state.actor.params,
            state_,
            actions,
        )

    def get_probs(self, actor_params: chex.ArrayTree, state: State, actions: jax.Array | np.ndarray):
        actions = jnp.asarray(actions)
        return self._actor.get_probs(actor_params, state, actions)

    def get_values(self, state_batches: jax.Array, action_batches: jax.Array):
        chex.assert_shape(state_batches, (self.ensemble, None, self.state_dim))

        q = self.critic.get_values(
            self._critic_state.params,
            state_batches,
            action_batches,
        )
        return EnsembleNetworkReturn(
            reduced_value=q.mean(axis=0),
            ensemble_values=q,
            ensemble_variance=q.var(axis=0),
        )

    def get_actions(self, state: State):
        actions, _ = self._actor.get_actions(self._actor_state.actor.params, state)

        # remove the n_samples dimension
        return actions.squeeze(axis=-2)

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

        state_features = jnp.asarray(state)
        jaxtion_lo = jnp.asarray(action_lo)
        jaxtion_hi = jnp.asarray(action_hi)
        state_ = State(
            features=state_features,
            a_lo=jaxtion_lo,
            a_hi=jaxtion_hi,
            dp=jnp.ones((1,)),
            last_a=jnp.zeros_like(jaxtion_lo),
        )
        jaxtion, metrics = self._actor.get_actions(
            self._actor_state.actor.params,
            state_,
        )
        # remove the n_samples dimension
        jaxtion = jaxtion.squeeze(axis=-2)

        self._app_state.metrics.write(
            agent_step=self._app_state.agent_step,
            metric="actor_var",
            value=metrics['actor_var'].mean().item(),
        )

        return np.asarray(jaxtion)

    def update_buffer(self, pr: PipelineReturn) -> None:
        if pr.transitions is None:
            return

        self._app_state.event_bus.emit_event(EventType.agent_update_buffer)
        self.critic_buffer.feed(pr.transitions, pr.data_mode)
        recent_actor_idxs = self._actor_buffer.feed(pr.transitions, pr.data_mode)

        # ---------------------------------- ingress actor loss metic --------------------------------- #
        if len(recent_actor_idxs) > 0:
            recent_actor_batch = self._actor_buffer.get_batch(recent_actor_idxs)

            state = State(
                features=jnp.asarray(recent_actor_batch.state),
                a_lo=jnp.asarray(recent_actor_batch.action_lo),
                a_hi=jnp.asarray(recent_actor_batch.action_hi),
                dp=jnp.ones((len(recent_actor_batch.state), 1)),
                last_a=jnp.asarray(recent_actor_batch.action),
            )

            actor_loss = -jnp.log(self._actor.get_probs(
                self._actor_state.actor.params,
                state,
                recent_actor_batch.action,
            )).item()

            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric=f"ingress_actor_loss_{pr.data_mode.name}",
                value=actor_loss,
            )

        # ------------------------- transition length metric ------------------------- #

        for t in pr.transitions:
            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric="pipeline_transition_len",
                value=len(t),
            )


    # --------------------------- critic updating-------------------------- #

    def update_critic(self) -> list[float]:
        if not self.critic_buffer.is_sampleable:
            return [0 for _ in range(self.ensemble)]

        batches = self.critic_buffer.sample()
        critic_batch = abs_transition_from_batch(batches)
        next_actions, _ = self._actor.get_actions(
            self._actor_state.actor.params,
            critic_batch.next_state,
            self.cfg.bootstrap_action_samples,
        )

        self._critic_state, metrics = self.critic.update(
            critic_state=self._critic_state,
            transitions=critic_batch,
            next_actions=next_actions,
        )

        # log critic metrics
        for metric_name, ens_metric in metrics._asdict().items():
            for i, metric_val in enumerate(ens_metric):
                assert isinstance(metric_val, jax.Array)
                metric_val = metric_val.mean().squeeze()
                self._app_state.metrics.write(
                    agent_step=self._app_state.agent_step,
                    metric=f"CRITIC{i}-{metric_name}",
                    value=metric_val,
                )

        # log stable ranks
        stable_ranks = get_stable_rank(self._critic_state.params)
        for i, rank in enumerate(stable_ranks):
            for layer_name, layer_rank in rank.items():
                self._app_state.metrics.write(
                    agent_step=self._app_state.agent_step,
                    metric=f"CRITIC{i}-stable_rank_{layer_name}",
                    value=float(layer_rank),
                )

        self._app_state.metrics.write(
            agent_step=self._app_state.agent_step,
            metric="CRITIC-avg_loss",
            value=metrics.loss.mean(),
        )

        return [loss.mean() for loss in metrics.loss]

    def ensemble_ve(self, params: chex.ArrayTree, x: jax.Array, a: jax.Array):
        ens_forward = jax_u.vmap_only(self.critic.forward, ['params'])
        qs = ens_forward(params, x, a)
        values = qs.mean(axis=0).squeeze(-1)

        chex.assert_rank(values, 0)
        return values

    def update_actor(self):
        if not self._actor_buffer.is_sampleable:
            return 0.

        batch = self._actor_buffer.sample()
        actor_batch = abs_transition_from_batch(batch)
        self._actor_state, metrics = self._actor.update(
            self._actor_state,
            self.ensemble_ve,
            self._critic_state.params,
            actor_batch,
        )
        return metrics.actor_loss.mean().item()

    def update(self) -> list[float]:
        if not self._has_preinitialized and self._app_state.cfg.feature_flags.nominal_setpoint_bias:
            self._has_preinitialized = True
            self._critic_state = self.critic.initialize_to_nominal_action(
                self._jax_rng,
                self._critic_state,
                self._nominal_setpoints,
            )

            actor_state = self._actor.initialize_to_nominal_action(
                self._jax_rng,
                self._actor_state.actor,
                self._nominal_setpoints,
                self.state_dim,
            )
            self._actor_state = self._actor_state._replace(actor=actor_state)


        q_losses = []

        alpha = self.cfg.loss_ema_factor
        n_updates = 0
        for _ in range(self.cfg.max_critic_updates):
            losses = self.update_critic()
            q_losses += losses
            avg_critic_loss = np.mean(losses)
            n_updates += 1

            for _ in range(self.cfg.max_internal_actor_updates):
                actor_loss = self.update_actor()
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
            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric="CRITIC-avg_loss_delta",
                value=self._avg_critic_delta,
            )

            if np.abs(self._avg_critic_delta) < self.cfg.loss_threshold:
                break

        self._app_state.metrics.write(
            agent_step=self._app_state.agent_step,
            metric="CRITIC-updates",
            value=n_updates,
        )
        return q_losses

    # ---------------------------- saving and loading ---------------------------- #

    def save(self, path: Path) -> None:
        self._app_state.event_bus.emit_event(EventType.agent_save)

        path.mkdir(parents=True, exist_ok=True)

        with open(path / 'actor.pkl', "wb") as f:
            pkl.dump(self._actor_state, f)

        with open(path / "actor_buffer.pkl", "wb") as f:
            pkl.dump(self._actor_buffer, f)

        with open(path / 'critic.pkl', "wb") as f:
            pkl.dump(self._critic_state, f)

        with open(path / "critic_buffer.pkl", "wb") as f:
            pkl.dump(self.critic_buffer, f)


    def load(self, path: Path) -> None:
        self._app_state.event_bus.emit_event(EventType.agent_load)

        actor_path = path / "actor.pkl"
        with open(actor_path, "rb") as f:
            self._actor_state = pkl.load(f)

        actor_buffer_path = path / "actor_buffer.pkl"
        with open(actor_buffer_path, "rb") as f:
            self._actor_buffer = pkl.load(f)
        self._actor_buffer.app_state = self._app_state

        critic_path = path / "critic.pkl"
        with open(critic_path, "rb") as f:
            self._critic_state = pkl.load(f)

        critic_buffer_path = path / "critic_buffer.pkl"
        with open(critic_buffer_path, "rb") as f:
            self.critic_buffer = pkl.load(f)
        self.critic_buffer.app_state = self._app_state

    def get_buffer_sizes(self) -> dict[str, list[int]]:
        return {
            "critic": self.critic_buffer.size,
            "policy": self._actor_buffer.size,
        }


def abs_transition_from_batch(batch: JaxTransition) -> AbsTransition:
    """
    Converts a JaxTransition batch to a State object, using the absolute state.
    """
    return AbsTransition(
        state=State(
            features=batch.state,
            a_lo=batch.action_lo,
            a_hi=batch.action_hi,
            dp=jnp.expand_dims(batch.dp, -1),
            last_a=batch.last_action,
        ),
        next_state=State(
            features=batch.next_state,
            a_lo=batch.next_action_lo,
            a_hi=batch.next_action_hi,
            dp=jnp.expand_dims(batch.next_dp, -1),
            last_a=batch.action,
        ),
        action=batch.action,
        reward=batch.reward,
        gamma=batch.gamma,
    )

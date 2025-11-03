import logging
import pickle as pkl
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import chex
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u
import numpy as np
from lib_agent.actor.percentile_actor import PAConfig, PercentileActor
from lib_agent.buffer.datatypes import State, Transition, convert_trajectory_to_transition
from lib_agent.buffer.factory import build_buffer
from lib_agent.critic.critic_utils import (
    create_ensemble_dict,
    extract_metrics,
    get_stable_rank,
)
from lib_agent.critic.qrc_critic import QRCConfig, QRCCritic
from lib_defs.config_defs.tag_config import TagType
from lib_utils.named_array import NamedArray

from corerl.agent.base import BaseAgent
from corerl.configs.agent.greedy_ac import (
    GreedyACConfig,
)
from corerl.data_pipeline.pipeline import ColumnDescriptions, PipelineReturn
from corerl.messages.events import RLEventType
from corerl.state import AppState
from corerl.utils.math import exp_moving_avg

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EnsembleNetworkReturn(NamedTuple):
    # some reduction over ensemble members, producing a single
    # value function
    reduced_value: jax.Array

    # the value function for every member of the ensemble
    ensemble_values: jax.Array

    # the variance of the ensemble values
    ensemble_variance: jax.Array


class GreedyAC(BaseAgent):
    def __init__(self, cfg: GreedyACConfig, app_state: AppState, col_desc: ColumnDescriptions):
        super().__init__(cfg, app_state, col_desc)
        self.cfg = cfg
        self._col_desc = col_desc

        actor_cfg = PAConfig(
            name='percentile',
            num_samples=cfg.policy.num_samples,
            actor_percentile=cfg.policy.actor_percentile,
            proposal_percentile=cfg.policy.proposal_percentile,
            uniform_weight=1-cfg.policy.prop_percentile_learned*cfg.policy.proposal_percentile,
            actor_lr=cfg.policy.actor_stepsize,
            proposal_lr=cfg.policy.sampler_stepsize,
            mu_multiplier=cfg.policy.mu_multiplier,
            sigma_multiplier=cfg.policy.sigma_multiplier,
            max_action_stddev=cfg.max_action_stddev,
            sort_noise=cfg.policy.sort_noise,
        )

        self._actor = PercentileActor(
            actor_cfg,
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
            use_all_layer_norm=app_state.cfg.feature_flags.all_layer_norm,
            rolling_reset_config=cfg.critic.rolling_reset_config,
            polyak_tau=cfg.critic.polyak_tau,
            weight_decay=cfg.weight_decay,
            return_scale=cfg.return_scale,
        )

        self.critic = QRCCritic(
            critic_cfg,
            col_desc.state_dim,
            col_desc.action_dim,
        )

        self._actor_buffer = build_buffer(cfg.policy.buffer.to_lib_config(), Transition)

        critic_buffer_config = cfg.critic.buffer.to_lib_config()
        critic_buffer_config.ensemble = cfg.critic.critic_network.ensemble
        if critic_buffer_config.name == 'recency_bias_buffer':
            ensemble_size = cfg.critic.critic_network.ensemble
            critic_buffer_config.gamma = (
                list(critic_buffer_config.gamma)[:1] * ensemble_size
            )
            critic_buffer_config.effective_episodes = (
                list(critic_buffer_config.effective_episodes)[:1] * ensemble_size
            )

        self.critic_buffer = build_buffer(critic_buffer_config, Transition)

        self.ensemble = len(self.critic._reset_manager.active_indices)

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
            if cfg.type == TagType.ai_setpoint
        ])


    @property
    def actor_percentile(self) -> float:
        return self.cfg.policy.actor_percentile

    @property
    def is_policy_buffer_sampleable(self)-> bool:
        return self._actor_buffer.is_sampleable

    def sample_policy_buffer(self) -> Transition:
        return self._actor_buffer.sample()


    def get_dist(self, states: NamedArray):
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

    def prob(self, states: NamedArray, actions: jax.Array) -> jax.Array:
        state_ = State(
            states,
            a_lo=actions,
            a_hi=actions,
            dp=jnp.ones_like(states.array),
            last_a=actions,
        )

        return self._actor.get_probs(
            self._actor_state.actor.params,
            state_,
            actions,
        )

    def get_active_values(self, state: NamedArray, action: jax.Array):
        """
        returns `EnsembleNetworkReturn` with state-action value estimates from ensemble of critics for:
            1 state, and
            1 action OR a batch of actions
        """
        chex.assert_shape(state, (self.state_dim,))
        self._jax_rng, rng = jax.random.split(self._jax_rng)
        assert action.ndim in {1, 2}

        # use active critic values for decision making
        qs = self.critic.forward(self._critic_state.params, rng, state.array, action).q

        return EnsembleNetworkReturn(
            reduced_value=qs.mean(axis=0),
            ensemble_values=qs,
            ensemble_variance=qs.var(axis=0),
        )

    def get_actions(self, state: State, n: int = 1):
        self._jax_rng, action_rng = jax.random.split(self._jax_rng)
        actions, _ = self._actor.get_actions(action_rng, self._actor_state.actor.params, state, n=n)

        return actions

    def get_action_interaction(self, state: State) -> np.ndarray:
        """
        Samples a single action during interaction.
        """
        self._app_state.event_bus.emit_event(RLEventType.agent_get_action)

        self._jax_rng, action_rng = jax.random.split(self._jax_rng)
        jaxtion, metrics = self._actor.get_actions(
            action_rng,
            self._actor_state.actor.params,
            state,
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
        if pr.trajectories is None:
            return

        self._app_state.event_bus.emit_event(RLEventType.agent_update_buffer)

        transitions = [convert_trajectory_to_transition(t) for t in pr.trajectories]

        self.critic_buffer.feed(transitions, pr.data_mode)
        recent_actor_idxs = self._actor_buffer.feed(transitions, pr.data_mode)
        self.log_buffer_sizes()

        # ---------------------------------- ingress actor loss metic --------------------------------- #
        if len(recent_actor_idxs) > 0:
            recent_actor_batch: Transition = self._actor_buffer.get_batch(recent_actor_idxs)

            state = recent_actor_batch.state

            # vmap over batch
            log_probs = jax_u.vmap_except(self._actor.get_log_probs, exclude=["params"])(
                self._actor_state.actor.params,
                state,
                jnp.expand_dims(recent_actor_batch.action, axis=1),
            )
            actor_loss = -log_probs.mean()

            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric=f"ingress_actor_loss_{pr.data_mode.name}",
                value=actor_loss,
            )

        # ------------------------- trajectory length metric ------------------------- #

        for t in pr.trajectories:
            self._app_state.metrics.write(
                agent_step=self._app_state.agent_step,
                metric="pipeline_trajectory_len",
                value=len(t),
            )

    # --------------------------- critic updating-------------------------- #

    def update_critic(self) -> list[float]:
        if not self.critic_buffer.is_sampleable:
            return [0 for _ in range(len(self.critic._reset_manager.active_indices))]

        critic_batch: Transition = self.critic_buffer.sample()
        self._jax_rng, next_action_rng = jax.random.split(self._jax_rng)
        next_actions, _ = self._actor.get_actions(
            next_action_rng,
            self._actor_state.actor.params,
            critic_batch.next_state,
            self.cfg.bootstrap_action_samples,
        )

        self._jax_rng, critic_update_rng = jax.random.split(self._jax_rng)
        self._critic_state, metrics = self.critic.update(
            critic_update_rng,
            self._critic_state,
            critic_batch,
            next_actions,
        )
        rolling_reset_metrics = self.critic.get_rolling_reset_metrics()

        metrics_dict = create_ensemble_dict(
            metrics,
            lambda m: extract_metrics(m, [
                'layer_grad_norms', 'layer_weight_norms', 'loss', 'q_loss', 'h_loss',
                'q', 'h', 'delta_l', 'delta_r', 'action_reg_loss', 'h_reg_loss',
                'ensemble_grad_norms', 'ensemble_weight_norms',
            ]),
        )
        self._app_state.metrics.write_dict(
            metrics_dict,
            agent_step=self._app_state.agent_step,
        )
        self._app_state.metrics.write_dict(
            rolling_reset_metrics,
            agent_step=self._app_state.agent_step,
        )

        stable_ranks_dict = create_ensemble_dict(
            self._critic_state.params,
            get_stable_rank,
            prefix='stable_rank_',
        )
        self._app_state.metrics.write_dict(
            stable_ranks_dict,
            agent_step=self._app_state.agent_step,
        )

        rolling_reset_metrics = self.critic.get_rolling_reset_metrics()
        self._app_state.metrics.write_dict(
            rolling_reset_metrics,
            agent_step=self._app_state.agent_step,
        )

        return [metrics.loss[i].mean().item() for i in self.critic._reset_manager.active_indices]

    @jax_u.method_jit
    def _aggregate_ensemble_values(self, ensemble_values: jax.Array) -> jax.Array:
        if self.cfg.policy.ensemble_aggregation == "mean":
            return ensemble_values.mean(axis=0)
        if self.cfg.policy.ensemble_aggregation == "percentile":
            return jnp.percentile(ensemble_values, self.cfg.policy.ensemble_percentile * 100, axis=0)
        raise ValueError(f"Unknown ensemble aggregation method: {self.cfg.policy.ensemble_aggregation}")

    def ensemble_ve(self, params: chex.ArrayTree, rng: chex.PRNGKey, x: jax.Array, a: jax.Array):
        """
        returns reduced state-action value estimate from ensemble of critics for:
            1 state, and
            1 action OR a batch of actions

        shape of returned q estimates is respectively () or (n_samples,)
        """
        qs = self.critic.forward(params, rng, x, a).q
        aggregated_values = self._aggregate_ensemble_values(qs)
        return aggregated_values.squeeze(-1)

    def update_actor(self):
        if not self._actor_buffer.is_sampleable:
            return 0.

        self._jax_rng, actor_update_rng = jax.random.split(self._jax_rng)
        actor_batch: Transition = self._actor_buffer.sample()
        self._actor_state, metrics = self._actor.update(
            actor_update_rng,
            self._actor_state,
            self.ensemble_ve,
            self._critic_state.params,
            actor_batch,
        )
        return metrics.actor_loss.mean().item()

    def update(self) -> list[float]:
        if not self._has_preinitialized:
            self._has_preinitialized = True
            self._jax_rng, critic_init_rng, actor_init_rng = jax.random.split(self._jax_rng, 3)

            self._critic_state = self.critic.initialize_to_nominal_action(
                critic_init_rng,
                self._critic_state,
                self._nominal_setpoints,
            )

            actor_state = self._actor.initialize_to_nominal_action(
                actor_init_rng,
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
        self._app_state.event_bus.emit_event(RLEventType.agent_save)

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
        self._app_state.event_bus.emit_event(RLEventType.agent_load)

        actor_path = path / "actor.pkl"
        with open(actor_path, "rb") as f:
            self._actor_state = pkl.load(f)

        actor_buffer_path = path / "actor_buffer.pkl"
        with open(actor_buffer_path, "rb") as f:
            self._actor_buffer = pkl.load(f)

        critic_path = path / "critic.pkl"
        with open(critic_path, "rb") as f:
            self._critic_state = pkl.load(f)

        critic_buffer_path = path / "critic_buffer.pkl"
        with open(critic_buffer_path, "rb") as f:
            self.critic_buffer = pkl.load(f)

    def get_buffer_sizes(self) -> dict[str, list[int]]:
        return {
            "critic": [self.critic_buffer.size],
            "policy": [self._actor_buffer.size],
        }

    def log_buffer_sizes(self):
        self._app_state.metrics.write(
            agent_step=self._app_state.agent_step,
            metric="BUFFER-CRITIC-size",
            value=self.critic_buffer.size,
        )

        self._app_state.metrics.write(
            agent_step=self._app_state.agent_step,
            metric="BUFFER-ACTOR-size",
            value=self._actor_buffer.size,
        )

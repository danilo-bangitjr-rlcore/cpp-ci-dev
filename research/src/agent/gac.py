from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import chex
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u
import numpy as np
from lib_agent.actor.actor_registry import get_actor
from lib_agent.actor.percentile_actor import PercentileActor
from lib_agent.buffer.buffer import EnsembleReplayBuffer, VectorizedTransition
from lib_agent.critic.critic_registry import get_critic
from ml_instrumentation.Collector import Collector

from interaction.transition_creator import Transition


class CriticState(Protocol):
    @property
    def params(self) -> chex.ArrayTree: ...

class PolicyState(Protocol):
    @property
    def params(self) -> chex.ArrayTree: ...

    @property
    def opt_state(self) -> chex.ArrayTree: ...

class ActorState(Protocol):
    @property
    def actor(self) -> PolicyState: ...

    @property
    def proposal(self) -> PolicyState: ...

class GACState(NamedTuple):
    critic: CriticState
    actor: ActorState

@dataclass
class GreedyACConfig:
    name: str
    batch_size: int
    critic: dict[str, Any]
    actor: dict[str, Any]

class GACCritic(Protocol):
    def init_state(self, rng: chex.PRNGKey, x: jax.Array, a: jax.Array) -> CriticState: ...
    def forward(self, params: chex.ArrayTree, x: jax.Array, a: jax.Array) -> jax.Array: ...
    def update(
        self,
        critic_state: CriticState,
        transitions: VectorizedTransition,
        next_actions: jax.Array,
    ) -> tuple[CriticState, dict]: ...


class GreedyAC:
    def __init__(self, cfg: GreedyACConfig, seed: int, state_dim: int, action_dim: int, collector: Collector):
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._cfg = cfg
        self._collector = collector

        self._critic: GACCritic = get_critic(cfg.critic, seed, state_dim, action_dim, collector)
        self._actor: PercentileActor = get_actor(cfg.actor, seed, state_dim, action_dim, collector)

        # Replay Buffers
        self.policy_buffer = EnsembleReplayBuffer(
            n_ensemble=1,
            ensemble_prob=1.0,
            batch_size=cfg.batch_size,
        )

        self.critic_buffer = EnsembleReplayBuffer(
            n_ensemble=cfg.critic['ensemble'],
            ensemble_prob=cfg.critic['ensemble_prob'],
            batch_size=cfg.batch_size,
        )

        # Agent State
        dummy_x = jnp.zeros(self.state_dim)
        dummy_a = jnp.zeros(self.action_dim)

        self.rng, c_rng = jax.random.split(self.rng)
        critic_state = self._critic.init_state(c_rng, dummy_x, dummy_a)
        actor_state = self._actor.init_state(c_rng, dummy_x)

        self.agent_state = GACState(critic_state, actor_state)

    def update_buffer(self, transition: Transition):
        self.critic_buffer.add(transition)
        self.policy_buffer.add(transition)

    def get_actions(self, state: jax.Array | np.ndarray):
        state = jnp.asarray(state)
        return self._actor.get_actions(self.agent_state.actor.actor.params, state)

    def get_action_values(self, state: jax.Array | np.ndarray, actions: jax.Array | np.ndarray):
        return self._critic.forward(
            self.agent_state.critic.params,
            x=jnp.asarray(state),
            a=jnp.asarray(actions),
        )

    def get_probs(self, actor_params: chex.ArrayTree, state: jax.Array | np.ndarray, actions: jax.Array | np.ndarray):
        state = jnp.asarray(state)
        actions = jnp.asarray(actions)
        return self._actor.get_probs(actor_params, state, actions)

    def update(self):
        self.critic_update()
        self.policy_update()

    @jax_u.method_jit
    def _get_actions_over_state(self, actor_params: chex.ArrayTree, rng: chex.PRNGKey, x: jax.Array):
        chex.assert_rank(x, 3)
        return jax_u.vmap_only(self._actor.get_actions_rng, ['state'])(
            actor_params,
            rng,
            x,
        )

    def critic_update(self):
        if self.critic_buffer.size == 0:
            return

        batch = self.critic_buffer.sample()
        next_actions = self._get_actions_over_state(self.agent_state.actor.actor.params, self.rng, batch.next_state)
        next_actions = jnp.expand_dims(next_actions, axis=2) # add singleton dimension for samples for expected update
        new_critic_state, _ = self._critic.update(
            critic_state=self.agent_state.critic,
            transitions=batch,
            next_actions=next_actions,
        )

        self.agent_state = self.agent_state._replace(critic=new_critic_state)

    def policy_update(self):
        if self.policy_buffer.size == 0:
            return

        batch = self.policy_buffer.sample()

        actor_state = self._actor.update(
            self.agent_state.actor,
            self.ensemble_ve,
            self.agent_state.critic.params,
            batch,
        )
        self.agent_state = self.agent_state._replace(actor=actor_state)



    def ensemble_ve(self, params: chex.ArrayTree, x: jax.Array, a: jax.Array):
        ens_forward = jax_u.vmap_only(self._critic.forward, ['params'])
        qs = ens_forward(params, x, a)
        return qs.mean(axis=0).squeeze(-1)

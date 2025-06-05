from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import chex
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u
import numpy as np
from lib_agent.actor.actor_registry import get_actor
from lib_agent.actor.percentile_actor import PercentileActor, State
from lib_agent.buffer.buffer import EnsembleReplayBuffer
from lib_agent.critic.critic_registry import get_critic
from ml_instrumentation.Collector import Collector

from agent.interface import Batch
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


class BatchWithState(NamedTuple):
    state: State
    action: jax.Array
    reward: jax.Array
    next_state: State
    gamma: jax.Array


class GACCritic(Protocol):
    def init_state(self, rng: chex.PRNGKey, x: jax.Array, a: jax.Array) -> CriticState: ...
    def forward(self, params: chex.ArrayTree, x: jax.Array, a: jax.Array) -> jax.Array: ...
    def update(
        self,
        critic_state: CriticState,
        transitions: BatchWithState,
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
        self.policy_buffer = EnsembleReplayBuffer[Batch](
            n_ensemble=1,
            ensemble_prob=1.0,
            batch_size=cfg.batch_size,
        )

        self.critic_buffer = EnsembleReplayBuffer[Batch](
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
        t = Batch(
            state=jnp.array(transition.state),
            action=jnp.array(transition.action),
            reward=jnp.array([transition.reward]),
            next_state=jnp.array(transition.next_state),
            gamma=jnp.array([transition.gamma]),

            a_lo=jnp.array(transition.a_lo),
            a_hi=jnp.array(transition.a_hi),
            next_a_lo=jnp.array(transition.next_a_lo),
            next_a_hi=jnp.array(transition.next_a_hi),
        )
        self.critic_buffer.add(t)
        self.policy_buffer.add(t)

    def get_actions(self, state: State):
        return self._actor.get_actions(self.agent_state.actor.actor.params, state)

    def get_action_values(self, state: State, actions: jax.Array | np.ndarray):
        return self._critic.forward(
            self.agent_state.critic.params,
            x=state.features,
            a=jnp.asarray(actions),
        )

    def get_probs(self, actor_params: chex.ArrayTree, state: State, actions: jax.Array | np.ndarray):
        actions = jnp.asarray(actions)
        return self._actor.get_probs(actor_params, state, actions)

    def update(self):
        self.critic_update()
        self.policy_update()

    @jax_u.method_jit
    def _get_actions_over_state(self, actor_params: chex.ArrayTree, rng: chex.PRNGKey, x: State):
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
        next_state = State(
            features=batch.next_state,
            a_lo=batch.next_a_lo,
            a_hi=batch.next_a_hi,
        )
        next_actions = self._get_actions_over_state(self.agent_state.actor.actor.params, self.rng, next_state)
        next_actions = jnp.expand_dims(next_actions, axis=2) # add singleton dimension for samples for expected update
        new_critic_state, _ = self._critic.update(
            critic_state=self.agent_state.critic,
            transitions=BatchWithState(
                state=State(
                    features=batch.state,
                    a_lo=batch.a_lo,
                    a_hi=batch.a_hi,
                ),
                action=batch.action,
                reward=batch.reward,
                next_state=next_state,
                gamma=batch.gamma,
            ),
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
            BatchWithState(
                state=State(
                    features=batch.state,
                    a_lo=batch.a_lo,
                    a_hi=batch.a_hi,
                ),
                action=batch.action,
                reward=batch.reward,
                next_state=State(
                    features=batch.next_state,
                    a_lo=batch.next_a_lo,
                    a_hi=batch.next_a_hi,
                ),
                gamma=batch.gamma,
            ),
        )
        self.agent_state = self.agent_state._replace(actor=actor_state)



    def ensemble_ve(self, params: chex.ArrayTree, x: jax.Array, a: jax.Array):
        ens_forward = jax_u.vmap_only(self._critic.forward, ['params'])
        qs = ens_forward(params, x, a)
        return qs.mean(axis=0).squeeze(-1)

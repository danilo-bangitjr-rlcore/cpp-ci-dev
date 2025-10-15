from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u
import numpy as np
from pytest_benchmark.fixture import BenchmarkFixture

from lib_agent.buffer.buffer import State
from lib_agent.critic.adv_critic import AdvConfig, AdvCritic
from lib_agent.critic.critic_utils import CriticBatch, RollingResetConfig


class FakeAdvCriticBatch(NamedTuple):
    """Simple critic batch for benchmarking."""
    state: State
    action: jax.Array
    reward: jax.Array
    next_state: State
    gamma: jax.Array


def _create_test_adv_critic(ensemble_size: int, state_dim: int, action_dim: int):
    """Create an AdvCritic for testing."""
    config = AdvConfig(
        name="ADV",
        stepsize=3e-4,
        ensemble=ensemble_size,
        ensemble_prob=0.5,
        num_policy_actions=64,
        advantage_centering_weight=0.1,
        l2_regularization=1.0,
        nominal_setpoint_updates=100,
        use_all_layer_norm=True,
        num_rand_actions=10,
        action_regularization=1.0,
        rolling_reset_config=RollingResetConfig(reset_period=10000, warm_up_steps=1000),
        adv_l2_regularization=1.0,
    )
    return AdvCritic(config, seed=42, state_dim=state_dim, action_dim=action_dim)


def _create_fake_adv_state(rng: np.random.Generator, state_dim: int, action_dim: int):
    """Create a fake state for testing."""
    return State(
        features=jnp.array(rng.random((state_dim,))),
        a_lo=jnp.array([-1.0] * action_dim),
        a_hi=jnp.array([1.0] * action_dim),
        dp=jnp.array([False]),
        last_a=jnp.array(rng.random((action_dim,))),
    )


def test_adv_critic_forward_pass_single(benchmark: BenchmarkFixture):
    """Benchmark single advantage critic forward pass."""
    rng = np.random.default_rng(0)
    state_dim, action_dim = 64, 4

    critic = _create_test_adv_critic(ensemble_size=1, state_dim=state_dim, action_dim=action_dim)

    init_rng = jax.random.PRNGKey(42)
    state = _create_fake_adv_state(rng, state_dim, action_dim)
    action = jnp.array(rng.random((action_dim,)))

    critic_state = critic.init_state(init_rng, state.features, action)

    def _inner(
        critic: AdvCritic,
        params: Any,
        rng_key: jax.Array,
        state_features: jax.Array,
        action: jax.Array,
    ):
        for _ in range(50):
            values = jax_u.vmap_only(critic.get_values, ['params'])(params, rng_key, state_features, action).adv
            # Force computation
            _ = values.sum()

    benchmark(
        _inner,
        critic,
        critic_state.params,
        init_rng,
        state.features,
        action,
    )


def test_adv_critic_forward_pass_ensemble(benchmark: BenchmarkFixture):
    """Benchmark ensemble advantage critic forward pass."""
    rng = np.random.default_rng(0)
    state_dim, action_dim = 128, 8
    ensemble_size = 4

    critic = _create_test_adv_critic(ensemble_size=ensemble_size, state_dim=state_dim, action_dim=action_dim)

    init_rng = jax.random.PRNGKey(42)
    state = _create_fake_adv_state(rng, state_dim, action_dim)
    action = jnp.array(rng.random((action_dim,)))

    critic_state = critic.init_state(init_rng, state.features, action)

    def _inner(
        critic: AdvCritic,
        params: Any,
        rng_key: jax.Array,
        state_features: jax.Array,
        action: jax.Array,
    ):
        for _ in range(25):
            values = jax_u.vmap_only(critic.get_values, ['params'])(params, rng_key, state_features, action).adv
            # Force computation
            _ = values.sum()

    benchmark(
        _inner,
        critic,
        critic_state.params,
        init_rng,
        state.features,
        action,
    )


def test_adv_critic_forward_pass_batch(benchmark: BenchmarkFixture):
    """Benchmark batched advantage critic forward pass."""
    rng = np.random.default_rng(0)
    state_dim, action_dim = 64, 4
    batch_size = 256

    critic = _create_test_adv_critic(ensemble_size=1, state_dim=state_dim, action_dim=action_dim)

    init_rng = jax.random.PRNGKey(42)
    single_state = _create_fake_adv_state(rng, state_dim, action_dim)
    single_action = jnp.array(rng.random((action_dim,)))

    critic_state = critic.init_state(init_rng, single_state.features, single_action)

    # Create batched data
    batch_states = jnp.array(rng.random((batch_size, state_dim)))
    batch_actions = jnp.array(rng.random((batch_size, action_dim)))

    def _inner(critic: AdvCritic, params: Any, rng_keys: jax.Array, states: jax.Array, actions: jax.Array):
        for _ in range(20):
            values = jax_u.vmap_only(critic.get_values, ['params'])(params, rng_keys, states, actions).adv
            # Force computation
            _ = values.sum()

    # Create batch of RNG keys for vmap
    batch_rngs = jax.random.split(init_rng, batch_size)

    benchmark(
        _inner,
        critic,
        critic_state.params,
        batch_rngs,
        batch_states,
        batch_actions,
    )


def test_adv_critic_representations_forward(benchmark: BenchmarkFixture):
    """Benchmark advantage critic representation computation."""
    rng = np.random.default_rng(0)
    state_dim, action_dim = 128, 8
    batch_size = 128

    critic = _create_test_adv_critic(ensemble_size=1, state_dim=state_dim, action_dim=action_dim)

    init_rng = jax.random.PRNGKey(42)
    single_state = _create_fake_adv_state(rng, state_dim, action_dim)
    single_action = jnp.array(rng.random((action_dim,)))

    critic_state = critic.init_state(init_rng, single_state.features, single_action)

    # Create batched data
    batch_states = jnp.array(rng.random((batch_size, state_dim)))
    batch_actions = jnp.array(rng.random((batch_size, action_dim)))

    def _inner(critic: AdvCritic, params: Any, rng_keys: jax.Array, states: jax.Array, actions: jax.Array):
        for _ in range(20):
            representations = jax_u.vmap_only(critic.get_representations, ['params'])(
                params, rng_keys, states, actions,
            )
            # Force computation
            _ = representations.sum()

    # Create batch of RNG keys for vmap
    batch_rngs = jax.random.split(init_rng, batch_size)

    benchmark(
        _inner,
        critic,
        critic_state.params,
        batch_rngs,
        batch_states,
        batch_actions,
    )


def test_adv_critic_ensemble_update_small(benchmark: BenchmarkFixture):
    """Benchmark ensemble advantage critic update with small ensemble."""
    rng = np.random.default_rng(0)
    state_dim, action_dim = 64, 4
    batch_size = 128
    ensemble_size = 2
    num_policy_actions = 64

    critic = _create_test_adv_critic(ensemble_size=ensemble_size, state_dim=state_dim, action_dim=action_dim)

    init_rng = jax.random.PRNGKey(42)
    single_state = _create_fake_adv_state(rng, state_dim, action_dim)
    single_action = jnp.array(rng.random((action_dim,)))

    critic_state = critic.init_state(init_rng, single_state.features, single_action)

    # Create batch of transitions
    states = jnp.array(rng.random((ensemble_size, batch_size, state_dim)))
    actions = jnp.array(rng.random((ensemble_size, batch_size, action_dim)))
    rewards = jnp.array(rng.random((ensemble_size, batch_size)))
    next_states = jnp.array(rng.random((ensemble_size, batch_size, state_dim)))
    gammas = jnp.full((ensemble_size, batch_size), 0.99)

    # Create state objects
    batch_state = State(
        features=states,
        a_lo=jnp.tile(jnp.array([-1.0] * action_dim), (ensemble_size, batch_size, 1)),
        a_hi=jnp.tile(jnp.array([1.0] * action_dim), (ensemble_size, batch_size, 1)),
        dp=jnp.zeros((ensemble_size, batch_size, 1), dtype=bool),
        last_a=actions,
    )

    next_batch_state = State(
        features=next_states,
        a_lo=jnp.tile(jnp.array([-1.0] * action_dim), (ensemble_size, batch_size, 1)),
        a_hi=jnp.tile(jnp.array([1.0] * action_dim), (ensemble_size, batch_size, 1)),
        dp=jnp.zeros((ensemble_size, batch_size, 1), dtype=bool),
        last_a=actions,
    )

    transitions = FakeAdvCriticBatch(
        state=batch_state,
        action=actions,
        reward=rewards,
        next_state=next_batch_state,
        gamma=gammas,
    )

    policy_actions = jnp.array(rng.random((ensemble_size, batch_size, num_policy_actions, action_dim)))
    policy_probs = jnp.array(rng.random((ensemble_size, batch_size, num_policy_actions)))

    def _inner(
        critic: AdvCritic,
        state: Any,
        transitions: CriticBatch,
        policy_actions: jax.Array,
        policy_probs: jax.Array,
    ):
        for _ in range(5):
            _new_state, metrics = critic.update(state, transitions, policy_actions, policy_probs)
            # Force computation
            _ = metrics.loss.sum()

    benchmark(_inner, critic, critic_state, transitions, policy_actions, policy_probs)


def test_adv_critic_ensemble_update_large(benchmark: BenchmarkFixture):
    """Benchmark ensemble advantage critic update with large ensemble."""
    rng = np.random.default_rng(0)
    state_dim, action_dim = 128, 8
    batch_size = 256
    ensemble_size = 8
    num_policy_actions = 64

    critic = _create_test_adv_critic(ensemble_size=ensemble_size, state_dim=state_dim, action_dim=action_dim)

    # Initialize critic
    init_rng = jax.random.PRNGKey(42)
    single_state = _create_fake_adv_state(rng, state_dim, action_dim)
    single_action = jnp.array(rng.random((action_dim,)))

    critic_state = critic.init_state(init_rng, single_state.features, single_action)

    # Create batch of transitions
    states = jnp.array(rng.random((ensemble_size, batch_size, state_dim)))
    actions = jnp.array(rng.random((ensemble_size, batch_size, action_dim)))
    rewards = jnp.array(rng.random((ensemble_size, batch_size)))
    next_states = jnp.array(rng.random((ensemble_size, batch_size, state_dim)))
    gammas = jnp.full((ensemble_size, batch_size), 0.99)

    # Create state objects
    batch_state = State(
        features=states,
        a_lo=jnp.tile(jnp.array([-1.0] * action_dim), (ensemble_size, batch_size, 1)),
        a_hi=jnp.tile(jnp.array([1.0] * action_dim), (ensemble_size, batch_size, 1)),
        dp=jnp.zeros((ensemble_size, batch_size, 1), dtype=bool),
        last_a=actions,
    )

    next_batch_state = State(
        features=next_states,
        a_lo=jnp.tile(jnp.array([-1.0] * action_dim), (ensemble_size, batch_size, 1)),
        a_hi=jnp.tile(jnp.array([1.0] * action_dim), (ensemble_size, batch_size, 1)),
        dp=jnp.zeros((ensemble_size, batch_size, 1), dtype=bool),
        last_a=actions,
    )

    transitions = FakeAdvCriticBatch(
        state=batch_state,
        action=actions,
        reward=rewards,
        next_state=next_batch_state,
        gamma=gammas,
    )

    policy_actions = jnp.array(rng.random((ensemble_size, batch_size, num_policy_actions, action_dim)))
    policy_probs = jnp.array(rng.random((ensemble_size, batch_size, num_policy_actions)))

    def _inner(
        critic: AdvCritic,
        state: Any,
        transitions: CriticBatch,
        policy_actions: jax.Array,
        policy_probs: jax.Array,
    ):
        for _ in range(3):
            _new_state, metrics = critic.update(state, transitions, policy_actions, policy_probs)
            # Force computation
            _ = metrics.loss.sum()

    benchmark(_inner, critic, critic_state, transitions, policy_actions, policy_probs)


def test_adv_critic_value_only_forward(benchmark: BenchmarkFixture):
    """Benchmark value-only forward pass (efficient state-only computation)."""
    rng = np.random.default_rng(0)
    state_dim, action_dim = 128, 8
    batch_size = 256

    critic = _create_test_adv_critic(ensemble_size=1, state_dim=state_dim, action_dim=action_dim)

    init_rng = jax.random.PRNGKey(42)
    single_state = _create_fake_adv_state(rng, state_dim, action_dim)
    single_action = jnp.array(rng.random((action_dim,)))

    critic_state = critic.init_state(init_rng, single_state.features, single_action)

    # Create batched data
    batch_states = jnp.array(rng.random((batch_size, state_dim)))

    def _inner(critic: AdvCritic, params: Any, rng: chex.PRNGKey, states: jax.Array):
        for _ in range(20):
            # Call internal _forward_val to test value-only computation
            values = jax_u.vmap_only(critic._forward_val, ['params'])(params, rng, states)
            # Force computation
            _ = values.v.sum() + values.h.sum()

    # Create batch of RNG keys for vmap
    benchmark(
        _inner,
        critic,
        critic_state.params,
        init_rng,
        batch_states,
    )

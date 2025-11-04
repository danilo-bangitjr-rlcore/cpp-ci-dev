from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from lib_utils.named_array import NamedArray
from pytest_benchmark.fixture import BenchmarkFixture

from lib_agent.buffer.datatypes import State, Transition
from lib_agent.critic.adv_critic import AdvConfig, AdvCritic
from lib_agent.critic.critic_utils import RollingResetConfig


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
    return AdvCritic(config, state_dim=state_dim, action_dim=action_dim)


def _create_fake_adv_state(rng: np.random.Generator, state_dim: int, action_dim: int):
    """Create a fake state for testing."""
    return State(
        features=NamedArray.unnamed(rng.random((state_dim,))),
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

    critic_state = critic.init_state(init_rng, state.features.array, action)

    def _inner(
        critic: AdvCritic,
        params: Any,
        rng_key: jax.Array,
        state_features: jax.Array,
        action: jax.Array,
    ):
        for _ in range(50):
            values = critic.forward(params, rng_key, state_features, action).adv
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

    critic_state = critic.init_state(init_rng, state.features.array, action)

    def _inner(
        critic: AdvCritic,
        params: Any,
        rng_key: jax.Array,
        state_features: jax.Array,
        action: jax.Array,
    ):
        for _ in range(25):
            values = critic.forward(params, rng_key, state_features, action).adv
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

    critic_state = critic.init_state(init_rng, single_state.features.array, single_action)

    # Create batched data
    batch_states = jnp.array(rng.random((batch_size, state_dim)))
    batch_actions = jnp.array(rng.random((batch_size, action_dim)))

    def _inner(critic: AdvCritic, params: Any, rng_key: jax.Array, states: jax.Array, actions: jax.Array):
        for _ in range(20):
            values = critic.forward(params, rng_key, states, actions).adv
            # Force computation
            _ = values.sum()

    benchmark(
        _inner,
        critic,
        critic_state.params,
        init_rng,
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

    critic_state = critic.init_state(init_rng, single_state.features.array, single_action)

    # Create batch of transitions
    states = jnp.array(rng.random((ensemble_size, batch_size, state_dim)))
    actions = jnp.array(rng.random((ensemble_size, batch_size, action_dim)))
    rewards = jnp.array(rng.random((ensemble_size, batch_size)))
    next_states = jnp.array(rng.random((ensemble_size, batch_size, state_dim)))
    gammas = jnp.full((ensemble_size, batch_size), 0.99)

    # Create state objects
    batch_state = State(
        features=NamedArray.unnamed(states),
        a_lo=jnp.tile(jnp.array([-1.0] * action_dim), (ensemble_size, batch_size, 1)),
        a_hi=jnp.tile(jnp.array([1.0] * action_dim), (ensemble_size, batch_size, 1)),
        dp=jnp.zeros((ensemble_size, batch_size, 1), dtype=bool),
        last_a=actions,
    )

    next_batch_state = State(
        features=NamedArray.unnamed(next_states),
        a_lo=jnp.tile(jnp.array([-1.0] * action_dim), (ensemble_size, batch_size, 1)),
        a_hi=jnp.tile(jnp.array([1.0] * action_dim), (ensemble_size, batch_size, 1)),
        dp=jnp.zeros((ensemble_size, batch_size, 1), dtype=bool),
        last_a=actions,
    )

    transitions = Transition(
        state=batch_state,
        action=actions,
        n_step_reward=rewards,
        next_state=next_batch_state,
        n_step_gamma=gammas,
    )

    policy_actions = jnp.array(rng.random((ensemble_size, batch_size, num_policy_actions, action_dim)))
    policy_probs = jnp.array(rng.random((ensemble_size, batch_size, num_policy_actions)))

    def _inner(
        critic: AdvCritic,
        rng_key: jax.Array,
        state: Any,
        transitions: Transition,
        policy_actions: jax.Array,
        policy_probs: jax.Array,
    ):
        for _ in range(5):
            _new_state, metrics = critic.update(rng_key, state, transitions, policy_actions, policy_probs)
            # Force computation
            _ = metrics.loss.sum()

    update_rng = jax.random.PRNGKey(999)
    benchmark(_inner, critic, update_rng, critic_state, transitions, policy_actions, policy_probs)


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

    critic_state = critic.init_state(init_rng, single_state.features.array, single_action)

    # Create batch of transitions
    states = jnp.array(rng.random((ensemble_size, batch_size, state_dim)))
    actions = jnp.array(rng.random((ensemble_size, batch_size, action_dim)))
    rewards = jnp.array(rng.random((ensemble_size, batch_size)))
    next_states = jnp.array(rng.random((ensemble_size, batch_size, state_dim)))
    gammas = jnp.full((ensemble_size, batch_size), 0.99)

    # Create state objects
    batch_state = State(
        features=NamedArray.unnamed(states),
        a_lo=jnp.tile(jnp.array([-1.0] * action_dim), (ensemble_size, batch_size, 1)),
        a_hi=jnp.tile(jnp.array([1.0] * action_dim), (ensemble_size, batch_size, 1)),
        dp=jnp.zeros((ensemble_size, batch_size, 1), dtype=bool),
        last_a=actions,
    )

    next_batch_state = State(
        features=NamedArray.unnamed(next_states),
        a_lo=jnp.tile(jnp.array([-1.0] * action_dim), (ensemble_size, batch_size, 1)),
        a_hi=jnp.tile(jnp.array([1.0] * action_dim), (ensemble_size, batch_size, 1)),
        dp=jnp.zeros((ensemble_size, batch_size, 1), dtype=bool),
        last_a=actions,
    )

    transitions = Transition(
        state=batch_state,
        action=actions,
        n_step_reward=rewards,
        next_state=next_batch_state,
        n_step_gamma=gammas,
    )

    policy_actions = jnp.array(rng.random((ensemble_size, batch_size, num_policy_actions, action_dim)))
    policy_probs = jnp.array(rng.random((ensemble_size, batch_size, num_policy_actions)))

    def _inner(
        critic: AdvCritic,
        rng_key: jax.Array,
        state: Any,
        transitions: Transition,
        policy_actions: jax.Array,
        policy_probs: jax.Array,
    ):
        for _ in range(3):
            _new_state, metrics = critic.update(rng_key, state, transitions, policy_actions, policy_probs)
            # Force computation
            _ = metrics.loss.sum()

    update_rng = jax.random.PRNGKey(999)
    benchmark(_inner, critic, update_rng, critic_state, transitions, policy_actions, policy_probs)

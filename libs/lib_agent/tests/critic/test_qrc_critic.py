from dataclasses import dataclass

import chex
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u
import pytest

from lib_agent.critic.qrc_critic import QRCConfig, QRCCritic


@dataclass
class MockState:
    features: jax.Array
    a_lo: jax.Array
    a_hi: jax.Array


@pytest.fixture
def critic_config() -> QRCConfig:
    return QRCConfig(
        name="qrc",
        stepsize=0.001,
        ensemble=3,
        ensemble_prob=0.7,
        num_rand_actions=10,
        action_regularization=0.1,
        action_regularization_epsilon=0.1,
        l2_regularization=1.0,
        nominal_setpoint_updates=10,
        use_noisy_nets=False,
    )


@pytest.fixture
def critic(critic_config: QRCConfig) -> QRCCritic:
    state_dim = 4
    action_dim = 2
    seed = 42
    return QRCCritic(critic_config, seed, state_dim, action_dim)


# ----------
# -- init --
# ----------


def test_init_state(critic: QRCCritic):
    """
    Test that the critic can be initialized and returns a valid state.
    """
    state = jnp.array([1.0, 2.0, 3.0, 4.0])
    action = jnp.array([0.5, -0.5])
    rng = jax.random.PRNGKey(123)

    critic_state = critic.init_state(rng, state, action)

    assert hasattr(critic_state, 'params')
    assert hasattr(critic_state, 'opt_state')
    chex.assert_tree_shape_prefix(critic_state.params, (critic._cfg.ensemble,))


# ----------------
# -- get_values --
# ----------------

def test_get_values_single_sample(critic: QRCCritic):
    """
    Given a valid critic state, an observation, and an action, the critic
    should produce a valid q value.

    There should be one q value for each member of the critic ensemble
    assuming a single observation and action is given.
    """
    state = jnp.array([1.0, 2.0, 3.0, 4.0])
    action = jnp.array([0.5, -0.5])
    rng = jax.random.PRNGKey(123)

    # Initialize critic state
    critic_state = critic.init_state(rng, state, action)

    # Get Q values
    rng_apply = jax.random.PRNGKey(456)
    q_values = jax_u.vmap_only(critic.get_values, ['params'])(
        critic_state.params,
        rng_apply,
        state,
        action,
    )

    # Check output shape: should be (ensemble, 1) for single sample
    assert q_values.shape == (critic._cfg.ensemble, 1)
    assert q_values.dtype == jnp.float32


def test_get_values_batch(critic: QRCCritic):
    """
    Given a valid critic state, an observation, and an action, the critic
    should produce a valid q value.

    Q value should have shape (ensemble, batch_size, 1) if a batch of observations
    and actions are given.
    """
    batch_size = 8
    rng_data = jax.random.PRNGKey(789)

    rng_states, rng_actions = jax.random.split(rng_data)
    batch_states = jax.random.normal(rng_states, (batch_size, 4))
    batch_actions = jax.random.normal(rng_actions, (batch_size, 2))

    rng = jax.random.PRNGKey(123)
    critic_state = critic.init_state(rng, batch_states[0], batch_actions[0])

    # Get Q values for batch
    rng_apply = jax.random.PRNGKey(456)
    rngs_batch = jax.random.split(rng_apply, batch_size)
    q_values = jax_u.vmap_only(critic.get_values, ['params'])(
        critic_state.params,
        rngs_batch,
        batch_states,
        batch_actions,
    )

    assert q_values.shape == (critic._cfg.ensemble, batch_size, 1)
    assert q_values.dtype == jnp.float32


def test_get_values_batch_n_samples(critic: QRCCritic):
    """
    Given a valid critic state, an observation, and an action, the critic
    should produce a valid q value.

    Q value should have shape (ensemble, batch_size, n_samples, 1) if a batch of observations
    and actions are given AND the action batch has an n_samples dimension.
    """
    batch_size = 4
    n_samples = 6
    rng_data = jax.random.PRNGKey(789)

    rng_states, rng_actions = jax.random.split(rng_data)
    batch_states = jax.random.normal(rng_states, (batch_size, 4))
    batch_actions = jax.random.normal(rng_actions, (batch_size, n_samples, 2))

    rng = jax.random.PRNGKey(123)

    # Initialize critic state
    critic_state = critic.init_state(rng, batch_states[0], batch_actions[0, 0])

    # Get Q values for batch with samples
    rng_apply = jax.random.PRNGKey(456)
    rngs_batch = jax.random.split(rng_apply, (batch_size, n_samples))
    q_values = jax_u.vmap_only(critic.get_values, ['params'])(
        critic_state.params,
        rngs_batch,
        batch_states,
        batch_actions,
    )

    # Check output shape: should be (ensemble, batch_size, n_samples, 1)
    assert q_values.shape == (critic._cfg.ensemble, batch_size, n_samples, 1)
    assert q_values.dtype == jnp.float32

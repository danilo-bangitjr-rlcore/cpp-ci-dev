from dataclasses import dataclass
from itertools import starmap
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import lib_utils.jax as jax_u
import pytest
from lib_utils.named_array import NamedArray

from lib_agent.buffer.buffer import State
from lib_agent.critic.adv_critic import AdvConfig, AdvCritic
from lib_agent.critic.critic_utils import RollingResetConfig


@dataclass
class MockState:
    features: jax.Array
    a_lo: jax.Array
    a_hi: jax.Array


@pytest.fixture
def critic_config() -> AdvConfig:
    return AdvConfig(
        name="adv",
        stepsize=0.001,
        ensemble=3,
        ensemble_prob=0.7,
        num_policy_actions=10,
        advantage_centering_weight=0.1,
        l2_regularization=1.0,
        nominal_setpoint_updates=10,
        rolling_reset_config=RollingResetConfig(
            reset_period=100,
            warm_up_steps=50,
        ),
        num_rand_actions=10,
        action_regularization=1.0,
        adv_l2_regularization=1.0,
    )


@pytest.fixture
def critic(critic_config: AdvConfig) -> AdvCritic:
    state_dim = 4
    action_dim = 2
    seed = 42
    return AdvCritic(critic_config, seed, state_dim, action_dim)


# ----------
# -- init --
# ----------


def test_init_state(critic: AdvCritic):
    """Test that the critic can be initialized and returns a valid state."""
    state = jnp.array([1.0, 2.0, 3.0, 4.0])
    action = jnp.array([0.5, -0.5])
    rng = jax.random.PRNGKey(123)

    critic_state = critic.init_state(rng, state, action)

    assert hasattr(critic_state, 'params')
    assert hasattr(critic_state, 'opt_state')
    chex.assert_tree_shape_prefix(critic_state.params, (critic._cfg.ensemble,))


# ----------------
# -- get_adv_values --
# ----------------

def test_get_adv_values_single_sample(critic: AdvCritic):
    """
    Given a valid critic state, an observation, and an action, the critic
    should produce a valid advantage value.

    There should be one advantage value for each member of the critic ensemble
    assuming a single observation and action is given.
    """
    state = jnp.array([1.0, 2.0, 3.0, 4.0])
    action = jnp.array([0.5, -0.5])
    rng = jax.random.PRNGKey(123)

    # Initialize critic state
    critic_state = critic.init_state(rng, state, action)

    # Get advantage values
    rng_apply = jax.random.PRNGKey(456)
    adv_values = jax_u.vmap_only(critic.get_values, ['params'])(
        critic_state.params,
        rng_apply,
        state,
        action,
    ).adv

    # Check output shape: should be (ensemble, 1) for single sample
    assert adv_values.shape == (critic._cfg.ensemble, 1)
    assert adv_values.dtype == jnp.float32


def test_get_adv_values_batch(critic: AdvCritic):
    """
    Given a valid critic state, an observation, and an action, the critic
    should produce a valid advantage value.

    Advantage value should have shape (ensemble, batch_size, 1) if a batch of observations
    and actions are given.
    """
    batch_size = 8
    rng_data = jax.random.PRNGKey(789)

    rng_states, rng_actions = jax.random.split(rng_data)
    batch_states = jax.random.normal(rng_states, (batch_size, 4))
    batch_actions = jax.random.normal(rng_actions, (batch_size, 2))

    rng = jax.random.PRNGKey(123)
    critic_state = critic.init_state(rng, batch_states[0], batch_actions[0])

    # Get advantage values for batch
    rng_apply = jax.random.PRNGKey(456)
    rngs_batch = jax.random.split(rng_apply, batch_size)
    adv_values = jax_u.vmap_only(critic.get_values, ['params'])(
        critic_state.params,
        rngs_batch,
        batch_states,
        batch_actions,
    ).adv

    # Check output shape: should be (ensemble, batch_size, 1)
    assert adv_values.shape == (critic._cfg.ensemble, batch_size, 1)
    assert adv_values.dtype == jnp.float32


def test_get_adv_values_batch_n_samples(critic: AdvCritic):
    """
    Given a valid critic state, an observation, and an action, the critic
    should produce a valid advantage value.

    Advantage value should have shape (ensemble, batch_size, n_samples, 1) if a batch of observations
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

    # Get advantage values for batch with samples
    rng_apply = jax.random.PRNGKey(456)
    rngs_batch = jax.random.split(rng_apply, (batch_size, n_samples))
    adv_values = jax_u.vmap_only(critic.get_values, ['params'])(
        critic_state.params,
        rngs_batch,
        batch_states,
        batch_actions,
    ).adv

    assert adv_values.shape == (critic._cfg.ensemble, batch_size, n_samples, 1)
    assert adv_values.dtype == jnp.float32


@pytest.fixture
def rolling_reset_config() -> AdvConfig:
    return AdvConfig(
        name="test_adv",
        stepsize=0.001,
        ensemble=3,
        ensemble_prob=1.0,
        num_policy_actions=1,
        advantage_centering_weight=0.0,
        l2_regularization=0.0,
        rolling_reset_config=RollingResetConfig(
            reset_period=100,
            warm_up_steps=50,
        ),
        num_rand_actions=10,
        action_regularization=1.0,
        adv_l2_regularization=1.0,
    )


@pytest.fixture
def rolling_critic(rolling_reset_config: AdvConfig) -> AdvCritic:
    state_dim = 4
    action_dim = 2
    seed = 42
    return AdvCritic(rolling_reset_config, seed, state_dim, action_dim)


def test_rolling_reset_initialization(rolling_critic: AdvCritic):
    """Test that rolling reset manager is properly initialized."""
    assert rolling_critic._reset_manager.total_critics == 3
    assert len(rolling_critic._reset_manager.active_indices) == 3
    assert rolling_critic._reset_manager.active_indices == {0, 1, 2}

    status = rolling_critic._reset_manager.get_status()
    assert status.total_critics == 3
    assert status.active_critics == 3
    assert status.active_indices == [0, 1, 2]


def test_rolling_reset(rolling_critic: AdvCritic):
    """Test that rolling reset correctly deactivates the oldest warmed-up critic."""
    state = jnp.array([1.0, 2.0, 3.0, 4.0])
    action = jnp.array([0.5, -0.5])
    rng = jax.random.PRNGKey(123)

    critic_state = rolling_critic.init_state(rng, state, action)

    # critic 0: oldest, warmed up (should be reset)
    rolling_critic._reset_manager._critic_info[0].training_steps = 60
    rolling_critic._reset_manager._critic_info[0].birthdate = 0
    rolling_critic._reset_manager._critic_info[0].is_warmed_up = True
    rolling_critic._reset_manager._critic_info[0].is_active = True

    # critic 1: younger, warmed up
    rolling_critic._reset_manager._critic_info[1].training_steps = 60
    rolling_critic._reset_manager._critic_info[1].birthdate = 20
    rolling_critic._reset_manager._critic_info[1].is_warmed_up = True
    rolling_critic._reset_manager._critic_info[1].is_active = True

    # critic 2: not warmed up yet
    rolling_critic._reset_manager._critic_info[2].training_steps = 30
    rolling_critic._reset_manager._critic_info[2].birthdate = 10
    rolling_critic._reset_manager._critic_info[2].is_warmed_up = False
    rolling_critic._reset_manager._critic_info[2].is_active = True

    # simulate reset
    rolling_critic._reset_manager.reset(
        critic_state,
        rng,
        rolling_critic._init_member_state,
        rolling_critic._state_dim,
        rolling_critic._action_dim,
    )

    # Critic 0 should be removed from active set (reset)
    current_active = rolling_critic._reset_manager.active_indices
    assert 0 not in current_active
    assert 1 in current_active
    assert 2 in current_active

    status = rolling_critic._reset_manager.get_status()
    assert status.active_critics == 2


def test_get_active_values(rolling_critic: AdvCritic):
    """Test that get_active_values only returns values for active critics."""
    state = jnp.array([1.0, 2.0, 3.0, 4.0])
    action = jnp.array([0.5, -0.5])
    rng = jax.random.PRNGKey(123)

    critic_state = rolling_critic.init_state(rng, state, action)

    rng_apply = jax.random.PRNGKey(456)
    active_values = rolling_critic.get_active_values(critic_state.params, rng_apply, state, action).adv
    assert active_values.shape == (len(rolling_critic._reset_manager.active_indices), 1)  # (3, 1)


# ------------
# -- update --
# ------------

class MockTransition(NamedTuple):
    """Mock transition for testing update."""
    state: State
    action: jax.Array
    reward: jax.Array
    next_state: State
    gamma: jax.Array


def _create_mock_transitions(
        rng: chex.PRNGKey,
        ensemble_size: int,
        batch_size: int,
        state_dim: int,
        action_dim: int,
        ):
    """Helper to create mock transitions for testing."""
    state_rng, action_rng, reward_rng = jax.random.split(rng, 3)

    batch_states = jax.random.normal(state_rng, (ensemble_size, batch_size, state_dim))
    batch_actions = jax.random.normal(action_rng, (ensemble_size, batch_size, action_dim))
    batch_rewards = jax.random.normal(reward_rng, (ensemble_size, batch_size))
    batch_gammas = jnp.ones((ensemble_size, batch_size)) * 0.99

    mock_states = State(
        features=NamedArray.unnamed(batch_states),
        a_lo=jnp.zeros((ensemble_size, batch_size, 2)),
        a_hi=jnp.ones((ensemble_size, batch_size, 2)),
        dp=jnp.ones((ensemble_size, batch_size, 1)),
        last_a=jnp.ones((ensemble_size, batch_size, 2)),
    )

    mock_next_states = State(
        features=NamedArray.unnamed(jax.random.normal(state_rng, (ensemble_size, batch_size, state_dim))),
        a_lo=jnp.zeros((ensemble_size, batch_size, 2)),
        a_hi=jnp.ones((ensemble_size, batch_size, 2)),
        dp=jnp.ones((ensemble_size, batch_size, 1)),
        last_a=batch_actions,
    )

    return MockTransition(
        state=mock_states,
        action=batch_actions,
        reward=batch_rewards,
        next_state=mock_next_states,
        gamma=batch_gammas,
    )


def test_update(critic: AdvCritic):
    """Test that update runs successfully and returns updated state and metrics."""
    batch_size = 8
    state_dim = 4
    action_dim = 2
    ensemble_size = critic._cfg.ensemble

    rng = jax.random.PRNGKey(42)
    init_rng, data_rng = jax.random.split(rng)

    # Initialize critic
    state = jnp.zeros(state_dim)
    action = jnp.zeros(action_dim)
    critic_state = critic.init_state(init_rng, state, action)

    # Create mock data
    trans_rng, policy_rng = jax.random.split(data_rng)
    transitions = _create_mock_transitions(trans_rng, ensemble_size, batch_size, state_dim, action_dim)

    policy_actions = jnp.ones((ensemble_size, batch_size, critic._cfg.num_policy_actions, action_dim))
    policy_probs = jax.random.normal(policy_rng, (ensemble_size, batch_size, critic._cfg.num_policy_actions))

    # Perform update
    new_state, metrics = critic.update(critic_state, transitions, policy_actions, policy_probs)

    # Verify state structure
    assert hasattr(new_state, 'params')
    assert hasattr(new_state, 'opt_state')
    chex.assert_tree_shape_prefix(new_state.params, (ensemble_size,))

    # Verify metrics structure
    assert hasattr(metrics, 'loss')
    assert hasattr(metrics, 'v_loss')
    assert hasattr(metrics, 'h_loss')
    assert hasattr(metrics, 'adv_loss')
    assert hasattr(metrics, 'delta')
    assert hasattr(metrics, 'v')
    assert hasattr(metrics, 'h')
    assert hasattr(metrics, 'adv')

    # Check metric shapes
    chex.assert_shape(metrics.loss, (ensemble_size, batch_size))
    chex.assert_shape(metrics.v_loss, (ensemble_size, batch_size))
    chex.assert_shape(metrics.h_loss, (ensemble_size, batch_size))
    chex.assert_shape(metrics.adv_loss, (ensemble_size, batch_size))

    # Check that params have changed
    leaves1 = jax.tree.leaves(critic_state.params)
    leaves2 = jax.tree.leaves(new_state.params)
    params_unchanged = all(starmap(jnp.allclose, zip(leaves1, leaves2, strict=True)))

    assert not params_unchanged, "Parameters should change after update"

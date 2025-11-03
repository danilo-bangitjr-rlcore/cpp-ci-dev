from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from lib_utils.named_array import NamedArray
from pytest_benchmark.fixture import BenchmarkFixture

from lib_agent.actor.percentile_actor import PAConfig, PercentileActor
from lib_agent.buffer.datatypes import State


def _create_test_actor(state_dim: int, action_dim: int, num_samples: int = 256):
    """Create a percentile actor for testing."""
    config = PAConfig(
        name="PA",
        num_samples=num_samples,
        actor_percentile=0.8,
        proposal_percentile=0.5,
        uniform_weight=0.1,
        actor_lr=1e-3,
        proposal_lr=1e-3,
        mu_multiplier=1.0,
        sigma_multiplier=1.0,
        max_action_stddev=1.0,
        sort_noise=0.0,
    )

    return PercentileActor(
        cfg=config,
        state_dim=state_dim,
        action_dim=action_dim,
    )


def test_actor_log_probs(benchmark: BenchmarkFixture):
    """Benchmark action log probability computation."""
    rng = np.random.default_rng(0)
    state_dim, action_dim = 64, 4
    batch_size = 128

    actor = _create_test_actor(state_dim, action_dim, num_samples=256)

    init_rng = jax.random.PRNGKey(42)
    single_state = jnp.array(rng.random((state_dim,)))
    actor_state = actor.init_state(init_rng, single_state)

    state = State(
        features=NamedArray.unnamed(rng.random((state_dim,))),
        a_lo=jnp.ones((action_dim,)) * -1.0,
        a_hi=jnp.ones((action_dim,)) * 1.0,
        dp=jnp.array([1.0]),
        last_a=jnp.zeros((action_dim,)),
    )
    actions = jnp.array(rng.random((batch_size, action_dim)))

    def _inner(actor: PercentileActor, params: Any, state: State, actions: jax.Array):
        for _ in range(20):
            log_probs = actor.get_log_probs(params, state, actions)
            # Force computation
            _ = log_probs.sum()

    benchmark(_inner, actor, actor_state.actor.params, state, actions)

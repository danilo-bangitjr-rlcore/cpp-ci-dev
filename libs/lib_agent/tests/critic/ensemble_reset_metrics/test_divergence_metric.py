from unittest.mock import Mock

import jax
import jax.numpy as jnp
import pytest
from lib_utils.named_array import NamedArray

from lib_agent.actor.actor_protocol import Actor, PolicyState
from lib_agent.buffer.datatypes import State, Transition
from lib_agent.critic.critic_utils import CriticState
from lib_agent.critic.ensemble_reset_metrics.base import EnsembleResetMetricCritic
from lib_agent.critic.ensemble_reset_metrics.divergence import DivergenceMetric, DivergenceMetricConfig


@pytest.fixture
def state_dim() -> int:
    return 10

@pytest.fixture
def action_dim() -> int:
    return 2

@pytest.fixture
def gamma() -> float:
    return 0.99

@pytest.fixture
def sample_transition(state_dim: int, action_dim: int, gamma: float) -> Transition:
    """Create a sample transition for testing."""
    state = State(
        features=NamedArray.unnamed(jnp.ones((state_dim,)) * 0.5),
        a_lo=jnp.zeros((action_dim,)),
        a_hi=jnp.ones((action_dim,)),
        dp=jnp.array([1.0]),
        last_a=jnp.zeros((action_dim,)),
        primitive_held=jnp.array([1.0]),
    )
    action = jnp.ones((action_dim,))
    next_state = State(
        features=NamedArray.unnamed(jnp.ones((state_dim,))),
        a_lo=jnp.zeros((action_dim,)),
        a_hi=jnp.ones((action_dim,)),
        dp=jnp.array([1.0]),
        last_a=action,
        primitive_held=jnp.array([1.0]),
    )

    return Transition(
        state=state,
        action=action,
        n_step_reward=jnp.array(0.0),
        n_step_gamma=jnp.array(gamma),
        next_state=next_state,
    )

@pytest.fixture
def mock_critic() -> EnsembleResetMetricCritic:
    """Create a mock critic for testing."""
    critic = Mock(spec=EnsembleResetMetricCritic)
    critic.forward.return_value.q = jnp.array([[-10.0, -900.0], [20.0, 25.0], [-22.2, -23.4], [0.5, 1.5]])
    return critic

@pytest.fixture
def mock_critic_state() -> CriticState:
    """Create a mock critic state for testing."""
    critic_state = Mock(spec=CriticState)
    critic_state.params = {}
    critic_state.opt_state = {}
    return critic_state

@pytest.fixture
def mock_actor() -> Actor:
    """Create a mock actor for testing."""
    return Mock(spec=Actor)

@pytest.fixture
def mock_actor_state() -> PolicyState:
    """Create a mock actor state for testing."""
    actor_state = Mock(spec=PolicyState)
    actor_state.params = {}
    actor_state.opt_state = {}
    actor_state.group_opt_state = {}
    return actor_state

@pytest.fixture
def divergence_metric_cfg() -> DivergenceMetricConfig:
    cfg = DivergenceMetricConfig()
    cfg.tolerance = 1.0
    cfg.num_actions = 2
    return cfg


@pytest.fixture
def divergence_metric(divergence_metric_cfg: DivergenceMetricConfig, gamma: float) -> DivergenceMetric:
    return DivergenceMetric(divergence_metric_cfg, gamma)


def test_divergence_metric_calculation(
    divergence_metric: DivergenceMetric,
    sample_transition: Transition,
    mock_critic_state: CriticState,
    mock_critic: EnsembleResetMetricCritic,
    mock_actor_state: PolicyState,
    mock_actor: Actor,
):
    """Test the divergence metric calculation."""
    rng = jax.random.PRNGKey(0)
    result = divergence_metric(
        rng,
        sample_transition,
        mock_critic_state,
        mock_critic,
        mock_actor_state,
        mock_actor,
    )

    # min_return = (-8.0 / (1 - 0.99)) - tolerance = -801.0
    # max_return = 0.0 + tolerance = 1.0
    # return_range = 1.0 - (-801.0) = 802.0
    # Q-values from mock critic: [[-10.0, -900.0], [20.0, 25.0], [-22.2, -23.4], [0.5, 1.5]]
    # Diffs: [[0.0, 99.0], [19.0, 24.0], [0.0, 0.0], [0.0, 0.5]]
    # Avg Diffs: [49.5, 21.5, 0.0, 0.25]
    # Divergence Scores: [49.5/802.0, 21.5/802.0, 0.0/802.0, 0.25/802.0]
    # Divergence Scores = [0.06172207, 0.02680898, 0.0, 0.00031172]
    expected = jnp.array([0.06172207, 0.02680898, 0.0, 0.00031172])

    assert jnp.allclose(result, expected, atol=1e-4), f"Expected {expected}, but got {result}"

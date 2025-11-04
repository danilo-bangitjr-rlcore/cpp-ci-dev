import datetime
from collections.abc import Callable

import jax.numpy as jnp
import jax.random as jrandom
import pytest
from lib_agent.buffer.datatypes import DataMode, Step, Trajectory, convert_trajectory_to_transition
from lib_utils.named_array import NamedArray

from corerl.agent.greedy_ac import GreedyAC
from corerl.configs.agent.greedy_ac import GreedyACConfig
from corerl.data_pipeline.pipeline import ColumnDescriptions
from corerl.state import AppState
from tests.infrastructure.config import create_config_with_overrides


@pytest.fixture
def minimal_agent_config() -> GreedyACConfig:
    config = create_config_with_overrides(
        overrides={
            'agent.critic.critic_network.ensemble': 2,
            'agent.critic.buffer.max_size': 1000,
            'agent.critic.buffer.batch_size': 16,
            'agent.policy.buffer.max_size': 1000,
            'agent.policy.buffer.batch_size': 16,
        },
    )
    return config.agent


@pytest.fixture
def sample_trajectories():
    def _factory(n: int, state_dim: int, action_dim: int) -> list[Trajectory]:
        key = jrandom.PRNGKey(42)
        trajectories = []
        start_time = datetime.datetime.now(datetime.UTC)

        for i in range(n):
            key, *subkeys = jrandom.split(key, 10)

            prior_state = NamedArray.unnamed(jrandom.normal(subkeys[0], (state_dim,)))
            action = jrandom.normal(subkeys[1], (action_dim,))
            post_state = NamedArray.unnamed(jrandom.normal(subkeys[2], (state_dim,)))
            reward = jrandom.normal(subkeys[3], ())
            prior_action = jrandom.normal(subkeys[4], (action_dim,))
            gamma = 0.99
            action_lo = jrandom.uniform(subkeys[5], (action_dim,), minval=-1.0, maxval=0.0)
            action_hi = jrandom.uniform(subkeys[6], (action_dim,), minval=0.0, maxval=1.0)
            post_action_lo = jrandom.uniform(subkeys[7], (action_dim,), minval=-1.0, maxval=0.0)
            post_action_hi = jrandom.uniform(subkeys[8], (action_dim,), minval=0.0, maxval=1.0)

            prior_step = Step(
                reward=0.0,
                action=prior_action,
                gamma=gamma,
                state=prior_state,
                action_lo=action_lo,
                action_hi=action_hi,
                dp=True,
                ac=True,
                primitive_held=jnp.ones_like(prior_state),
                timestamp=start_time + datetime.timedelta(seconds=i * 2),
            )

            post_step = Step(
                reward=float(reward),
                action=action,
                gamma=gamma,
                state=post_state,
                action_lo=post_action_lo,
                action_hi=post_action_hi,
                dp=True,
                ac=True,
                primitive_held=jnp.ones_like(post_state),
                timestamp=start_time + datetime.timedelta(seconds=i * 2 + 1),
            )

            trajectory = Trajectory(
                steps=[prior_step, post_step],
                n_step_reward=float(reward),
                n_step_gamma=float(gamma),
            )
            trajectories.append(trajectory)

        return trajectories

    return _factory


@pytest.fixture
def greedy_ac_agent(
    minimal_agent_config: GreedyACConfig,
    dummy_app_state: AppState,
):
    state_dim = 3
    action_dim = 2

    col_desc = ColumnDescriptions(
        action_tags=[],
        state_cols=[f's{i}' for i in range(state_dim)],
        action_cols=[f'a{i}' for i in range(action_dim)],
    )

    return GreedyAC(minimal_agent_config, dummy_app_state, col_desc)


@pytest.fixture
def populated_agent(
    greedy_ac_agent: GreedyAC,
    sample_trajectories: Callable[[int, int, int], list[Trajectory]],
):
    n_transitions = 30

    trajectories = sample_trajectories(n_transitions, greedy_ac_agent.state_dim, greedy_ac_agent.action_dim)
    transitions = [convert_trajectory_to_transition(t) for t in trajectories]

    greedy_ac_agent._actor_buffer.feed(transitions, DataMode.ONLINE)
    greedy_ac_agent.critic_buffer.feed(transitions, DataMode.ONLINE)

    return greedy_ac_agent


@pytest.fixture
def agent_with_training_history(populated_agent: GreedyAC):
    for _ in range(3):
        populated_agent.update_critic()
        populated_agent.update_actor()

    return populated_agent

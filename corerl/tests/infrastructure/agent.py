from collections.abc import Callable

import jax.numpy as jnp
import jax.random as jrandom
import pytest
from lib_agent.buffer.datatypes import DataMode, JaxTransition
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
def sample_jax_transitions():
    def _factory(n: int, state_dim: int, action_dim: int) -> list[JaxTransition]:
        key = jrandom.PRNGKey(42)
        transitions = []

        for _ in range(n):
            key, *subkeys = jrandom.split(key, 10)

            state = NamedArray.unnamed(jrandom.normal(subkeys[0], (state_dim,)))
            action = jrandom.normal(subkeys[1], (action_dim,))
            next_state = NamedArray.unnamed(jrandom.normal(subkeys[2], (state_dim,)))
            reward = jrandom.normal(subkeys[3], ())
            last_action = jrandom.normal(subkeys[4], (action_dim,))
            gamma = jnp.array(0.99)
            action_lo = jrandom.uniform(subkeys[5], (action_dim,), minval=-1.0, maxval=0.0)
            action_hi = jrandom.uniform(subkeys[6], (action_dim,), minval=0.0, maxval=1.0)
            next_action_lo = jrandom.uniform(subkeys[7], (action_dim,), minval=-1.0, maxval=0.0)
            next_action_hi = jrandom.uniform(subkeys[8], (action_dim,), minval=0.0, maxval=1.0)
            dp = jnp.array(True)
            next_dp = jnp.array(True)

            transition = JaxTransition(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                last_action=last_action,
                gamma=gamma,
                action_lo=action_lo,
                action_hi=action_hi,
                next_action_lo=next_action_lo,
                next_action_hi=next_action_hi,
                dp=dp,
                next_dp=next_dp,
                n_step_reward=reward,
                n_step_gamma=gamma,
            )
            transitions.append(transition)

        return transitions

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
    sample_jax_transitions: Callable[[int, int, int], list[JaxTransition]],
):
    n_transitions = 30

    transitions = sample_jax_transitions(n_transitions, greedy_ac_agent.state_dim, greedy_ac_agent.action_dim)

    greedy_ac_agent._actor_buffer.feed(transitions, DataMode.ONLINE)
    greedy_ac_agent.critic_buffer.feed(transitions, DataMode.ONLINE)

    return greedy_ac_agent


@pytest.fixture
def agent_with_training_history(populated_agent: GreedyAC):
    for _ in range(3):
        populated_agent.update_critic()
        populated_agent.update_actor()

    return populated_agent

import jax
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from corerl.agent.greedy_ac import GreedyAC
from corerl.config import MainConfig
from corerl.data_pipeline.constructors.ac import ActionConstructor
from corerl.data_pipeline.constructors.preprocess import Preprocessor
from corerl.data_pipeline.constructors.sc import StateConstructor
from corerl.data_pipeline.pipeline import ColumnDescriptions
from corerl.state import AppState


@pytest.fixture
def dummy_agent(dummy_app_state: AppState, basic_config: MainConfig):
    tags = basic_config.pipeline.tags
    state_tags = StateConstructor.state_configs(tags)
    action_tags = ActionConstructor.action_configs(tags)

    state_constructor = StateConstructor(dummy_app_state, tags, basic_config.pipeline.state_constructor)
    action_constructor = ActionConstructor(dummy_app_state, tags, Preprocessor(tags))
    state_cols = state_constructor.columns
    action_cols = action_constructor.columns
    col_desc = ColumnDescriptions(
        state_tags=state_tags,
        action_tags=action_tags,
        state_cols=state_cols,
        action_cols=action_cols,
    )
    agent = GreedyAC(basic_config.agent, dummy_app_state, col_desc)
    return agent, len(state_cols), len(action_cols)


def test_critic_value_query_benchmark(benchmark: BenchmarkFixture, dummy_agent: tuple[GreedyAC, int, int]):
    agent, state_dim, action_dim = dummy_agent
    state = jax.numpy.zeros(state_dim)
    action = jax.numpy.zeros(action_dim)

    def _inner(agent: GreedyAC, state: jax.Array, action: jax.Array):
        return agent.get_active_values(state, action)

    result = benchmark(_inner, agent, state, action)
    assert result is not None

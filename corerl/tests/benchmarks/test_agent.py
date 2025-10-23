
import jax
import pandas as pd
import pytest
from lib_agent.buffer.datatypes import State
from lib_utils.named_array import NamedArray
from pytest_benchmark.fixture import BenchmarkFixture

from corerl.agent.greedy_ac import GreedyAC, PipelineReturn
from corerl.config import MainConfig
from corerl.data_pipeline.constructors.ac import ActionConstructor
from corerl.data_pipeline.constructors.preprocess import Preprocessor
from corerl.data_pipeline.constructors.sc import StateConstructor
from corerl.data_pipeline.pipeline import ColumnDescriptions, DataMode, Pipeline
from corerl.state import AppState


@pytest.fixture
def dummy_agent(dummy_app_state: AppState, basic_config: MainConfig):
    tags = basic_config.pipeline.tags
    action_tags = ActionConstructor.action_configs(tags)

    state_constructor = StateConstructor(dummy_app_state, tags, basic_config.pipeline.state_constructor)
    action_constructor = ActionConstructor(dummy_app_state, tags, Preprocessor(tags))
    state_cols = state_constructor.columns
    action_cols = action_constructor.columns
    col_desc = ColumnDescriptions(
        action_tags=action_tags,
        state_cols=state_cols,
        action_cols=action_cols,
    )
    agent = GreedyAC(basic_config.agent, dummy_app_state, col_desc)
    return agent, len(state_cols), len(action_cols)


def test_critic_value_query_benchmark(benchmark: BenchmarkFixture, dummy_agent: tuple[GreedyAC, int, int]):
    agent, state_dim, action_dim = dummy_agent
    state = NamedArray.unnamed(jax.numpy.zeros(state_dim))
    action = jax.numpy.zeros(action_dim)

    def _inner(agent: GreedyAC, state: NamedArray, action: jax.Array):
        return agent.get_active_values(state, action)

    result = benchmark(_inner, agent, state, action)
    assert result is not None


def test_actor_query_benchmark(benchmark: BenchmarkFixture, dummy_agent: tuple[GreedyAC, int, int]):
    agent, state_dim, action_dim = dummy_agent
    state = State(
        features=NamedArray.unnamed(jax.numpy.zeros((1, state_dim))),
        a_lo=jax.numpy.zeros((1, action_dim)),
        a_hi=jax.numpy.ones((1, action_dim)),
        dp=jax.numpy.ones((1, 1), dtype=bool),
        last_a=jax.numpy.zeros((1, action_dim)),
    )

    def _inner(agent: GreedyAC, state: State):
        return agent.get_action_interaction(state)

    result = benchmark(_inner, agent, state)
    assert result is not None


def test_agent_ingress_benchmark(
    benchmark: BenchmarkFixture,
    dummy_agent: tuple[GreedyAC, int, int],
    dummy_pipeline: Pipeline,
    fake_pipeline_data: pd.DataFrame,
):
    agent, _, _ = dummy_agent
    pr = dummy_pipeline(fake_pipeline_data, data_mode=DataMode.ONLINE)

    def _inner(agent: GreedyAC, pr: PipelineReturn):
        agent.update_buffer(pr)

    benchmark(_inner, agent, pr)


def test_agent_update_benchmark(
    benchmark: BenchmarkFixture,
    dummy_agent: tuple[GreedyAC, int, int],
    dummy_pipeline: Pipeline,
    fake_pipeline_data: pd.DataFrame,
):
    agent, _, _ = dummy_agent
    pr = dummy_pipeline(fake_pipeline_data, data_mode=DataMode.ONLINE)
    agent.update_buffer(pr)

    def _inner(agent: GreedyAC):
        return agent.update()

    result = benchmark(_inner, agent)
    assert result is not None


def test_critic_update_benchmark(
    benchmark: BenchmarkFixture,
    dummy_agent: tuple[GreedyAC, int, int],
    dummy_pipeline: Pipeline,
    fake_pipeline_data: pd.DataFrame,
):
    agent, _, _ = dummy_agent
    pr = dummy_pipeline(fake_pipeline_data, data_mode=DataMode.ONLINE)
    agent.update_buffer(pr)

    def _inner(agent: GreedyAC):
        return agent.update_critic()

    result = benchmark(_inner, agent)
    assert result is not None


def test_actor_update_benchmark(
    benchmark: BenchmarkFixture,
    dummy_agent: tuple[GreedyAC, int, int],
    dummy_pipeline: Pipeline,
    fake_pipeline_data: pd.DataFrame,
):
    agent, _, _ = dummy_agent
    pr = dummy_pipeline(fake_pipeline_data, data_mode=DataMode.ONLINE)
    agent.update_buffer(pr)

    def _inner(agent: GreedyAC):
        return agent.update_actor()

    result = benchmark(_inner, agent)
    assert result is not None

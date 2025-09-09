import datetime as dt

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from lib_defs.config_defs.tag_config import TagType
from lib_sql.inspection import table_exists
from sqlalchemy import Engine

from corerl.agent.greedy_ac import GreedyAC
from corerl.config import MainConfig
from corerl.data_pipeline.datatypes import DataMode, Step, Transition
from corerl.data_pipeline.db.data_reader import TagDBConfig
from corerl.data_pipeline.db.data_writer import DataWriter
from corerl.data_pipeline.pipeline import Pipeline, PipelineReturn
from corerl.data_pipeline.transforms.norm import NormalizerConfig
from corerl.eval.evals import EvalsTable
from corerl.eval.metrics import create_metrics_writer
from corerl.messages.event_bus import DummyEventBus
from corerl.offline.utils import load_offline_transitions, offline_rl_from_buffer
from corerl.state import AppState


def make_step(
    reward: float,
    action: jax.Array,
    gamma: float,
    state: jax.Array,
    dp: bool, # decision point
    ac: bool, # action change
):
    return Step(
        reward=reward,
        action=action,
        gamma=gamma,
        state=state,
        action_lo=jnp.zeros_like(action),
        action_hi=jnp.ones_like(action),
        dp=dp,
        ac=ac,
    )


@pytest.fixture()
def data_writer(offline_cfg: MainConfig, test_db_config: TagDBConfig):
    data_writer = DataWriter(cfg=test_db_config)

    steps = 5
    obs_period = offline_cfg.interaction.obs_period

    # Generate timestamps
    start_time = dt.datetime(year=2023, month=7, day=13, hour=10, minute=0, tzinfo=dt.UTC)
    offline_cfg.offline.offline_start_time = start_time
    # The index of the first row produced by the data reader given start_time will be
    # obs_period after start_time.
    first_step = start_time + obs_period

    step_timestamps = [
        first_step + obs_period * i
        for i in range(steps)
    ]

    # Generate tag data and write to tsdb
    steps_per_decision = int(
        offline_cfg.interaction.action_period.total_seconds() / offline_cfg.interaction.obs_period.total_seconds(),
    )
    for i in range(steps):
        for tag_cfg in offline_cfg.pipeline.tags:
            tag = tag_cfg.name
            if tag_cfg.type == TagType.ai_setpoint:
                val = int(i / steps_per_decision) % 2
            else:
                val = i

            data_writer.write(timestamp=step_timestamps[i], name=tag, val=val)

    data_writer.blocking_sync()

    yield data_writer

    data_writer.close()

@pytest.fixture
def offline_pipeout(offline_cfg: MainConfig, dummy_app_state: AppState, data_writer: DataWriter):
    """
    Generate offline data for tests and return the OfflineTraining object
    """
    # Produce offline transitions

    pipeline = Pipeline(dummy_app_state, offline_cfg.pipeline)
    dummy_app_state.cfg = offline_cfg
    pipeout, _ =  load_offline_transitions(dummy_app_state, pipeline)
    assert pipeout is not None
    return pipeout

def test_load_offline_transitions(offline_cfg: MainConfig, offline_pipeout: PipelineReturn):
    """
    Ensure the test data generated in the 'offline_trainer' fixture was written to TSDB,
    read from TSDB, and that the correct transitions were produced by the data pipeline
    """
    created_transitions = offline_pipeout.transitions
    assert created_transitions is not None

    # Expected transitions
    gamma = offline_cfg.agent.gamma
    step_0 = make_step(reward=1.0, action=jnp.array([0.0]), gamma=gamma, state=jnp.array([0, 1, 0]), dp=False, ac=False)
    step_1 = make_step(reward=1.0, action=jnp.array([0.0]), gamma=gamma, state=jnp.array([1, 1, 0]), dp=True,  ac=False)
    step_2 = make_step(reward=1.0, action=jnp.array([1.0]), gamma=gamma, state=jnp.array([2, 1, 0]), dp=False, ac=True)
    step_3 = make_step(reward=0.0, action=jnp.array([1.0]), gamma=gamma, state=jnp.array([3, 1, 0]), dp=True,  ac=False)
    step_4 = make_step(reward=0.0, action=jnp.array([0.0]), gamma=gamma, state=jnp.array([4, 1, 0]), dp=False, ac=True)
    expected_transitions = [Transition([step_0, step_1], 1.0, gamma),
                            Transition([step_1, step_2], 1.0, gamma),
                            Transition([step_2, step_3], 0.0, gamma),
                            Transition([step_1, step_2, step_3], 1.0, gamma**2.0),
                            Transition([step_3, step_4], 0.0, gamma)]

    assert len(created_transitions) == len(expected_transitions)
    for i, created_transition in enumerate(created_transitions):
        assert created_transition == expected_transitions[i]

@pytest.mark.skip(reason="failing on master, requires further investigation")
def test_offline_training(
    offline_cfg: MainConfig,
    offline_pipeout: PipelineReturn,
    tsdb_engine: Engine,
    dummy_app_state: AppState,
):
    """
    Ensure the agent's critic loss decreases over the test transitions produced by the 'offline_trainer' fixture.
    Make sure the enabled evaluators write to the metrics table and/or evals table
    """
    app_state = AppState(
        cfg=offline_cfg,
        metrics=create_metrics_writer(offline_cfg.metrics),
        evals=EvalsTable(offline_cfg.evals),
        event_bus=DummyEventBus(),
    )

    pipeline = Pipeline(dummy_app_state, offline_cfg.pipeline)
    col_desc = pipeline.column_descriptions
    agent = GreedyAC(offline_cfg.agent, app_state, col_desc)

    agent.update_buffer(offline_pipeout)

    critic_losses = offline_rl_from_buffer(agent, offline_cfg.offline.offline_steps)
    first_loss = critic_losses[0]
    last_loss = critic_losses[-1]

    assert last_loss < first_loss

    # ensure metrics and evals tables exist
    app_state.metrics.close()
    app_state.evals.close()
    assert table_exists(tsdb_engine, 'metrics')
    assert table_exists(tsdb_engine, 'evals')

    with tsdb_engine.connect() as conn:
        # Ensure Monte-Carlo evaluator writes state-value, observed action-value, and partial returns
        # twice to metrics table (partial return horizon = math.ceil(np.log(1.0 - self.precision) / np.log(self.gamma))
        metrics = pd.read_sql_table('metrics', con=conn)
        state_v_entries = metrics.loc[metrics["metric"] == "state_v_0"]
        observed_a_q_entries = metrics.loc[metrics["metric"] == "observed_a_q_0"]
        partial_return_entries = metrics.loc[metrics["metric"] == "partial_return_0"]
        assert len(state_v_entries) == len(observed_a_q_entries) == len(partial_return_entries) == 2


def test_regression_normalizer_bounds_reset(offline_cfg: MainConfig, dummy_app_state: AppState):
    normalizer = NormalizerConfig(from_data=True)

    # add normalizer to pipeline config
    offline_cfg.pipeline.tags[1].state_constructor = [normalizer]
    pipeline = Pipeline(dummy_app_state, offline_cfg.pipeline)

    # trigger pipeline to test dummy data
    _ = pipeline.column_descriptions

    # create test data and run through pipeline
    dates = [dt.datetime(2024, 1, 1, 1, i, tzinfo=dt.UTC) for i in range(5)]
    df = pd.DataFrame({
        "Tag_1":  [0.1, -0.1, 0, 0, 0],
        "Action": [  0,    0, 1, 1, 0],
        "reward": [  0,    0, 0, 0, 0],
    }, index=pd.DatetimeIndex(dates))

    # check if tag is normalized using [-0.1, 0.1] as bounds
    # prior implementation would mistakenly use [-0.1, 1] as bounds
    pr = pipeline(df, data_mode=DataMode.OFFLINE)

    assert np.all(pr.df['Tag_1_norm'] == [1., 0, 0.5, 0.5, 0.5])

def test_offline_start_end(offline_cfg: MainConfig, dummy_app_state: AppState, data_writer: DataWriter):
    obs_period = offline_cfg.interaction.obs_period
    start_time = dt.datetime(year=2023, month=7, day=13, hour=10, minute=0, tzinfo=dt.UTC)
    # The index of the first row produced by the data reader given start_time will be
    # obs_period after start_time.
    first_step = start_time + obs_period

    # Produce offline transitions
    offline_cfg.offline.offline_start_time = first_step
    offline_cfg.offline.offline_end_time = first_step + 2 * obs_period

    dummy_app_state.cfg = offline_cfg
    pipeline = Pipeline(dummy_app_state, offline_cfg.pipeline)
    offline_pipeout, _ = load_offline_transitions(dummy_app_state, pipeline)

    # Since start_time and end_time are specified,
    # make sure PipelineReturn's df spans (end_time - start_time) / obs_period entries
    assert isinstance(offline_pipeout, PipelineReturn)
    df = offline_pipeout.df
    assert len(df) == (offline_cfg.offline.offline_end_time - offline_cfg.offline.offline_start_time) / obs_period


def test_test_split(offline_cfg: MainConfig, dummy_app_state: AppState, data_writer: DataWriter):
    """
    Tests ability to split offline transitions into a train and test set.
    """
    offline_cfg.offline.test_split = 0.2
    dummy_app_state.cfg = offline_cfg
    pipeline = Pipeline(dummy_app_state, offline_cfg.pipeline)
    offline_pipeout, test_transitions = load_offline_transitions(dummy_app_state, pipeline)
    assert offline_pipeout is not None
    assert offline_pipeout.transitions is not None
    assert test_transitions is not None
    assert len(offline_pipeout.transitions) == 4
    assert len(test_transitions) == 1

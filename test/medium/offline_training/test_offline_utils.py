import datetime as dt

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine
from torch import Tensor

from corerl.agent.greedy_ac import GreedyAC
from corerl.config import MainConfig
from corerl.configs.loader import direct_load_config
from corerl.data_pipeline.datatypes import DataMode, Step, Transition
from corerl.data_pipeline.db.data_reader import TagDBConfig
from corerl.data_pipeline.db.data_writer import DataWriter
from corerl.data_pipeline.pipeline import Pipeline, PipelineReturn
from corerl.data_pipeline.transforms.norm import NormalizerConfig
from corerl.environment.async_env.async_env import DepAsyncEnvConfig
from corerl.eval.actor_critic import PlotInfoBatch
from corerl.eval.evals import EvalDBConfig, evals_group
from corerl.eval.metrics import MetricsDBConfig, metrics_group
from corerl.eval.xy_metrics import XYTable
from corerl.messages.event_bus import EventBus
from corerl.offline.utils import OfflineTraining
from corerl.sql_logging.sql_logging import table_exists
from corerl.state import AppState


@pytest.fixture()
def test_db_config(tsdb_engine: Engine, tsdb_tmp_db_name: str) -> TagDBConfig:
    port = tsdb_engine.url.port
    assert port is not None

    db_cfg = TagDBConfig(
        drivername="postgresql+psycopg2",
        username="postgres",
        password="password",
        ip="localhost",
        port=port,
        db_name=tsdb_tmp_db_name,
        table_name="tags",
        table_schema='public',
    )

    return db_cfg

@pytest.fixture()
def data_writer(test_db_config: TagDBConfig):
    data_writer = DataWriter(cfg=test_db_config)

    yield data_writer

    data_writer.close()

@pytest.fixture(scope="function")
def offline_cfg(test_db_config: TagDBConfig) -> MainConfig:
    cfg = direct_load_config(
        MainConfig,
        base='test/medium/offline_training/assets',
        config_name='offline_config.yaml',
    )
    assert isinstance(cfg, MainConfig)

    cfg.agent.critic.buffer.online_weight = 0.0
    cfg.agent.policy.buffer.online_weight = 0.0

    assert isinstance(cfg.env, DepAsyncEnvConfig)
    cfg.env.db = test_db_config

    assert isinstance(cfg.metrics, MetricsDBConfig)
    cfg.metrics.port = test_db_config.port
    cfg.metrics.db_name = test_db_config.db_name

    assert isinstance(cfg.evals, EvalDBConfig)
    cfg.evals.port = test_db_config.port
    cfg.evals.db_name = test_db_config.db_name

    return cfg

@pytest.fixture
def offline_trainer(offline_cfg: MainConfig, data_writer: DataWriter, dummy_app_state: AppState) -> OfflineTraining:
    """
    Generate offline data for tests and return the OfflineTraining object
    """
    steps = 5
    obs_period = offline_cfg.interaction.obs_period

    # Generate timestamps
    step_timestamps = []
    start_time = dt.datetime(year=2023, month=7, day=13, hour=10, minute=0, tzinfo=dt.timezone.utc)
    offline_cfg.experiment.offline_start_time = start_time
    # The index of the first row produced by the data reader given start_time will be
    # obs_period after start_time.
    first_step = start_time + obs_period

    for i in range(steps):
        step_timestamps.append(first_step + obs_period * i)

    # Generate tag data and write to tsdb
    steps_per_decision = int(
        offline_cfg.interaction.action_period.total_seconds() / offline_cfg.interaction.obs_period.total_seconds()
    )
    for i in range(steps):
        for tag_cfg in offline_cfg.pipeline.tags:
            tag = tag_cfg.name
            if tag_cfg.action_constructor is not None:
                val = int(i / steps_per_decision) % 2
            else:
                val = i

            data_writer.write(timestamp=step_timestamps[i], name=tag, val=val)

    data_writer.blocking_sync()

    # Produce offline transitions
    offline_training = OfflineTraining(offline_cfg)
    pipeline = Pipeline(dummy_app_state, offline_cfg.pipeline)
    offline_training.load_offline_transitions(pipeline)

    return offline_training

def test_load_offline_transitions(offline_cfg: MainConfig, offline_trainer: OfflineTraining):
    """
    Ensure the test data generated in the 'offline_trainer' fixture was written to TSDB,
    read from TSDB, and that the correct transitions were produced by the data pipeline
    """
    assert offline_trainer.pipeline_out is not None
    assert offline_trainer.pipeline_out.transitions is not None
    created_transitions = offline_trainer.pipeline_out.transitions

    # Expected transitions
    gamma = offline_cfg.experiment.gamma
    step_0 = Step(reward=1.0, action=Tensor([0.0]), gamma=gamma, state=Tensor([0.0]), dp=False, ac=False)
    step_1 = Step(reward=1.0, action=Tensor([0.0]), gamma=gamma, state=Tensor([1.0]), dp=True,  ac=False) # dp
    step_2 = Step(reward=1.0, action=Tensor([1.0]), gamma=gamma, state=Tensor([2.0]), dp=False, ac=True)  # ac
    step_3 = Step(reward=0.0, action=Tensor([1.0]), gamma=gamma, state=Tensor([3.0]), dp=True,  ac=False) # dp
    step_4 = Step(reward=0.0, action=Tensor([0.0]), gamma=gamma, state=Tensor([4.0]), dp=False, ac=True)  # ac
    expected_transitions = [Transition([step_0, step_1], 1.0, gamma),
                            Transition([step_1, step_2], 1.0, gamma),
                            Transition([step_2, step_3], 0.0, gamma),
                            Transition([step_1, step_2, step_3], 1.0, gamma**2.0),
                            Transition([step_3, step_4], 0.0, gamma)]

    assert len(created_transitions) == len(expected_transitions)
    for i in range(len(created_transitions)):
        assert created_transitions[i] == expected_transitions[i]

def test_offline_training(
    offline_cfg: MainConfig,
    offline_trainer: OfflineTraining,
    tsdb_engine: Engine,
    dummy_app_state: AppState,
):
    """
    Ensure the agent's critic loss decreases over the test transitions produced by the 'offline_trainer' fixture.
    Make sure the enabled evaluators write to the metrics table and/or evals table
    """
    app_state = AppState(
        cfg=offline_cfg,
        metrics=metrics_group.dispatch(offline_cfg.metrics),
        xy_metrics=XYTable(offline_cfg.xy_metrics),
        evals=evals_group.dispatch(offline_cfg.evals),
        event_bus=EventBus(offline_cfg.event_bus, offline_cfg.env),
    )

    pipeline = Pipeline(dummy_app_state, offline_cfg.pipeline)
    col_desc = pipeline.column_descriptions
    agent = GreedyAC(offline_cfg.agent, app_state, col_desc)

    # Offline training
    critic_losses = offline_trainer.train(app_state, agent, pipeline, col_desc)
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

        # Ensure Actor-Critic evaluator writes an entry to the evals table
        ac_cfg = offline_cfg.eval_cfgs.actor_critic
        evals = pd.read_sql_table('evals', con=conn)
        ac_eval_rows = evals.loc[evals["evaluator"] == "actor-critic_0"]
        assert len(ac_eval_rows) == len(offline_cfg.experiment.offline_eval_iters)
        for i in range(len(ac_eval_rows)):
            ac_out = PlotInfoBatch.model_validate_json(ac_eval_rows["value"].iloc[i])
            assert len(ac_out.states) == ac_cfg.num_test_states
            for test_state in ac_out.states:
                for action_plot_info in test_state.a_dims:
                    pdf_x_range = np.array(action_plot_info.pdf.x_range)
                    pdfs = np.array(action_plot_info.pdf.pdfs)
                    direct_critic_x_range = np.array(action_plot_info.direct_critic.x_range)
                    qs = np.array(action_plot_info.direct_critic.q_vals)
                    assert pdf_x_range.shape == (ac_cfg.num_uniform_actions,)
                    assert pdfs.shape == (ac_cfg.critic_samples, ac_cfg.num_uniform_actions)
                    assert direct_critic_x_range.shape == (ac_cfg.num_uniform_actions,)
                    assert qs.shape == (ac_cfg.critic_samples, ac_cfg.num_uniform_actions)

def test_regression_normalizer_bounds_reset(offline_cfg: MainConfig, dummy_app_state: AppState):
    normalizer = NormalizerConfig(from_data=True)

    # add normalizer to pipeline config
    offline_cfg.pipeline.tags[1].state_constructor = [normalizer]
    pipeline = Pipeline(dummy_app_state, offline_cfg.pipeline)

    # trigger pipeline to test dummy data
    _ = pipeline.column_descriptions

    # create test data and run through pipeline
    dates = [dt.datetime(2024, 1, 1, 1, i, tzinfo=dt.timezone.utc) for i in range(5)]
    df = pd.DataFrame({
        "Tag_1":  [0.1, -0.1, 0, 0, 0],
        "Action": [  0,    0, 1, 1, 0],
        "reward": [  0,    0, 0, 0, 0],
    }, index=pd.DatetimeIndex(dates))

    # check if tag is normalized using [-0.1, 0.1] as bounds
    # prior implementation would mistakenly use [-0.1, 1] as bounds
    pr = pipeline(df, data_mode=DataMode.OFFLINE)

    assert np.all(pr.df['Tag_1_norm'] == [1., 0, 0.5, 0.5, 0.5])

def test_offline_start_end(offline_cfg: MainConfig, data_writer: DataWriter, dummy_app_state: AppState):
    steps = 5
    obs_period = offline_cfg.interaction.obs_period

    # Generate timestamps
    step_timestamps = []
    start_time = dt.datetime(year=2023, month=7, day=13, hour=10, minute=0, tzinfo=dt.timezone.utc)
    # The index of the first row produced by the data reader given start_time will be
    # obs_period after start_time.
    first_step = start_time + obs_period

    for i in range(steps):
        step_timestamps.append(first_step + obs_period * i)

    # Generate tag data and write to tsdb
    steps_per_decision = int(
        offline_cfg.interaction.action_period.total_seconds() / offline_cfg.interaction.obs_period.total_seconds()
    )
    for i in range(steps):
        for tag_cfg in offline_cfg.pipeline.tags:
            tag = tag_cfg.name
            if tag_cfg.action_constructor is not None:
                val = int(i / steps_per_decision) % 2
            else:
                val = i

            data_writer.write(timestamp=step_timestamps[i], name=tag, val=val)

    data_writer.blocking_sync()

    # Produce offline transitions
    offline_cfg.experiment.offline_start_time = first_step
    offline_cfg.experiment.offline_end_time = first_step + 2 * obs_period
    offline_training = OfflineTraining(offline_cfg)
    pipeline = Pipeline(dummy_app_state, offline_cfg.pipeline)
    offline_training.load_offline_transitions(pipeline)

    # Since start_time and end_time are specified,
    # make sure PipelineReturn's df spans (end_time - start_time) / obs_period entries
    assert isinstance(offline_training.pipeline_out, PipelineReturn)
    df = offline_training.pipeline_out.df
    assert len(df) == (offline_cfg.experiment.offline_end_time - offline_cfg.experiment.offline_start_time) / obs_period

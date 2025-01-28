import datetime as dt
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from docker.models.containers import Container
from sqlalchemy import Engine
from torch import Tensor

from corerl.agent.factory import init_agent
from corerl.agent.greedy_ac import GreedyACConfig
from corerl.component.actor.network_actor import NetworkActorConfig
from corerl.component.buffer.uniform import UniformReplayBufferConfig
from corerl.component.critic.ensemble_critic import EnsembleCriticConfig
from corerl.component.optimizers.torch_opts import AdamConfig
from corerl.config import MainConfig
from corerl.data_pipeline.all_the_time import AllTheTimeTCConfig
from corerl.data_pipeline.constructors.sc import SCConfig
from corerl.data_pipeline.datatypes import CallerCode, Step, Transition
from corerl.data_pipeline.db.data_reader import TagDBConfig
from corerl.data_pipeline.db.data_writer import DataWriter
from corerl.data_pipeline.pipeline import Pipeline, PipelineConfig
from corerl.data_pipeline.state_constructors.countdown import CountdownConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transforms import LessThanConfig, NullConfig
from corerl.data_pipeline.transforms.norm import NormalizerConfig
from corerl.data_pipeline.transition_filter import TransitionFilterConfig
from corerl.eval.config import EvalConfig
from corerl.eval.monte_carlo import MonteCarloEvalConfig
from corerl.eval.writer import metrics_group, MetricsDBConfig
from corerl.experiment.config import ExperimentConfig
from corerl.messages.event_bus import EventBus
from corerl.offline.utils import OfflineTraining
from corerl.sql_logging.sql_logging import table_exists
from corerl.state import AppState
from test.infrastructure.utils.docker import init_docker_container  # noqa: F401


@pytest.fixture(scope="module")
def test_db_config() -> TagDBConfig:
    db_cfg = TagDBConfig(
        drivername="postgresql+psycopg2",
        username="postgres",
        password="password",
        ip="localhost",
        port=5432,  # default is 5432, but we want to use different port for test db
        db_name="offline_test",
        table_name="tags",
    )

    return db_cfg

@pytest.fixture(scope="module")
def data_writer(tsdb_container: Container, test_db_config: TagDBConfig) -> Generator[DataWriter, None, None]: # noqa: F811, E501
    data_writer = DataWriter(cfg=test_db_config)

    yield data_writer

    data_writer.close()

@pytest.fixture(scope="function")
def offline_cfg(test_db_config: TagDBConfig) -> MainConfig:
    obs_period = dt.timedelta(seconds=60)
    action_period = 2*obs_period
    seed = 0

    cfg = MainConfig(
        metrics=MetricsDBConfig(
            enabled=True,
        ),
        eval=EvalConfig(
            monte_carlo=MonteCarloEvalConfig(
                enabled=True,
                precision=0.2,
                gamma=0.9
            )
        ),
        agent=GreedyACConfig(
            actor=NetworkActorConfig(
                buffer=UniformReplayBufferConfig(
                    seed=seed
                )
            ),
            critic=EnsembleCriticConfig(
                buffer=UniformReplayBufferConfig(
                    seed=seed
                ),
                critic_optimizer=AdamConfig(
                    lr=0.0001,
                    weight_decay=0.1
                )
            )
        ),
        experiment=ExperimentConfig(
            gamma=0.9,
            offline_steps=100,
            offline_eval_iters=[0]
        ),
        pipeline=PipelineConfig(
            tags=[
                TagConfig(
                    name="Action",
                    preprocess=[],
                    state_constructor=[NullConfig()],
                    action_constructor=[],
                    operating_range=(0.0, 1.0)
                ),
                TagConfig(
                    name="Tag_1",
                    preprocess=[],
                    reward_constructor=[
                        LessThanConfig(threshold=3),
                    ],
                ),
                TagConfig(
                    name="reward",
                    preprocess=[],
                    state_constructor=[NullConfig()],
                ),
            ],
            db=test_db_config,
            obs_period=obs_period,
            action_period=action_period,
            state_constructor=SCConfig(
                defaults=[],
                countdown=CountdownConfig(
                    action_period=action_period,
                    obs_period=obs_period
                ),
            ),
            transition_creator=AllTheTimeTCConfig(
                gamma=0.9,
                min_n_step=1,
                max_n_step=2,
            ),
            transition_filter=TransitionFilterConfig(
                filters=[
                    'only_no_action_change',
                    'no_nan'
                ]
            )
        )
    )

    return cfg

@pytest.fixture
def offline_trainer(offline_cfg: MainConfig) -> OfflineTraining:
    return OfflineTraining(offline_cfg)

def generate_offline_data(offline_cfg: MainConfig,
                          offline_trainer: OfflineTraining,
                          data_writer: DataWriter,
                          steps: int = 5) -> list[Transition]:
    obs_period = offline_cfg.pipeline.obs_period

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
        offline_cfg.pipeline.action_period.total_seconds() / offline_cfg.pipeline.obs_period.total_seconds()
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
    pipeline = Pipeline(offline_cfg.pipeline)
    offline_trainer.load_offline_transitions(pipeline, start_time=start_time)

    assert offline_trainer.pipeline_out is not None
    assert offline_trainer.pipeline_out.transitions is not None

    return offline_trainer.pipeline_out.transitions

def test_load_offline_transitions(offline_cfg: MainConfig, offline_trainer: OfflineTraining, data_writer: DataWriter):
    """
    Generate a few offline time steps, write them to TSDB, read the data from TSDB into a dataframe,
    pass data through the 'Anytime' data pipeline, and ensure the correct transitions are produced
    """
    steps = 5

    created_transitions = generate_offline_data(offline_cfg, offline_trainer, data_writer, steps)

    # Expected transitions
    gamma = offline_cfg.experiment.gamma
    step_0 = Step(reward=1.0, action=Tensor([0.0]), gamma=gamma, state=Tensor([0.0]), dp=True)
    step_1 = Step(reward=1.0, action=Tensor([0.0]), gamma=gamma, state=Tensor([1.0]), dp=False)
    step_2 = Step(reward=1.0, action=Tensor([1.0]), gamma=gamma, state=Tensor([2.0]), dp=True)
    step_3 = Step(reward=0.0, action=Tensor([1.0]), gamma=gamma, state=Tensor([3.0]), dp=False)
    step_4 = Step(reward=0.0, action=Tensor([0.0]), gamma=gamma, state=Tensor([4.0]), dp=True)
    expected_transitions = [Transition([step_0, step_1], 1.0, gamma),
                            Transition([step_1, step_2], 1.0, gamma),
                            Transition([step_2, step_3], 0.0, gamma),
                            Transition([step_1, step_2, step_3], 1.0, gamma**2.0),
                            Transition([step_3, step_4], 0.0, gamma)]

    assert len(created_transitions) == len(expected_transitions)
    for i in range(len(created_transitions)):
        assert created_transitions[i] == expected_transitions[i]

def test_offline_training(offline_cfg: MainConfig,
                          offline_trainer: OfflineTraining,
                          data_writer: DataWriter,
                          tsdb_engine: Engine,
                          tsdb_tmp_db_name: str):
    """
    Generate a few offline time steps, write them to TSDB, read the data from TSDB into a dataframe,
    pass data through the 'Anytime' data pipeline, train an agent on the produced transitions,
    and ensure the critic's training loss decreases
    """
    assert isinstance(offline_cfg.metrics, MetricsDBConfig)
    assert isinstance(tsdb_engine.url.port, int)
    offline_cfg.metrics.port = tsdb_engine.url.port
    offline_cfg.metrics.db_name = tsdb_tmp_db_name

    steps = 5

    generate_offline_data(offline_cfg, offline_trainer, data_writer, steps)

    app_state = AppState(
        metrics=metrics_group.dispatch(offline_cfg.metrics),
        event_bus=EventBus(offline_cfg.event_bus, offline_cfg.env),
    )

    pipeline = Pipeline(offline_cfg.pipeline)
    col_desc = pipeline.column_descriptions
    agent = init_agent(offline_cfg.agent, app_state, col_desc)

    # Offline training
    critic_losses = offline_trainer.train(app_state, agent)
    first_loss = critic_losses[0]
    last_loss = critic_losses[-1]

    assert last_loss < first_loss

    # ensure metrics table exists
    assert table_exists(tsdb_engine, 'metrics')

    # Ensure Monte-Carlo evaluator writes state value, observed action value, and partial returns
    # twice to metrics table (partial return horizon = math.ceil(np.log(1.0 - self.precision) / np.log(self.gamma))
    with tsdb_engine.connect() as conn:
        metrics = pd.read_sql_table('metrics', con=conn)

        state_v_entries = metrics.loc[metrics["metric"] == "state_v_0"]
        observed_a_q_entries = metrics.loc[metrics["metric"] == "observed_a_q_0"]
        partial_return_entries = metrics.loc[metrics["metric"] == "partial_return_0"]

        assert len(state_v_entries) == len(observed_a_q_entries) == len(partial_return_entries) == 2

def test_regression_normalizer_bounds_reset(offline_cfg: MainConfig):
    normalizer = NormalizerConfig(from_data=True)

    # add normalizer to pipeline config
    offline_cfg.pipeline.tags[1].state_constructor = [normalizer]
    pipeline = Pipeline(offline_cfg.pipeline)

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
    pr = pipeline(df, caller_code=CallerCode.OFFLINE)

    assert np.all(pr.df['Tag_1_norm'] == [1., 0, 0.5, 0.5, 0.5])

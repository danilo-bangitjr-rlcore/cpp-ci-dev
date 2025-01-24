import datetime as dt
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from docker.models.containers import Container
from torch import Tensor

from corerl.agent.factory import init_agent
from corerl.agent.greedy_ac import GreedyACConfig
from corerl.component.actor.network_actor import NetworkActorConfig
from corerl.component.buffer.uniform import UniformReplayBufferConfig
from corerl.component.critic.ensemble_critic import EnsembleCriticConfig
from corerl.component.optimizers.torch_opts import AdamConfig
from corerl.config import MainConfig, MetricsDBConfig
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
from corerl.eval.writer import metrics_group
from corerl.experiment.config import ExperimentConfig
from corerl.messages.event_bus import EventBus
from corerl.offline.utils import OfflineTraining
from corerl.state import AppState
from test.infrastructure.utils.docker import init_docker_container  # noqa: F401


@pytest.fixture(scope="module")
def test_db_config() -> TagDBConfig:
    db_cfg = TagDBConfig(
        drivername="postgresql+psycopg2",
        username="postgres",
        password="password",
        ip="localhost",
        port=5433,  # default is 5432, but we want to use different port for test db
        db_name="offline_test",
        table_name="tags",
    )

    return db_cfg


@pytest.fixture(scope="module")
def init_offline_tsdb_container():
    container = init_docker_container()
    yield container
    container.stop()
    container.remove()


@pytest.fixture(scope="module")
def data_writer(init_offline_tsdb_container: Container, test_db_config: TagDBConfig) -> Generator[DataWriter, None, None]: # noqa: F811, E501
    data_writer = DataWriter(cfg=test_db_config)

    yield data_writer

    data_writer.close()

@pytest.fixture
def offline_cfg(test_db_config: TagDBConfig) -> MainConfig:
    obs_period = dt.timedelta(seconds=60)
    action_period = 2*obs_period
    seed = 0

    cfg = MainConfig(
        metrics=MetricsDBConfig(
            enabled=False,
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
            offline_steps=100
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

def test_offline_training(offline_cfg: MainConfig, offline_trainer: OfflineTraining, data_writer: DataWriter):
    """
    Generate a few offline time steps, write them to TSDB, read the data from TSDB into a dataframe,
    pass data through the 'Anytime' data pipeline, train an agent on the produced transitions,
    and ensure the critic's training loss decreases
    """
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
    critic_losses = offline_trainer.train(agent)
    first_loss = critic_losses[0]
    last_loss = critic_losses[-1]

    assert last_loss < first_loss

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

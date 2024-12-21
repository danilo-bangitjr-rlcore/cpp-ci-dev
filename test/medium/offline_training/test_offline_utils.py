import pytest
import datetime as dt

from torch import Tensor
from typing import Generator

from corerl.agent.factory import init_agent
from corerl.agent.greedy_ac import GreedyACConfig
from corerl.component.actor.network_actor import NetworkActorConfig
from corerl.component.critic.ensemble_critic import EnsembleCriticConfig
from corerl.component.buffer.buffers import UniformReplayBufferConfig
from corerl.component.optimizers.torch_opts import AdamConfig
from corerl.config import MainConfig
from corerl.data_pipeline.datatypes import Step, NewTransition
from corerl.data_pipeline.db.data_writer import DataWriter
from corerl.data_pipeline.db.data_reader import TagDBConfig
from corerl.data_pipeline.pipeline import Pipeline, PipelineConfig
from corerl.data_pipeline.state_constructors.sc import SCConfig
from corerl.data_pipeline.state_constructors.countdown import CountdownConfig
from corerl.data_pipeline.tag_config import TagConfig
from corerl.data_pipeline.transition_filter import TransitionFilterConfig
from corerl.data_pipeline.all_the_time import AllTheTimeTCConfig
from corerl.data_pipeline.transforms import LessThanConfig
from corerl.experiment.config import ExperimentConfig
from corerl.offline.utils import load_offline_transitions, offline_training

from test.medium.data_loaders.utils import timescale_docker

@pytest.fixture(scope="module")
def test_db_config() -> TagDBConfig:
    db_cfg = TagDBConfig(
        drivername="postgresql+psycopg2",
        username="postgres",
        password="password",
        ip="localhost",
        port=5433,  # default is 5432, but we want to use different port for test db
        db_name="offline_test",
        sensor_table_name="tags",
    )

    return db_cfg

@pytest.fixture(scope="module")
def data_writer(timescale_docker, test_db_config: TagDBConfig) -> Generator[DataWriter, None, None]:
    data_writer = DataWriter(db_cfg=test_db_config)

    yield data_writer

    data_writer.close()

@pytest.fixture
def offline_cfg(test_db_config: TagDBConfig) -> MainConfig:
    obs_period_sec = 60 # seconds
    action_period = 2 # Number of obs_periods
    action_period_sec = obs_period_sec * action_period
    seed = 0

    cfg = MainConfig(
        action_period=action_period_sec,
        obs_period=obs_period_sec,
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
            load_env_obs_space_from_data=True,
            offline_steps=100
        ),
        pipeline=PipelineConfig(
            tags=[
                TagConfig(
                    name="Action",
                    is_action=True,
                    bounds=(0.0, 1.0)
                ),
                TagConfig(
                    name="Tag_1",
                    reward_constructor=[
                        LessThanConfig(threshold=3),
                    ],
                ),
            ],
            db=test_db_config,
            obs_interval_minutes=int(obs_period_sec / 60),
            state_constructor=SCConfig(
                defaults=[],
                countdown=CountdownConfig(action_period=action_period),
            ),
            transition_creator=AllTheTimeTCConfig(
                gamma=0.9,
                min_n_step=1,
                max_n_step=2,
            ),
            transition_filter=TransitionFilterConfig(
                filters=['only_no_action_change']
            )
        )
    )

    return cfg

def generate_offline_data(offline_cfg: MainConfig, data_writer: DataWriter, steps: int = 5) -> list[NewTransition]:
    # Generate timestamps
    timestamps = []
    start_time = dt.datetime(year=2023, month=7, day=13, hour=10, minute=0, tzinfo=dt.timezone.utc)
    delta = dt.timedelta(minutes=int(offline_cfg.obs_period / 60))
    for i in range(steps):
        timestamps.append(start_time + (delta * i))

    # Generate tag data and write to tsdb
    steps_per_decision = offline_cfg.action_period / offline_cfg.obs_period
    for i in range(steps):
        for tag_cfg in offline_cfg.pipeline.tags:
            tag = tag_cfg.name
            if tag_cfg.is_action:
                val = int(i / steps_per_decision) % 2
            else:
                val = i

            data_writer.write(timestamp=timestamps[i], name=tag, val=val)

    data_writer.background_sync()

    # Produce offline transitions
    pipeline = Pipeline(offline_cfg.pipeline)
    created_transitions = load_offline_transitions(offline_cfg, pipeline)

    return created_transitions

def test_load_offline_transitions(offline_cfg: MainConfig, data_writer: DataWriter):
    """
    Generate a few offline time steps, write them to TSDB, read the data from TSDB into a dataframe,
    pass data through the 'Anytime' data pipeline, and ensure the correct transitions are produced
    """
    steps = 5

    created_transitions = generate_offline_data(offline_cfg, data_writer, steps)

    # Expected transitions
    gamma = offline_cfg.experiment.gamma
    step_0 = Step(reward=1.0, action=Tensor([0.0]), gamma=gamma, state=Tensor([0.0]), dp=True)
    step_1 = Step(reward=1.0, action=Tensor([0.0]), gamma=gamma, state=Tensor([1.0]), dp=False)
    step_2 = Step(reward=1.0, action=Tensor([1.0]), gamma=gamma, state=Tensor([2.0]), dp=True)
    step_3 = Step(reward=0.0, action=Tensor([1.0]), gamma=gamma, state=Tensor([3.0]), dp=False)
    step_4 = Step(reward=0.0, action=Tensor([0.0]), gamma=gamma, state=Tensor([4.0]), dp=True)
    expected_transitions = [NewTransition([step_0, step_1], 1.0, gamma),
                            NewTransition([step_1, step_2], 1.0, gamma),
                            NewTransition([step_2, step_3], 0.0, gamma),
                            NewTransition([step_1, step_2, step_3], 1.0, gamma**2.0),
                            NewTransition([step_3, step_4], 0.0, gamma)]

    assert len(created_transitions) == len(expected_transitions)
    for i in range(len(created_transitions)):
        assert created_transitions[i] == expected_transitions[i]

def test_offline_training(offline_cfg: MainConfig, data_writer: DataWriter):
    """
    Generate a few offline time steps, write them to TSDB, read the data from TSDB into a dataframe,
    pass data through the 'Anytime' data pipeline, train an agent on the produced transitions,
    and ensure the critic's training loss decreases
    """
    steps = 5

    offline_transitions = generate_offline_data(offline_cfg, data_writer, steps)

    pipeline = Pipeline(offline_cfg.pipeline)
    state_dim, action_dim = pipeline.get_state_action_dims()
    agent = init_agent(offline_cfg.agent, state_dim, action_dim)

    # Offline training
    critic_losses = offline_training(offline_cfg, agent, offline_transitions)
    first_loss = critic_losses[0]
    last_loss = critic_losses[-1]

    assert last_loss < first_loss

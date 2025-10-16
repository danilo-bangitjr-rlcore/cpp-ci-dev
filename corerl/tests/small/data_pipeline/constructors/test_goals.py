import numpy as np
import pandas as pd
import pytest

from corerl.config import MainConfig
from corerl.configs.environment.reward.config import RewardConfig
from corerl.data_pipeline.constructors.goals import GoalConstructor
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.pipeline import Pipeline
from corerl.state import AppState
from tests.infrastructure.config import create_config_with_overrides
from tests.sdk.factories import PipelineFrameFactory


@pytest.fixture
def cfg():
    return create_config_with_overrides(
        base_config_path='tests/small/data_pipeline/constructors/assets/reward_config.yaml',
    )

@pytest.fixture
def cfg_with_oob():
    return create_config_with_overrides(
        base_config_path='tests/small/data_pipeline/constructors/assets/oob_reward_config.yaml',
    )

@pytest.fixture
def only_optimization_cfg():
    return create_config_with_overrides(
        base_config_path='tests/small/data_pipeline/constructors/assets/only_optimization_config.yaml',
    )

@pytest.fixture
def pipeline(cfg: MainConfig, dummy_app_state: AppState):
    return Pipeline(dummy_app_state, cfg.pipeline)


def test_reward_cfg_schema(cfg: MainConfig):
    assert isinstance(cfg.pipeline.reward, RewardConfig)


def test_goal1(cfg: MainConfig, pipeline: Pipeline, dummy_app_state: AppState):
    assert cfg.pipeline.reward is not None

    rc = GoalConstructor(dummy_app_state, cfg.pipeline.reward, cfg.pipeline.tags, pipeline.preprocessor)

    # we want to execute all stages up to (but not including)
    # the reward constructor
    stages = []
    for stage in pipeline.default_stages:
        if stage == StageCode.RC:
            break

        stages.append(stage)

    df = PipelineFrameFactory.build(
        data={
            'tag-0': [0, 4, 8, 10, 10, 5, 8.1, 9.1, 10, 9],
            'tag-1': [0, 3, 8, 5, 8, 9, 9, 9, 9, 10],
            'tag-2': [0, 7, 2, 3, 3, 1, 3, 1, 0, 1],
            'tag-3': [0, 1, 7, 6, 1, 3, 4, 4, 2, 1],
        },
    ).data

    pr = pipeline(df, stages=stages)
    pf = PipelineFrame(pr.df, pr.data_mode)
    pf.actions = pr.actions

    out = rc(pf)

    expected_rewards = pd.DataFrame(
        index=df.index,
        columns=['reward'],
        data=[
            # priority 1 - [-1, -0.75]
            [-1],
            [-0.888889],
            [-0.777778],
            # priority 2 - [-0.75, -0.5]
            [-0.593750],
            [-0.531250],
            # priority 1 - [-1, -0.75]
            [-0.861111],
            [-0.775000],
            # priority 3 - [-0.5, 0]
            [-0.4],
            [-0.2],
            [-0.1],
        ],
        dtype=np.float32,
    )

    pd.testing.assert_frame_equal(out.rewards, expected_rewards)

def test_ignore_oob_goal_tags(cfg_with_oob: MainConfig, pipeline: Pipeline, dummy_app_state: AppState):
    assert cfg_with_oob.pipeline.reward

    rc = GoalConstructor(
        dummy_app_state,
        cfg_with_oob.pipeline.reward,
        cfg_with_oob.pipeline.tags,
        pipeline.preprocessor,
    )

    stages = []
    for stage in pipeline.default_stages:
        if stage == StageCode.RC:
            break

        stages.append(stage)

    df = PipelineFrameFactory.build(
        data={
            'tag-0': [0, 2.5, 4.9, 5, 6, 7, 8, 8, 8, 8, 8, 8],
            'tag-1': [0, 3, 8, 5, 3, 2, -0.1, 1.1, 1.5, 1.5, 1.5, 1.5],
            'tag-2': [0, 7, 2, 3, 3, 3, 3, 1.1, -4, 0.5, 0.5, 0.5],
            'tag-3': [0, 1, 7, 6, 1, 1, 1, 3, 4, 4, 2, 1],
        },
    ).data

    pr = pipeline(df, stages=stages)
    pf = PipelineFrame(pr.df, pr.data_mode)
    pf.actions = pr.actions

    out = rc(pf)

    expected_rewards = pd.DataFrame(
        index=df.index,
        columns=['reward'],
        data=[
            # priority 1
            [-1.0],
            [-0.9166666666666667],
            [-0.8366666666666667],
            # priority 2
            [-0.7291666666666667],
            [-0.7083333333333334],
            [-0.6875],
            [-0.6666666666666667], # should handle tag-0 out of bounds with a violation percent of 0
            # priority 3
            [-0.5018518518518519],
            [-0.5092592592592593], # failed the OR condition, violation percent of 100
            # priority 4
            [-0.4],
            [-0.2],
            [-0.1],
        ],
        dtype=np.float32,
    )

    pd.testing.assert_frame_equal(out.rewards, expected_rewards)

def test_only_optimization(only_optimization_cfg: MainConfig, dummy_app_state: AppState):
    """
    Since the only Priority in this test is an Optimization,
    make sure rewards are in the range [-1, 0] rather than [-0.5, 0]
    """
    cfg = only_optimization_cfg
    pipeline = Pipeline(dummy_app_state, cfg.pipeline)
    assert cfg.pipeline.reward is not None

    rc = GoalConstructor(dummy_app_state, cfg.pipeline.reward, cfg.pipeline.tags, pipeline.preprocessor)

    # we want to execute all stages up to (but not including)
    # the reward constructor
    stages = []
    for stage in pipeline.default_stages:
        if stage == StageCode.RC:
            break

        stages.append(stage)

    df = PipelineFrameFactory.build(
        data={
            'tag-0': [0, 4, 8, 10, 10, 5, 8.1, 9.1, 10, 9],
            'tag-1': [0, 3, 8, 5, 8, 9, 9, 9, 9, 10],
            'tag-2': [0, 7, 2, 3, 3, 1, 3, 1, 0, 1],
            'tag-3': [0, 1, 3, 5, 1, 3, 4, 4, 2, 1],
        },
    ).data

    pr = pipeline(df, stages=stages)
    pf = PipelineFrame(pr.df, pr.data_mode)
    pf.actions = pr.actions

    out = rc(pf)

    expected_rewards = pd.DataFrame(
        index=df.index,
        columns=['reward'],
        data=[
            [0],
            [-0.2],
            [-0.6],
            [-1],
            [-0.2],
            [-0.6],
            [-0.8],
            [-0.8],
            [-0.4],
            [-0.2],
        ],
        dtype=np.float32,
    )

    pd.testing.assert_frame_equal(out.rewards, expected_rewards)

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from corerl.config import MainConfig
from corerl.configs.loader import direct_load_config
from corerl.data_pipeline.constructors.goals import GoalConstructor
from corerl.data_pipeline.datatypes import PipelineFrame, StageCode
from corerl.data_pipeline.pipeline import Pipeline
from corerl.environment.reward.config import RewardConfig
from corerl.state import AppState


@pytest.fixture
def cfg():
    return direct_load_config(
        MainConfig,
        base='corerl/test/small/data_pipeline/constructors/assets',
        config_name='reward_config.yaml',
    )

@pytest.fixture
def cfg_with_oob():
    return direct_load_config(
        MainConfig,
        base='corerl/test/small/data_pipeline/constructors/assets',
        config_name='oob_reward_config.yaml',
    )

@pytest.fixture
def pipeline(cfg: MainConfig, dummy_app_state: AppState):
    return Pipeline(dummy_app_state, cfg.pipeline)


def test_reward_cfg_schema(cfg: MainConfig):
    assert isinstance(cfg.pipeline.reward, RewardConfig)


def test_goal1(cfg: MainConfig, pipeline: Pipeline):
    assert cfg.pipeline.reward is not None

    rc = GoalConstructor(cfg.pipeline.reward, cfg.pipeline.tags, pipeline.preprocessor)

    # we want to execute all stages up to (but not including)
    # the reward constructor
    stages = []
    for stage in pipeline.default_stages:
        if stage == StageCode.RC:
            break

        stages.append(stage)

    start = datetime.now(UTC)
    Δ = timedelta(minutes=5)

    dates = [start + i * Δ for i in range(10)]
    idx = pd.DatetimeIndex(dates)

    cols: Any = ['tag-0', 'tag-1', 'tag-2', 'tag-3']
    df = pd.DataFrame(
        data=[
            # priority 1 - tag-0 up to 9
            [0,     0,     0,     0],
            [4,     3,     7,     1],
            [8,     8,     2,     7],
            # priority 2 - tag-1 up to 8 AND tag-2 down to 2
            [10,    5,     3,     6],
            [10,    8,     3,     1],
            # priority 1 because we dipped below tag-0 thresh
            # even though we now satisfy priority 2
            [5,     9,     1,     3],
            [8.1,   9,     3,     4],
            # priority 3 - optimize tag-3
            [9.1,   9,     1,     4],
            [10,    9,     0,     2],
            [9,     10,    1,     1],
        ],
        columns=cols,
        index=idx,
    )

    pr = pipeline(df, stages=stages)
    pf = PipelineFrame(pr.df, pr.data_mode)
    pf.actions = pr.actions

    out = rc(pf)

    expected_rewards = pd.DataFrame(
        index=idx,
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
    )

    pd.testing.assert_frame_equal(out.rewards, expected_rewards)

def test_ignore_oob_goal_tags(cfg_with_oob: MainConfig, pipeline: Pipeline):
    assert cfg_with_oob.pipeline.reward
    assert cfg_with_oob.pipeline.reward.ignore_oob_tags_in_compound_goals, "feature flag is not set"

    rc = GoalConstructor(cfg_with_oob.pipeline.reward, cfg_with_oob.pipeline.tags, pipeline.preprocessor)

    stages = []
    for stage in pipeline.default_stages:
        if stage == StageCode.RC:
            break

        stages.append(stage)

    start = datetime.now(UTC)
    dt = timedelta(minutes=5)

    dates = [start + i * dt for i in range(12)]
    idx = pd.DatetimeIndex(dates)

    cols: Any = ['tag-0', 'tag-1', 'tag-2', 'tag-3']
    df = pd.DataFrame(
        data=[
            # priority 1 - tag-0 up to 5
            [0,     0,     0,     0],
            [2.5,   3,     7,     1],
            [4.9,   8,     2,     7],
            # priority 2 - tag-0 up to 8 AND tag-1 down to 2
            [5,     5,     3,     6],
            [6,     3,     3,     1],
            [7,     2,     3,     1],
            [8,     -0.1,  3,     1], # tag-1 below operating range of 0, assume violation percent of 0
            # priority 3 - tag-1 down to 1 OR tag-2 down to 1
            [8,     1.1,   1.1,     3],
            [8,     1.5,   -4,    4], # tag-2 below operating range of 0, assume violation percent of 100
            # priority 4 - optimize tag-3
            [8,     1.5,   0.5,   4],
            [8,     1.5,   0.5,   2],
            [8,     1.5,   0.5,   1],
        ],
        columns=cols,
        index=idx,
    )

    pr = pipeline(df, stages=stages)
    pf = PipelineFrame(pr.df, pr.data_mode)
    pf.actions = pr.actions

    out = rc(pf)

    expected_rewards = pd.DataFrame(
        index=idx,
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
    )

    pd.testing.assert_frame_equal(out.rewards, expected_rewards)

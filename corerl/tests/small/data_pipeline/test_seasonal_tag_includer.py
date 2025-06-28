import pandas as pd
from lib_defs.config_defs.tag_config import TagType
from test.infrastructure.utils.pandas import dfs_close

from corerl.data_pipeline.datatypes import DataMode, PipelineFrame
from corerl.data_pipeline.seasonal_tags import SeasonalTagIncluder
from corerl.tags.tag_config import TagConfig


def test_no_seasonal_tags_enabled():
    tag_cfgs = [
        TagConfig(
            name="tag_1",
        ),
        TagConfig(
            name="action_1",
            type=TagType.ai_setpoint,
        ),
    ]

    seasonal_tag_stage = SeasonalTagIncluder(tag_cfgs)

    assert not seasonal_tag_stage.has_day_of_year
    assert not seasonal_tag_stage.has_day_of_week
    assert not seasonal_tag_stage.has_time_of_day

    df = pd.DataFrame({
        "tag_1": [1, 5],
        "action_1": [0, 0],
    })
    df.index = pd.DatetimeIndex(["9/24/2023 10:12", "7/13/2024 6:37"])
    pf = PipelineFrame(df, DataMode.ONLINE)
    out = seasonal_tag_stage(pf)

    expected = df

    assert dfs_close(out.data, expected)

def test_all_seasonal_tags_enabled():
    tag_cfgs = [
        TagConfig(
            name="day_of_year",
            type=TagType.day_of_year,
        ),
        TagConfig(
            name="day_of_week",
            type=TagType.day_of_week,
        ),
        TagConfig(
            name="time_of_day",
            type=TagType.time_of_day,
        ),
        TagConfig(
            name="tag_1",
        ),
        TagConfig(
            name="action_1",
            type=TagType.ai_setpoint,
        ),
    ]

    seasonal_tag_stage = SeasonalTagIncluder(tag_cfgs)

    assert seasonal_tag_stage.has_day_of_year
    assert seasonal_tag_stage.has_day_of_week
    assert seasonal_tag_stage.has_time_of_day

    df = pd.DataFrame({
        "tag_1": [1, 5],
        "action_1": [0, 0],
    })
    dates = pd.DatetimeIndex(["9/24/2023 10:12", "7/13/2024 6:37"])
    df.index = dates
    pf = PipelineFrame(df, DataMode.ONLINE)
    out = seasonal_tag_stage(pf)

    expected = pd.DataFrame({
        "tag_1": [1, 5],
        "action_1": [0, 0],
        "day_of_year": [267, 195],
        "day_of_week": [6, 5],
        "time_of_day": [36720, 23820],
    })

    assert dfs_close(out.data, expected)

def test_some_seasonal_tags_enabled():
    tag_cfgs = [
        TagConfig(
            name="day_of_year",
            type=TagType.day_of_year,
        ),
        TagConfig(
            name="time_of_day",
            type=TagType.time_of_day,
        ),
        TagConfig(
            name="tag_1",
        ),
        TagConfig(
            name="action_1",
            type=TagType.ai_setpoint,
        ),
    ]

    seasonal_tag_stage = SeasonalTagIncluder(tag_cfgs)

    assert seasonal_tag_stage.has_day_of_year
    assert not seasonal_tag_stage.has_day_of_week
    assert seasonal_tag_stage.has_time_of_day

    df = pd.DataFrame({
        "tag_1": [1, 5],
        "action_1": [0, 0],
    })
    dates = pd.DatetimeIndex(["9/24/2023 10:12", "7/13/2024 6:37"])
    df.index = dates
    pf = PipelineFrame(df, DataMode.ONLINE)
    out = seasonal_tag_stage(pf)

    expected = pd.DataFrame({
        "tag_1": [1, 5],
        "action_1": [0, 0],
        "day_of_year": [267, 195],
        "time_of_day": [36720, 23820],
    })

    assert dfs_close(out.data, expected)

import pandas as pd
import pytest
from lib_defs.config_defs.tag_config import TagType
from test.infrastructure.utils.pandas import dfs_close

from corerl.data_pipeline.datatypes import DataMode, PipelineFrame
from corerl.data_pipeline.state_constructors.seasonal import SeasonalTagFeatures
from corerl.tags.tag_config import BasicTagConfig


@pytest.fixture()
def data() -> pd.DataFrame:
    return pd.DataFrame(
        data={"tag-1": [1,2,3], "tag-2": [4,5,6]},
        index=pd.DatetimeIndex(["1/18/2023 7:00:00", "3/27/2023 10:30:00", "7/13/2024 18:17:11"]),
    )


def test_time_of_day():
    df = pd.DataFrame(
        data={"tag-1": [1, 2, 3], "tag-2": [4, 5, 6], "time_of_day": [25200, 37800, 65831]},
        index=pd.DatetimeIndex(["1/18/2023 7:00:00", "3/27/2023 10:30:00", "7/13/2024 18:17:11"]),
    )

    pf = PipelineFrame(
        data=df,
        data_mode=DataMode.ONLINE,
    )

    tag_cfgs = [
        BasicTagConfig(name='time_of_day', type=TagType.seasonal),
    ]

    seasonal_features = SeasonalTagFeatures(tag_cfgs)
    pf = seasonal_features(pf)

    expected = pd.DataFrame({
        "tag-1": [1, 2, 3],
        "tag-2": [4, 5, 6],
        "time_of_day": [25200, 37800, 65831],
        "time_of_day_sin": [0.9659258, 0.382683, -0.99719058],
        "time_of_day_cos": [-0.258819, -0.9238795, 0.074906209],
    })

    assert dfs_close(pf.data, expected)

def test_day_of_week():
    df = pd.DataFrame(
        data={"tag-1": [1, 2, 3], "tag-2": [4, 5, 6], "day_of_week": [2, 0, 5]},
        index=pd.DatetimeIndex(["1/18/2023 7:00:00", "3/27/2023 10:30:00", "7/13/2024 18:17:11"]),
    )

    pf = PipelineFrame(
        data=df,
        data_mode=DataMode.ONLINE,
    )

    tag_cfgs = [
        BasicTagConfig(name='day_of_week', type=TagType.seasonal),
    ]

    seasonal_features = SeasonalTagFeatures(tag_cfgs)
    pf = seasonal_features(pf)

    expected = pd.DataFrame({
        "tag-1": [1, 2, 3],
        "tag-2": [4, 5, 6],
        "day_of_week": [2, 0, 5],
        "day_of_week_0": [0, 1, 0],
        "day_of_week_1": [0, 0, 0],
        "day_of_week_2": [1, 0, 0],
        "day_of_week_3": [0, 0, 0],
        "day_of_week_4": [0, 0, 0],
        "day_of_week_5": [0, 0, 1],
        "day_of_week_6": [0, 0, 0],
    })

    assert dfs_close(pf.data, expected)

def test_day_of_year():
    df = pd.DataFrame(
        data={"tag-1": [1, 2, 3], "tag-2": [4, 5, 6], "day_of_year": [18, 86, 195]},
        index=pd.DatetimeIndex(["1/18/2023 7:00:00", "3/27/2023 10:30:00", "7/13/2024 18:17:11"]),
    )

    pf = PipelineFrame(
        data=df,
        data_mode=DataMode.ONLINE,
    )

    tag_cfgs = [
        BasicTagConfig(name='day_of_year', type=TagType.seasonal),
    ]

    seasonal_features = SeasonalTagFeatures(tag_cfgs)
    pf = seasonal_features(pf)

    expected = pd.DataFrame({
        "tag-1": [1, 2, 3],
        "tag-2": [4, 5, 6],
        "day_of_year": [18, 86, 195],
        "time_of_year_sin": [0.3049212, 0.995919, -0.204552],
        "time_of_year_cos": [0.9523775, 0.0902516, -0.978855],
    })

    assert dfs_close(pf.data, expected)

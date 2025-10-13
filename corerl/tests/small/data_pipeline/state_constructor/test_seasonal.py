import pandas as pd
import pytest
from test.infrastructure.utils.pandas import dfs_close

from corerl.configs.tags.seasonal import SeasonalTagConfig, SeasonalTags
from corerl.data_pipeline.datatypes import DataMode, PipelineFrame
from corerl.data_pipeline.state_constructors.seasonal import SeasonalTagFeatures


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
        SeasonalTagConfig(name=SeasonalTags.time_of_day),
    ]

    seasonal_features = SeasonalTagFeatures(tag_cfgs)
    pf = seasonal_features(pf)

    expected = pd.DataFrame({
        "tag-1": [1, 2, 3],
        "tag-2": [4, 5, 6],
        "time_of_day": [25200, 37800, 65831],
        "time_of_day_sin": [0.9829629, 0.6913415, 0.00140471],
        "time_of_day_cos": [0.3705905, 0.03806025, 0.5374531],
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
        SeasonalTagConfig(name=SeasonalTags.day_of_week),
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
        SeasonalTagConfig(name=SeasonalTags.day_of_year),
    ]

    seasonal_features = SeasonalTagFeatures(tag_cfgs)
    pf = seasonal_features(pf)

    expected = pd.DataFrame(
        data={
            "tag-1": [1, 2, 3],
            "tag-2": [4, 5, 6],
            "day_of_year": [18, 86, 195],
            "time_of_year_sin": [0.6524606, 0.9979595, 0.397724],
            "time_of_year_cos": [0.97618875, 0.5451258, 0.0105725],
            },
        index=pd.DatetimeIndex(["1/18/2023 7:00:00", "3/27/2023 10:30:00", "7/13/2024 18:17:11"]),
    )

    assert dfs_close(pf.data, expected, tol=1e-5)


def test_second_in_hour():
    df = pd.DataFrame(
        data={"tag-1": [1, 2, 3], "tag-2": [4, 5, 6], "second_in_hour": [0, 1800, 1031]},
        index=pd.DatetimeIndex(["1/18/2023 7:00:00", "3/27/2023 10:30:00", "7/13/2024 18:17:11"]),
    )

    pf = PipelineFrame(
        data=df,
        data_mode=DataMode.ONLINE,
    )

    tag_cfgs = [
        SeasonalTagConfig(name=SeasonalTags.second_in_hour),
    ]

    seasonal_features = SeasonalTagFeatures(tag_cfgs)
    pf = seasonal_features(pf)

    expected = pd.DataFrame({
        "tag-1": [1, 2, 3],
        "tag-2": [4, 5, 6],
        "second_in_hour": [0, 1800, 1031],
        "second_in_hour_sin": [0.5, 0.5, 0.9869879],
        "second_in_hour_cos": [1.0, 0.0, 0.3866743],
    })

    assert dfs_close(pf.data, expected)

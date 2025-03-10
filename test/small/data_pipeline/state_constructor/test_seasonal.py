import pandas as pd
import pytest

from corerl.data_pipeline.datatypes import DataMode, PipelineFrame
from corerl.data_pipeline.state_constructors.seasonal import SeasonalConfig, add_seasonal_features
from test.infrastructure.utils.pandas import dfs_close


@pytest.fixture()
def data() -> pd.DataFrame:
    df = pd.DataFrame(
        data={"tag-1": [1,2,3], "tag-2": [4,5,6]},
        index=pd.DatetimeIndex(["1/18/2023 7:00:00", "3/27/2023 10:30:00", "7/13/2024 18:17:11"])
    )

    return df

def test_time_of_day(data: pd.DataFrame):
    pf = PipelineFrame(
        data=data,
        data_mode=DataMode.ONLINE
    )

    cfg = SeasonalConfig(
        time_of_day_enabled=True,
        day_of_week_enabled=False,
        time_of_year_enabled=False
    )

    pf = add_seasonal_features(cfg, pf)

    expected = pd.DataFrame({
        "tag-1": [1, 2, 3],
        "tag-2": [4, 5, 6],
        "time_of_day_sin": [0.9659258, 0.382683, -0.99719058],
        "time_of_day_cos": [-0.258819, -0.9238795, 0.074906209]
    })

    print("pf.data:")
    print(pf.data)

    assert dfs_close(pf.data, expected)

def test_day_of_week(data: pd.DataFrame):
    pf = PipelineFrame(
        data=data,
        data_mode=DataMode.ONLINE
    )

    cfg = SeasonalConfig(
        time_of_day_enabled=False,
        day_of_week_enabled=True,
        time_of_year_enabled=False
    )

    pf = add_seasonal_features(cfg, pf)

    expected = pd.DataFrame({
        "tag-1": [1, 2, 3],
        "tag-2": [4, 5, 6],
        "day_of_week_0": [0, 1, 0],
        "day_of_week_1": [0, 0, 0],
        "day_of_week_2": [1, 0, 0],
        "day_of_week_3": [0, 0, 0],
        "day_of_week_4": [0, 0, 0],
        "day_of_week_5": [0, 0, 1],
        "day_of_week_6": [0, 0, 0]
    })

    assert dfs_close(pf.data, expected)

def test_time_of_year(data: pd.DataFrame):
    pf = PipelineFrame(
        data=data,
        data_mode=DataMode.ONLINE
    )

    cfg = SeasonalConfig(
        time_of_day_enabled=False,
        day_of_week_enabled=False,
        time_of_year_enabled=True
    )

    pf = add_seasonal_features(cfg, pf)

    expected = pd.DataFrame({
        "tag-1": [1, 2, 3],
        "tag-2": [4, 5, 6],
        "time_of_year_sin": [0.3049212, 0.995919, -0.204552],
        "time_of_year_cos": [0.9523775, 0.0902516, -0.978855]
    })

    assert dfs_close(pf.data, expected)

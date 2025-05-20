import datetime

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from corerl.data_pipeline.datatypes import DataMode, PipelineFrame
from corerl.data_pipeline.imputers.per_tag.backfill import BackfillImputer, BackfillImputerConfig


def test_backfill():
    cols = {"sensor_x": [np.nan, np.nan, 2.0], "sensor_y": [1.0, np.nan, 3.0], "reward": [1.0, 2.0, 3.0]}
    dates = [
        datetime.datetime(2024, 1, 1, 1, 1),
        datetime.datetime(2024, 1, 2, 1, 2),
        datetime.datetime(2024, 1, 3, 1, 3),
    ]
    datetime_index = pd.DatetimeIndex(dates)
    df = pd.DataFrame(cols, index=datetime_index)

    expected_cols = {"sensor_x": [-1.0, -1.0, 2.0], "sensor_y": [3.0, 3.0, 3.0], "reward": [1.0, 2.0, 3.0]}
    expected_df = pd.DataFrame(expected_cols, index=datetime_index)

    backfill_x_cfg = BackfillImputerConfig(backfill_to=datetime.datetime(2024, 1, 2, 1, 2), backfill_val=-1.0)
    backfill_x = BackfillImputer(backfill_x_cfg)
    backfill_y_cfg = BackfillImputerConfig(backfill_to=datetime.datetime(2024, 1, 2, 1, 2), backfill_val=3.0)
    backfill_y = BackfillImputer(backfill_y_cfg)

    pf = PipelineFrame(data=df, data_mode=DataMode.ONLINE)
    pf = backfill_x(pf, tag="sensor_x")
    pf = backfill_y(pf, tag="sensor_y")

    assert_frame_equal(pf.data, expected_df)

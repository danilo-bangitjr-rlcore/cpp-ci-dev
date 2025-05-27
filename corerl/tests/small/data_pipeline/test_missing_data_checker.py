import numpy as np
import pandas as pd

from corerl.data_pipeline.datatypes import DataMode, MissingType, PipelineFrame
from corerl.data_pipeline.missing_data_checker import missing_data_checker


def test_missing_data_gets_correct_missingtype():
    cols = {"sensor_x": [np.nan, 1.0], "sensor_y": [2.0, np.nan]}

    data = pd.DataFrame(cols)
    pf = PipelineFrame(data, DataMode.ONLINE)

    pf = missing_data_checker(pf, 'sensor_x')
    pf = missing_data_checker(pf, 'sensor_y')

    missing_info = pf.missing_info

    assert missing_info["sensor_x"].iloc[0] == MissingType.MISSING
    assert missing_info["sensor_x"].iloc[1] == MissingType.NULL
    assert missing_info["sensor_y"].iloc[0] == MissingType.NULL
    assert missing_info["sensor_y"].iloc[1] == MissingType.MISSING

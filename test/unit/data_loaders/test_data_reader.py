import unittest

import numpy as np
from pandas import DataFrame
from corerl.data_pipeline.db.data_reader import fill_data_for_changed_setpoint
from datetime import datetime, timedelta
from typing import List

class DataReaderTestCase(unittest.TestCase):
    def test_fill_data_for_changed_setpoint(self):
        sensor_names: List[str] = ["ChangeSP", "SP"]
        ts_start = datetime.now()
        d = 10
        test_size = 30
        delta_t = timedelta(seconds=d)
        timestamps = [ts_start + delta_t * i for i in range(test_size)]

        data_sp = {
            'Timestamp': timestamps,
            'Tag': 'SP',
            'Value': list(np.arange(len(timestamps)))
        }
        df_sp = DataFrame(data_sp)
        data_sp_change = {
            'Timestamp': [timestamps[i] for i in list(range(0, len(timestamps), 3))],
            'Tag': 'ChangeSP',
            'Value': list(range(0, len(timestamps), 3))
        }
        df_sp_change = DataFrame(data_sp_change)
        dfs = [df_sp_change, df_sp]

        result_tuple = fill_data_for_changed_setpoint(
            change_tags=["ChangeSP"],
            dfs=dfs,
            delta_t=timedelta(seconds=d)
        )

        assert len(result_tuple) == len(timestamps) * len(sensor_names), \
            print("Length of result tuple is", len(result_tuple))

if __name__ == '__main__':
    unittest.main()

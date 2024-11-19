import unittest

import numpy as np
from pandas import DataFrame
from datetime import datetime, timedelta
from corerl.data_pipeline.db.data_cleaner import fill_in_missing_data, clean_by_multiple_sliding_window


class DataCleanerTestCase(unittest.TestCase):
    def initialize_data(self):
        ts_start = datetime.now()
        d = 10
        test_size = 10
        delta_t = timedelta(seconds=d)
        delta_t_gap = timedelta(seconds=d*2)
        timestamps1 = [ts_start + delta_t * i for i in range(test_size)]
        start2 = timestamps1[-1] + delta_t_gap
        timestamps2 = [start2 + delta_t * i for i in range(test_size)]
        timestamps = timestamps1 + timestamps2
        values = list(np.arange(len(timestamps)))
        values[len(timestamps1)] = np.nan
        return timestamps, timestamps1, timestamps2, values, d

    def test_fill_in_missing_data_with_average(self):
        timestamps, timestamps1, timestamps2, values, d = self.initialize_data()
        data_sp = {
            'Timestamp': timestamps,
            'Value': values
        }
        df_sp = DataFrame(data_sp)
        result_df = fill_in_missing_data(
            data=df_sp,
            remove_mode='Average',
            remove_length_threshold=d * 1.5
        )
        assert result_df.iloc[len(timestamps1)]['Value'] == \
               (values[len(timestamps1)-1] + values[len(timestamps1)+1]) / 2., \
            "The value at gap should be equal to the average of the values before and after the gap."
        assert len(result_df) == len(timestamps), \
            "The returned dataframe should be same as the original dataframe."

    def test_remove_missing_data(self):
        timestamps, timestamps1, timestamps2, values, d = self.initialize_data()
        data_sp = {
            'Timestamp': timestamps,
            'Value': values
        }
        df_sp = DataFrame(data_sp)
        result_df = fill_in_missing_data(
            data=df_sp,
            remove_mode='Delete'
        )
        assert result_df.iloc[len(timestamps1)]['Value'] == values[len(timestamps1)+1], \
            "The value at gap should be removed"
        assert len(result_df) == len(timestamps)-1, \
            "The returned dataframe should be shorter than the original dataframe."

    def test_fill_in_missing_data_with_zero(self):
        timestamps, timestamps1, timestamps2, values, d = self.initialize_data()
        data_sp = {
            'Timestamp': timestamps,
            'Value': values
        }
        df_sp = DataFrame(data_sp)
        result_df = fill_in_missing_data(
            data=df_sp,
            remove_mode='Zero'
        )
        assert result_df.iloc[len(timestamps1)]['Value'] == 0., \
            "The value at gap should be 0"
        assert len(result_df) == len(timestamps), \
            "The returned dataframe should be same as the original dataframe."

    def test_fill_in_missing_data_with_copying_last(self):
        timestamps, timestamps1, timestamps2, values, d = self.initialize_data()
        data_sp = {
            'Timestamp': timestamps,
            'Value': values
        }
        df_sp = DataFrame(data_sp)
        result_df = fill_in_missing_data(
            data=df_sp,
            remove_mode='Copy'
        )
        assert (result_df.iloc[len(timestamps1)]['Value'] ==
                result_df.iloc[len(timestamps1)-1]['Value']), \
            "The value at gap should be same as the previous data"
        assert len(result_df) == len(timestamps), \
            "The returned dataframe should be same as the original dataframe."

    def test_clean_by_multiple_sliding_window(self):
        timestamps, timestamps1, timestamps2, values, d = self.initialize_data()
        values[len(timestamps1)//2] = -100.
        values[len(timestamps1) + len(timestamps2)//2] = -50.
        values = np.asarray(values)
        outliers, infos = clean_by_multiple_sliding_window(
            data=values,
            window_length_lst=[len(timestamps1)-1],
            in_range_threshold=0.1
        )
        assert len(outliers) == 3, "-100, NaN, -50 should be recognized as outliers"
        assert outliers[0] == len(timestamps1)//2, "The first outlier should be -100"
        assert outliers[1] == len(timestamps1), "The second outlier should be NaN"
        assert outliers[2] == len(timestamps1) + len(timestamps2)//2, \
            "The third outlier should be -50"


if __name__ == '__main__':
    unittest.main()


import pandas as pd
from sqlalchemy import Engine

from test.behavior.bsuite import BSuiteTestCase


class CalibrationTest(BSuiteTestCase):
    name = 'calibration'
    config = 'test/behavior/calibration/calibration.yaml'

    # The best possible reward is either -0.571 or -0.286, depending on the bias.
    lower_bounds = {'reward': -1}

    def evaluate_outcomes(self, tsdb: Engine, metrics_table: pd.DataFrame, features: dict[str, bool]) -> pd.DataFrame:
        summary_df = super().evaluate_outcomes(tsdb, metrics_table, features)
        self._analyze_calibration_recovery(metrics_table)

        return summary_df

    def _analyze_calibration_recovery(self, metrics_table: pd.DataFrame):
        reward_data = metrics_table[metrics_table['metric'] == 'reward']

        assert len(reward_data) > 0, "No reward data found for calibration analysis"

        reward_data = reward_data.sort_values('agent_step')

        pre_calibration = reward_data[reward_data['agent_step'] < 500]
        post_calibration = reward_data[reward_data['agent_step'] > 500]

        if len(pre_calibration) > 0:
            pre_avg = pre_calibration['value'].mean()
            if len(pre_calibration) >= 100:
                pre_stable = pre_calibration.iloc[-100:]['value'].mean()
            else:
                pre_stable = pre_avg
        else:
            pre_avg = None
            pre_stable = None


        if len(post_calibration) > 0:
            post_avg = post_calibration['value'].mean()
        else:
            post_avg = None


        if pre_stable is not None and post_avg is not None:
            recovery_percent = post_avg / pre_stable * 100 if pre_stable != 0 else 0
            assert recovery_percent > 80, f"Recovery {recovery_percent:.2f}% below 80% threshold"

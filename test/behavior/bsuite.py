import subprocess
import warnings

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from corerl.sql_logging.sql_logging import table_exists


class BSuiteTestCase:
    name: str
    config: str
    lower_bounds: dict[str, float] = {}
    upper_bounds: dict[str, float] = {}
    lower_warns: dict[str, float] = {}
    upper_warns: dict[str, float] = {}

    overrides: dict[str, object] | None = None

    def __init__(self):
        self._overrides = self.overrides or {}


    def execute_test(self, tsdb: Engine, port: int, db_name: str):
        overrides = self._overrides | {
            'infra.db.port': port,
            'infra.db.db_name': db_name,
        }

        parts = [f'{k}={v}' for k, v in overrides.items()]

        proc = subprocess.run([
            'uv', 'run', 'python', 'main.py',
            '--base', '.',
            '--config-name', self.config,
        ] + parts)
        proc.check_returncode()

        # ensure metrics table exists
        assert table_exists(tsdb, 'metrics')

        # ensure some metrics were logged to table
        with tsdb.connect() as conn:
            metrics_table = pd.read_sql_table('metrics', con=conn)

        metrics_table = metrics_table.sort_values('agent_step', ascending=True)
        return metrics_table

    def evaluate_outcomes(self, metrics_table: pd.DataFrame):
        self._evaluate_warnings(metrics_table)
        self._evaluate_bounds(metrics_table)


    def summarize_over_time(self, metric: str, metrics_table: pd.DataFrame):
        values = get_metric(metrics_table, metric)
        return values[-100:].mean()


    def _evaluate_bounds(self, metrics_table: pd.DataFrame):
        for metric, expected in self.lower_bounds.items():
            got = self.summarize_over_time(metric, metrics_table)
            assert got >= expected, \
                f'[{self.name}] - {metric} outside of lower bound - {got} >= {expected}'

        for metric, expected in self.upper_bounds.items():
            got = self.summarize_over_time(metric, metrics_table)
            assert got <= expected, \
                f'[{self.name}] - {metric} outside of upper bound - {got} <= {expected}'


    def _evaluate_warnings(self, metrics_table: pd.DataFrame):
        for metric, expected in self.lower_bounds.items():
            got = self.summarize_over_time(metric, metrics_table)
            if got < expected:
                warnings.warn(f'[{self.name}] - {metric} outside of lower bound - {got} >= {expected}', stacklevel=0)

        for metric, expected in self.upper_bounds.items():
            got = self.summarize_over_time(metric, metrics_table)
            if got > expected:
                warnings.warn(f'[{self.name}] - {metric} outside of upper bound - {got} <= {expected}', stacklevel=0)


def get_metric(df: pd.DataFrame, metric: str) -> np.ndarray:
    return df[df['metric'] == metric]['value'].to_numpy()

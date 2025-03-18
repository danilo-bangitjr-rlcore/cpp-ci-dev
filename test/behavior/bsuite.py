import subprocess
import warnings
from enum import Enum, auto

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from corerl.sql_logging.sql_logging import table_exists


class Behaviour(Enum):
    exploration = auto()


class BSuiteTestCase:
    name: str
    config: str
    behaviours: dict[str, Behaviour] = {}
    lower_bounds: dict[str, float] = {}
    upper_bounds: dict[str, float] = {}
    lower_warns: dict[str, float] = {}
    upper_warns: dict[str, float] = {}
    lower_goals: dict[str, float] = {}
    upper_goals: dict[str, float] = {}

    aggregators: dict[str, str] = {}

    overrides: dict[str, object] | None = None

    def __init__(self):
        self._overrides = self.overrides or {}

    def execute_test(self, tsdb: Engine, db_name: str, schema: str):
        ip = tsdb.url.host
        port = tsdb.url.port
        overrides = self._overrides | {
            'infra.db.ip': ip,
            'infra.db.port': port,
            'infra.db.db_name': db_name,
            'infra.db.schema': schema,
        }

        parts = [f'{k}={v}' for k, v in overrides.items()]

        proc = subprocess.run([
            'uv', 'run', 'python', 'main.py',
            '--base', '.',
            '--config-name', self.config,
        ] + parts)
        proc.check_returncode()

        # ensure metrics table exists
        assert table_exists(tsdb, 'metrics', schema=schema)

        # ensure some metrics were logged to table
        with tsdb.connect() as conn:
            metrics_table = pd.read_sql_table('metrics', schema=schema, con=conn)

        metrics_table = metrics_table.sort_values('agent_step', ascending=True)
        return metrics_table

    def evaluate_outcomes(self, metrics_table: pd.DataFrame) -> pd.DataFrame:
        extracted = []
        extracted += self._extract(self.lower_bounds, 'lower_bounds', metrics_table)
        extracted += self._extract(self.upper_bounds, 'upper_bounds', metrics_table)
        extracted += self._extract(self.lower_warns,  'lower_warns',  metrics_table)
        extracted += self._extract(self.upper_warns,  'upper_warns',  metrics_table)
        extracted += self._extract(self.lower_goals,  'lower_goals',  metrics_table)
        extracted += self._extract(self.upper_goals,  'upper_goals',  metrics_table)

        summary_df = pd.DataFrame(extracted, columns=['metric', 'behaviour', 'bound_type', 'expected', 'got'])
        self._evaluate_bounds(summary_df)
        self._evaluate_warnings(summary_df)

        return summary_df

    def summarize_over_time(self, metric: str, metrics_table: pd.DataFrame) -> float:
        values = get_metric(metrics_table, metric)
        aggregation_name = self.aggregators.get(metric, 'last_100_mean')
        aggregated_values = aggregate(values, name=aggregation_name)
        return aggregated_values


    def _extract(self, tests: dict[str, float], bound_type: str, metrics_table: pd.DataFrame)-> list[list[str | float]]:
        extracted = []
        for metric, expected in tests.items():
            got = self.summarize_over_time(metric, metrics_table)
            extracted.append([
                metric,
                self.behaviours.get(metric, 'None'),
                bound_type,
                expected,
                got,
            ])

        return extracted

    def _evaluate_bounds(self, summary_df: pd.DataFrame):
        for row in summary_df.itertuples():
            metric = row.metric
            expected = row.expected
            got = row.got
            bound_type = row.bound_type

            if bound_type == 'lower_bounds':
                assert got >= expected, f'[{self.name}] - {metric} outside of lower bound - {got} >= {expected}'  # type: ignore
            elif bound_type == 'upper_bounds':
                assert got <= expected, f'[{self.name}] - {metric} outside of upper bound - {got} <= {expected}'  # type: ignore

    def _evaluate_warnings(self, summary_df: pd.DataFrame):
        for row in summary_df.itertuples():
            metric = row.metric
            expected = row.expected
            got = row.got
            bound_type = row.bound_type

            if bound_type == 'lower_warns':
                warnings.warn(f'[{self.name}] - {metric} outside of lower bound - {got} >= {expected}', stacklevel=0)
            elif bound_type == 'upper_warns':
                warnings.warn(f'[{self.name}] - {metric} outside of upper bound - {got} <= {expected}', stacklevel=0)

def get_metric(df: pd.DataFrame, metric: str) -> np.ndarray:
    return df[df['metric'] == metric]['value'].to_numpy()


def aggregate(values: np.ndarray, name: str = 'last_100_mean') -> float:
    if name == 'last_100_mean':
        return values[-100:].mean()
    elif name == 'mean':
        return values.mean()
    elif name == 'max':
        return values.max()
    elif name == 'min':
        return values.min()
    else:
        raise NotImplementedError

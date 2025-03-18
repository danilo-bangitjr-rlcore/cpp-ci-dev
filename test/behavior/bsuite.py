import subprocess
from enum import Enum, auto
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from corerl.sql_logging.sql_logging import table_exists
from corerl.utils.time import now_iso


class Behaviour(Enum):
    exploration = auto()


class BSuiteTestCase:
    name: str
    config: str
    behaviours: dict[str, Behaviour] = {}
    lower_bounds: dict[str, float] = {}
    upper_bounds: dict[str, float] = {}
    goals: dict[str, float] = {}

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

    def evaluate_outcomes(self, tsdb: Engine, metrics_table: pd.DataFrame) -> pd.DataFrame:
        extracted = []
        extracted += self._extract(self.lower_bounds, 'lower_bounds', metrics_table)
        extracted += self._extract(self.upper_bounds, 'upper_bounds', metrics_table)
        extracted += self._extract(self.goals,  'goals',  metrics_table)

        summary_df = pd.DataFrame(extracted, columns=['metric', 'behaviour', 'bound_type', 'expected', 'got'])
        self._store_outcomes(tsdb, summary_df)
        self._evaluate_bounds(summary_df)

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
        for _, row in summary_df.iterrows():
            metric = row['metric']
            expected = row['expected']
            got = row['got']
            bound_type = row['bound_type']

            if bound_type == 'lower_bounds':
                assert got >= expected, f'[{self.name}] - {metric} outside of lower bound - {got} >= {expected}'
            elif bound_type == 'upper_bounds':
                assert got <= expected, f'[{self.name}] - {metric} outside of upper bound - {got} <= {expected}'


    def _store_outcomes(self, tsdb: Engine, summary_df: pd.DataFrame):
        outcomes: list[dict[str, Any]] = []
        now = now_iso()
        for _, row in summary_df.iterrows():
            # report deltaized value for goals
            v = row['got']
            if row['bound_type'] == 'goals':
                v = v - row['expected']

            outcomes.append({
                'time': now,
                'test_name': self.name.replace(' ', '_'),
                'metric': row['metric'],
                'behaviour': row['behaviour'],
                'bound_type': row['bound_type'],
                'expected': row['expected'],
                'got': v,
            })

        create_table_sql = text("""
            CREATE TABLE bsuite_outcomes (
                time TIMESTAMP WITH time zone NOT NULL,
                test_name text NOT NULL,
                metric text NOT NULL,
                behaviour text NOT NULL,
                bound_type text NOT NULL,
                expected float NOT NULL,
                got float NOT NULL
            );
            SELECT create_hypertable('bsuite_outcomes', 'time', chunk_time_interval => INTERVAL '14d');
            CREATE INDEX test_name_idx ON bsuite_outcomes (test_name);
            CREATE INDEX metric_idx ON bsuite_outcomes (metric);
            CREATE INDEX behavior_idx ON bsuite_outcomes (behaviour);
            CREATE INDEX bound_type_idx ON bsuite_outcomes (bound_type);
            ALTER TABLE bsuite_outcomes SET (
                timescaledb.compress,
                timescaledb.compress_segmentby='metric'
            );
        """)

        insert_sql = text("""
            INSERT INTO bsuite_outcomes
            (time, test_name, metric, behaviour, bound_type, expected, got)
            VALUES (TIMESTAMP WITH TIME ZONE :time, :test_name, :metric, :behaviour, :bound_type, :expected, :got)
        """)

        with tsdb.connect() as conn:
            if not table_exists(tsdb, 'bsuite_outcomes'):
                conn.execute(create_table_sql)

            conn.execute(insert_sql, outcomes)
            conn.commit()

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

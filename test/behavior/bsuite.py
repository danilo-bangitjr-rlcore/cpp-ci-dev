import json
import subprocess
from enum import Enum, auto
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

import corerl.utils.git as git
from corerl.config import MainConfig
from corerl.configs.loader import direct_load_config
from corerl.sql_logging.sql_logging import table_exists
from corerl.sql_logging.utils import SQLColumn, create_tsdb_table_query
from corerl.utils.time import now_iso


class Behaviour(Enum):
    exploration = auto()


class BSuiteTestCase:
    name: str
    config: str
    required_features: set[str] = set()
    behaviours: dict[str, Behaviour] = {}
    lower_bounds: dict[str, float] = {}
    upper_bounds: dict[str, float] = {}
    goals: dict[str, float] = {}

    aggregators: dict[str, str] = {}

    overrides: dict[str, object] | None = None

    def __init__(self):
        self._overrides = self.overrides or {}
        cfg = direct_load_config(MainConfig, base='.', config_name=self.config)
        assert isinstance(cfg, MainConfig)
        self._cfg = cfg
        self.seed = np.random.randint(0, 1_000_000)

    def execute_test(self, tsdb: Engine, db_name: str, schema: str, features: dict[str, bool]):
        ip = tsdb.url.host
        port = tsdb.url.port

        feature_overrides = {
            f'feature_flags.{k}': v for k, v in features.items() if k != 'base'
        }

        overrides = self._overrides | {
            'infra.db.ip': ip,
            'infra.db.port': port,
            'infra.db.db_name': db_name,
            'infra.db.schema': schema,
            'experiment.num_threads': 1,
            'experiment.seed': self.seed,
            'metrics.enabled': True,
            'xy_metrics.enabled': True,
            'evals.enabled': True,
        } | feature_overrides

        parts = [f'{k}={v}' for k, v in overrides.items()]

        proc = subprocess.run([
            'python', 'main.py',
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

    def evaluate_outcomes(self, tsdb: Engine, metrics_table: pd.DataFrame, features: dict[str, bool]) -> pd.DataFrame:
        extracted = []
        extracted += self._extract(self.lower_bounds, 'lower_bounds', metrics_table)
        extracted += self._extract(self.upper_bounds, 'upper_bounds', metrics_table)
        extracted += self._extract(self.goals,  'goals',  metrics_table)

        summary_df = pd.DataFrame(extracted, columns=['metric', 'behaviour', 'bound_type', 'expected', 'got'])
        self._store_outcomes(tsdb, summary_df, features)
        self._evaluate_bounds(summary_df)

        return summary_df

    def summarize_over_time(self, metric: str, metrics_table: pd.DataFrame) -> float:
        values = get_metric(metrics_table, metric)
        aggregation_name = self.aggregators.get(metric, 'last_100_mean')
        aggregated_values = self.aggregate(values, name=aggregation_name)
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


    def _store_outcomes(self, tsdb: Engine, summary_df: pd.DataFrame, features: dict[str, bool]):
        feature_json = {
            'enabled_features': [
                feature for feature, enabled in features.items() if enabled
            ],
        }

        outcomes: list[dict[str, Any]] = []
        now = now_iso()
        branch = git.get_active_branch()
        for _, row in summary_df.iterrows():
            outcomes.append({
                'time': now,
                'test_name': self.name.replace(' ', '_'),
                'metric': row['metric'],
                'behaviour': row['behaviour'],
                'bound_type': row['bound_type'],
                'seed': self.seed,
                'branch': branch,
                'expected': row['expected'],
                'got': row['got'],
                'features': json.dumps(feature_json),
            })

        create_table_sql = create_tsdb_table_query(
            schema='public',
            table='bsuite_outcomes',
            columns=[
                SQLColumn(name='time', type='TIMESTAMP WITH TIME ZONE', nullable=False),
                SQLColumn(name='test_name', type='TEXT', nullable=False),
                SQLColumn(name='metric', type='TEXT', nullable=False),
                SQLColumn(name='behaviour', type='TEXT', nullable=False),
                SQLColumn(name='bound_type', type='TEXT', nullable=False),
                SQLColumn(name='seed', type='INTEGER', nullable=False),
                SQLColumn(name='branch', type='TEXT', nullable=False),
                SQLColumn(name='expected', type='FLOAT', nullable=False),
                SQLColumn(name='got', type='FLOAT', nullable=False),
                SQLColumn(name='features', type='jsonb', nullable=False),
            ],
            partition_column='test_name',
            index_columns=['test_name', 'metric', 'behaviour', 'bound_type'],
            chunk_time_interval='14d',
        )

        insert_sql = text("""
            INSERT INTO bsuite_outcomes
            (time, test_name, metric, behaviour, bound_type, seed, branch, expected, got, features)
            VALUES (
                TIMESTAMP WITH TIME ZONE :time, :test_name, :metric, :behaviour,
                          :bound_type, :seed, :branch, :expected, :got, :features
            )
        """)

        with tsdb.connect() as conn:
            if not table_exists(tsdb, 'bsuite_outcomes'):
                conn.execute(create_table_sql)

            conn.execute(insert_sql, outcomes)
            conn.commit()

    def aggregate(self, values: np.ndarray, name: str = 'last_100_mean') -> float:
        if name == 'last_100_mean':
            return values[-100:].mean()
        elif name == 'mean':
            return values.mean()
        elif name == 'max':
            return values.max()
        elif name == 'min':
            return values.min()
        elif name == 'percent_of_steps':
            return float(np.sum(values > 0) / self._cfg.experiment.max_steps)
        else:
            raise NotImplementedError

def get_metric(df: pd.DataFrame, metric: str) -> np.ndarray:
    return df[df['metric'] == metric]['value'].to_numpy()

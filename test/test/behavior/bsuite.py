import json
import subprocess
import time
from datetime import UTC, datetime, timedelta
from enum import Enum, auto
from typing import Any

import numpy as np
import pandas as pd
import psutil
from corerl.config import MainConfig
from corerl.configs.loader import direct_load_config
from corerl.sql_logging.sql_logging import add_retention_policy, table_exists
from corerl.sql_logging.utils import SQLColumn, create_tsdb_table_query
from corerl.utils import git
from corerl.utils.time import now_iso
from sqlalchemy import Engine, text


class Behaviour(Enum):
    exploration = auto()

class BehaviourCategory(Enum):
    PERFORMANCE = auto()
    NONSTATIONARY = auto()
    REACTIVITY = auto()
    REPRESENTATION = auto()
    ROBUSTNESS = auto()
    GREEDY = auto()

class BSuiteTestCase:
    name: str
    config: str
    setup_cfgs: list[str] = []
    required_features: set[str] = set()
    behaviours: dict[str, Behaviour] = {}
    lower_bounds: dict[str, float] = {}
    upper_bounds: dict[str, float] = {}
    goals: dict[str, float] = {}
    category: set[BehaviourCategory] = set()
    aggregators: dict[str, str] = {}

    overrides: dict[str, object] | None = None

    def __init__(self):
        self._overrides = self.overrides or {}
        cfg = direct_load_config(MainConfig, config_name=self.config)
        assert isinstance(cfg, MainConfig)
        self._cfg = cfg
        self.seed = np.random.randint(0, 1_000_000)

    def _test_infra_overrides(self, tsdb: Engine, db_name: str, schema: str) -> dict[str, object]:
        ip = tsdb.url.host
        port = tsdb.url.port

        return {
            'infra.db.ip': ip,
            'infra.db.port': port,
            'infra.db.db_name': db_name,
            'infra.db.schema': schema,
            'infra.num_threads': 1,
            'seed': self.seed,
            'metrics.enabled': True,
            'evals.enabled': True,
            'silent': True,
        }

    def setup(self, engine: Engine, infra_overrides: dict[str, object], feature_overrides: dict[str, bool]):
        """
        Setup the given BSuiteTestCase before main.py is called in execute_test()
        """

    def execute_test(self, tsdb: Engine, db_name: str, schema: str, features: dict[str, bool]):
        infra_overrides = self._test_infra_overrides(tsdb, db_name, schema)

        feature_overrides = {
            f'feature_flags.{k}': v for k, v in features.items() if k != 'base'
        }

        self.setup(tsdb, infra_overrides, feature_overrides)

        overrides = self._overrides | infra_overrides | feature_overrides

        parts = [f'{k}={v}' for k, v in overrides.items()]

        start = datetime.now(UTC)
        exec_start = time.time()
        max_memory = 0

        proc = subprocess.Popen([
            'python', 'corerl/main.py',
            '--base', '../test/',
            '--config-name', self.config,
            *parts,
        ], cwd='../corerl')

        psutil_proc = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                memory_info = psutil_proc.memory_info()
                max_memory = max(max_memory, memory_info.rss / 1024 / 1024)  # convert to MB
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, proc.args)

        exec_end = time.time()
        exec_time = exec_end - exec_start

        # ensure metrics table exists
        assert table_exists(tsdb, 'metrics', schema=schema)

        with tsdb.connect() as conn:
            # ensure some metrics were logged to table
            metrics_table = pd.read_sql_table('metrics', schema=schema, con=conn)
            metrics_table = metrics_table[metrics_table['time'] > (start - timedelta(minutes=1))]

            # ensure tables have retention policies
            add_retention_policy(conn, 'metrics', schema, days=3)
            add_retention_policy(conn, 'evals', schema, days=3)

        metrics_table = metrics_table.sort_values('agent_step', ascending=True)
        return metrics_table, (exec_time, max_memory)

    def evaluate_outcomes(self, tsdb: Engine, metrics_table: pd.DataFrame, features: dict[str, bool],
                          runtime_info: tuple[float, float]) -> pd.DataFrame:
        extracted = []
        extracted += self._extract(self.lower_bounds, 'lower_bounds', metrics_table)
        extracted += self._extract(self.upper_bounds, 'upper_bounds', metrics_table)
        extracted += self._extract(self.goals,  'goals',  metrics_table)

        summary_df = pd.DataFrame(extracted, columns=['metric', 'behaviour', 'bound_type', 'expected', 'got'])
        summary_df['exec_time'] = runtime_info[0]
        summary_df['max_memory'] = runtime_info[1]
        self._store_outcomes(tsdb, summary_df, features, runtime_info)
        self._evaluate_bounds(summary_df)

        return summary_df

    def summarize_over_time(self, metric: str, metrics_table: pd.DataFrame) -> float:
        values = get_metric(metrics_table, metric)
        aggregation_name = self.aggregators.get(metric, 'last_100_mean')
        return self.aggregate(values, name=aggregation_name)


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


    def _store_outcomes(self, tsdb: Engine, summary_df: pd.DataFrame, features: dict[str, bool],
                        runtime_info: tuple[float, float]):
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

        create_runtime_table_sql = create_tsdb_table_query(
            schema='public',
            table='bsuite_metadata',
            columns= [
                SQLColumn(name='time', type='TIMESTAMP WITH TIME ZONE', nullable=False),
                SQLColumn(name='test_name', type='TEXT', nullable=False),
                SQLColumn(name='branch', type='TEXT', nullable=False),
                SQLColumn(name='exec_time', type='FLOAT', nullable=False),
                SQLColumn(name='max_memory', type='FLOAT', nullable=False),
                SQLColumn(name='features', type='jsonb', nullable=False),
            ],
            partition_column='test_name',
            index_columns=['test_name', 'branch'],
            chunk_time_interval='14d',
        )

        insert_runtime_sql = text("""
            INSERT INTO bsuite_metadata
            (time, test_name, branch, exec_time, max_memory, features)
            VALUES (
                TIMESTAMP WITH TIME ZONE :time,
                :test_name,
                :branch,
                :exec_time,
                :max_memory,
                :features
            )
            )
        """)

        exec_time, max_memory = runtime_info
        runtime_data = {
            'time': now,
            'test_name': self.name,
            'branch': branch,
            'exec_time': exec_time,
            'max_memory': max_memory,
            'features': json.dumps(feature_json),
        }

        with tsdb.connect() as conn:
            if not table_exists(tsdb, 'bsuite_outcomes'):
                conn.execute(create_table_sql)

            conn.execute(insert_sql, outcomes)
            conn.commit()

            if not table_exists(tsdb, 'bsuite_runtime'):
                conn.execute(create_runtime_table_sql)

            conn.execute(insert_runtime_sql, runtime_data)
            conn.commit()

    def aggregate(self, values: np.ndarray, name: str = 'last_100_mean') -> float:
        if name == 'last_100_mean':
            return values[-100:].mean()
        if name == 'mean':
            return values.mean()
        if name == 'max':
            return values.max()
        if name == 'min':
            return values.min()
        if name == 'percent_of_steps':
            assert self._cfg.max_steps is not None
            return float(np.sum(values > 0) / self._cfg.max_steps)
        raise NotImplementedError

def get_metric(df: pd.DataFrame, metric: str) -> np.ndarray:
    return df[df['metric'] == metric]['value'].to_numpy()

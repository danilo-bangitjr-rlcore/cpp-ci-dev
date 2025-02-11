import subprocess

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine

from corerl.sql_logging.sql_logging import table_exists


@pytest.mark.parametrize('config,overrides,expected_outcomes', [
    ('config.yaml', {},   {'reward': -0.2, 'avg_critic_loss': 0.02, 'actor_loss': -1.0}),
    ('delta.yaml',  {},   {'reward': -0.3, 'avg_critic_loss': 0.005, 'actor_loss': -1.0}),
])
@pytest.mark.timeout(900)
def test_saturation(
    config: str,
    overrides: dict[str, object],
    expected_outcomes: dict[str, float],
    tsdb_engine: Engine,
    tsdb_tmp_db_name: str,

):
    port = tsdb_engine.url.port
    assert port is not None

    overrides |= {
        'metrics.port': port,
        'metrics.db_name': tsdb_tmp_db_name,
        'evals.port': port,
        'evals.db_name': tsdb_tmp_db_name,
    }
    parts = [f'{k}={v}' for k, v in overrides.items()]

    proc = subprocess.run([
        'uv', 'run', 'python', 'main.py',
        '--base', 'test/behavior/saturation/',
        '--config-name', config,
    ] + parts)
    proc.check_returncode()

    # ensure metrics table exists
    assert table_exists(tsdb_engine, 'metrics')

    # ensure evals table exists
    assert table_exists(tsdb_engine, 'evals')

    # ensure some metrics were logged to table
    with tsdb_engine.connect() as conn:
        metrics = pd.read_sql_table('metrics', con=conn)
        evals = pd.read_sql_table('evals', con=conn)

    metrics = metrics.sort_values('agent_step', ascending=True)

    for metric, expected in expected_outcomes.items():
        values = get_metric(metrics, metric)

        if metric in {'reward'}:
            assert values[-100:].mean() >= expected
        else:
            assert values[-100:].mean() <= expected

    assert len(evals) > 0


def get_metric(df: pd.DataFrame, metric: str) -> np.ndarray:
    return df[df['metric'] == metric]['value'].to_numpy()

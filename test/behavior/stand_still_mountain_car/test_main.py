import subprocess

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine

from corerl.sql_logging.sql_logging import table_exists


@pytest.mark.parametrize('overrides,expected_outcomes', [
    ({}, {'reward': -0.05, 'avg_critic_loss': 0.03, 'actor_loss': -1.0}),
    ({'experiment.gamma': 0}, {'reward': -0.05, 'avg_critic_loss': 0.003, 'actor_loss': -0.5}),
])
@pytest.mark.timeout(600)
def test_stand_still_mountain_car(
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
    }
    parts = [f'{k}={v}' for k, v in overrides.items()]

    proc = subprocess.run([
        'uv', 'run', 'python', 'main.py',
        '--base', 'test/behavior/stand_still_mountain_car/',
        '--config-name', 'config.yaml',
    ] + parts)
    proc.check_returncode()

    # ensure metrics table exists
    assert table_exists(tsdb_engine, 'metrics')

    # ensure some metrics were logged to table
    with tsdb_engine.connect() as conn:
        metrics = pd.read_sql_table('metrics', con=conn)

    metrics = metrics.sort_values('agent_step', ascending=True)

    # these values are based on running the agent
    # 10 times and selecting the "worst" values of the
    # 10 runs. This should be reasonably stable, and these
    # values (particularly the reward) illustrate reasonable
    # learning performance.
    for metric, expected in expected_outcomes.items():
        values = get_metric(metrics, metric)

        if metric in {'reward'}:
            assert values[-100:].mean() >= expected
        else:
            assert values[-100:].mean() <= expected


def get_metric(df: pd.DataFrame, metric: str) -> np.ndarray:
    return df[df['metric'] == metric]['value'].to_numpy()

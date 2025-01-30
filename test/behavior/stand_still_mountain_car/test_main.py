import subprocess

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine

from corerl.sql_logging.sql_logging import table_exists


@pytest.mark.timeout(600)
def test_stand_still_mountain_car(
    tsdb_engine: Engine,
    tsdb_tmp_db_name: str,
):
    port = tsdb_engine.url.port
    assert port is not None

    overrides = {
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
    rewards = get_metric(metrics, 'reward')
    critic_loss = get_metric(metrics, 'critic_loss')
    actor_loss = get_metric(metrics, 'actor_loss')

    assert rewards[-100:].mean() >= -0.01
    assert critic_loss[-100:].mean() <= 0.01
    assert actor_loss[-100:].mean() <= -1.5


def get_metric(df: pd.DataFrame, metric: str) -> np.ndarray:
    return df[df['metric'] == metric]['value'].to_numpy()

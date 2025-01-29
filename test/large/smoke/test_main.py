import subprocess

import pandas as pd
import pytest
from sqlalchemy import Engine

from corerl.sql_logging.sql_logging import table_exists


@pytest.mark.parametrize('config_name', [
    'pendulum',
    'saturation',
    'mountain_car_continuous',
])
@pytest.mark.timeout(120)
def test_main_configs(
    tsdb_engine: Engine,
    config_name: str,
    tsdb_tmp_db_name: str,
):
    """
    Should be able to execute the main script for several configs
    without error. If an error code is returned (i.e. the process crashes),
    then test fails.

    This test does no checking of result validity.
    """
    port = tsdb_engine.url.port
    assert port is not None

    proc = subprocess.run([
        'uv', 'run', 'python', 'main.py',
        '--config-name', f'{config_name}', 'experiment.max_steps=25',
        f'metrics_writer.port={port}', 'metrics_writer.enabled=True', 'metrics_writer.name=db',
        f'metrics_writer.db_name={tsdb_tmp_db_name}', 'metrics_writer.lo_wm=1',
    ])
    proc.check_returncode()

    # ensure metrics table exists
    assert table_exists(tsdb_engine, 'metrics')

    # ensure some metrics were logged to table
    with tsdb_engine.connect() as conn:
        metrics = pd.read_sql_table('metrics', con=conn)
        assert len(metrics) >= 50

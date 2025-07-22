import datetime as dt
from copy import deepcopy

import pandas as pd
import pytest
import pytz
from sqlalchemy import Engine

from corerl.eval.metrics import MetricsDBConfig, MetricsTable


@pytest.fixture()
def db_metrics_table(
    tsdb_engine: Engine,
    tsdb_tmp_db_name: str,
):
    port = tsdb_engine.url.port
    assert port is not None

    metrics_db_cfg = MetricsDBConfig(
        enabled=True,
        drivername='postgresql+psycopg2',
        username='postgres',
        password='password',
        ip='localhost',
        port=port,
        db_name=tsdb_tmp_db_name,
        table_schema='public',
    )

    metrics_table = MetricsTable(metrics_db_cfg)

    yield metrics_table

    metrics_table.close()

@pytest.fixture()
def populated_metrics_table(db_metrics_table: MetricsTable):
    start_time = dt.datetime(2023, 7, 13, 6, tzinfo=pytz.UTC)
    curr_time = deepcopy(start_time)
    delta = dt.timedelta(hours=1)
    for i in range(5):
        db_metrics_table.write(
            agent_step=i,
            metric="q",
            value=i,
            timestamp=curr_time.isoformat(),
        )
        db_metrics_table.write(
            agent_step=i,
            metric="reward",
            value=2*i,
            timestamp=curr_time.isoformat(),
        )
        curr_time += delta
    db_metrics_table.blocking_sync()
    return db_metrics_table, start_time, delta

def test_db_metrics_read_by_time(
    tsdb_engine: Engine,
    populated_metrics_table: tuple[MetricsTable, dt.datetime, dt.timedelta],
):
    db_metrics_table, start_time, delta = populated_metrics_table

    with tsdb_engine.connect() as conn:
        metrics_df = pd.read_sql_table('metrics', con=conn)

    # Check that all rows were written
    assert len(metrics_df) == 10  # 5 'q' + 5 'reward'

    # Read metrics table by timestamp
    start_ind = 1
    end_ind = 3
    query_start = start_time + (start_ind * delta)
    query_end = start_time + (end_ind * delta)
    rewards_df = db_metrics_table.read("reward", start_time=query_start, end_time=query_end)
    q_df = db_metrics_table.read("q", start_time=query_start, end_time=query_end)

    assert len(rewards_df) == end_ind - start_ind + 1
    assert len(q_df) == end_ind - start_ind + 1
    for i in range(start_ind, end_ind + 1):
        assert rewards_df.iloc[i - start_ind]["time"] == pd.Timestamp(start_time + (i * delta))
        assert rewards_df.iloc[i - start_ind]["value"] == 2 * i
        assert q_df.iloc[i - start_ind]["time"] == pd.Timestamp(start_time + (i * delta))
        assert q_df.iloc[i - start_ind]["value"] == i

def test_db_metrics_read_by_step(
    populated_metrics_table: tuple[MetricsTable, dt.datetime, dt.timedelta],
):
    db_metrics_table, _, _ = populated_metrics_table

    # Read metrics table by agent_step
    start_step = None
    end_step = 3
    rewards_df = db_metrics_table.read("reward", step_start=start_step, step_end=end_step)
    q_df = db_metrics_table.read("q", step_start=start_step, step_end=end_step)

    assert len(rewards_df) == end_step + 1
    assert len(q_df) == end_step + 1
    for i in range(end_step + 1):
        assert rewards_df.iloc[i]["agent_step"] == i
        assert rewards_df.iloc[i]["value"] == 2 * i
        assert q_df.iloc[i]["agent_step"] == i
        assert q_df.iloc[i]["value"] == i

def test_db_metrics_read_by_metric(
    populated_metrics_table: tuple[MetricsTable, dt.datetime, dt.timedelta],
):
    db_metrics_table, _, _ = populated_metrics_table

    rewards_df = db_metrics_table.read("reward")
    q_df = db_metrics_table.read("q")

    assert len(rewards_df) == 5
    assert len(q_df) == 5
    for i in range(5):
        assert rewards_df.iloc[i]["agent_step"] == i
        assert rewards_df.iloc[i]["value"] == 2 * i
        assert q_df.iloc[i]["agent_step"] == i
        assert q_df.iloc[i]["value"] == i



def test_disconnect_between_writes(tsdb_engine: Engine, db_metrics_table: MetricsTable):
    # 1. Start the database and connect (handled by fixture)
    # 2. Write a few metrics
    db_metrics_table.write(agent_step=0, metric="q", value=1.0, timestamp="2023-07-13T06:00:00+00:00")
    db_metrics_table.write(agent_step=0, metric="reward", value=2.0, timestamp="2023-07-13T06:00:00+00:00")
    db_metrics_table.blocking_sync()

    # 3. Simulate an accidental disconnect
    tsdb_engine.dispose()

    # 4. Attempt to write more metrics (should be buffered, not written immediately)
    db_metrics_table.write(agent_step=1, metric="q", value=2.0, timestamp="2023-07-13T07:00:00+00:00")
    db_metrics_table.write(agent_step=1, metric="reward", value=4.0, timestamp="2023-07-13T07:00:00+00:00")

    # 5. Sync and verify all metrics are accessible
    db_metrics_table.blocking_sync()
    with tsdb_engine.connect() as conn:
        metrics_df = pd.read_sql_table('metrics', con=conn)

    assert len(metrics_df) >= 4
    assert ((metrics_df['agent_step'] == 0) & (metrics_df['metric'] == 'q')).any()
    assert ((metrics_df['agent_step'] == 0) & (metrics_df['metric'] == 'reward')).any()
    assert ((metrics_df['agent_step'] == 1) & (metrics_df['metric'] == 'q')).any()
    assert ((metrics_df['agent_step'] == 1) & (metrics_df['metric'] == 'reward')).any()

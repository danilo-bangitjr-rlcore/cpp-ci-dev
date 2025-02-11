import datetime as dt
from copy import deepcopy

import pandas as pd
import pytest
import pytz
from sqlalchemy import Engine

from corerl.eval.metrics import MetricsDBConfig, MetricsTable, PandasMetricsConfig, PandasMetricsTable
from corerl.sql_logging.sql_logging import table_exists
from corerl.utils.time import now_iso


@pytest.fixture()
def db_metrics_table(
    tsdb_engine: Engine,
    tsdb_tmp_db_name: str
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
        lo_wm=1,
    )

    metrics_table = MetricsTable(metrics_db_cfg)

    yield metrics_table

    metrics_table.close()

@pytest.fixture()
def pandas_metrics_table() -> PandasMetricsTable:
    pandas_metrics_cfg = PandasMetricsConfig(
        enabled=True,
        buffer_size=1
    )

    metrics_table = PandasMetricsTable(pandas_metrics_cfg)

    return metrics_table

def test_db_metrics_writer(tsdb_engine: Engine, db_metrics_table: MetricsTable):
    metrics_val = 1.5
    db_metrics_table.write(
        agent_step=0,
        metric="q",
        value=metrics_val,
        timestamp=now_iso()
    )
    db_metrics_table.blocking_sync()

    # ensure metrics table exists
    assert table_exists(tsdb_engine, 'metrics')

    # Ensure the entry written above exists in the metrics table
    with tsdb_engine.connect() as conn:
        metrics_df = pd.read_sql_table('metrics', con=conn)
        assert len(metrics_df) == 1

        entry = metrics_df.iloc[0]
        assert entry["agent_step"] == 0
        assert entry["metric"] == "q"
        assert entry["value"] == 1.5

def test_db_metrics_read_by_time(tsdb_engine: Engine, db_metrics_table: MetricsTable):
    start_time = dt.datetime(2023, 7, 13, 6, tzinfo=pytz.UTC)
    curr_time = deepcopy(start_time)
    delta = dt.timedelta(hours=1)
    for i in range(5):
        db_metrics_table.write(
            agent_step=i,
            metric="q",
            value=i,
            timestamp=curr_time.isoformat()
        )
        db_metrics_table.write(
            agent_step=i,
            metric="reward",
            value=2*i,
            timestamp=curr_time.isoformat()
        )
        curr_time += delta
    db_metrics_table.blocking_sync()

    # ensure metrics table exists
    assert table_exists(tsdb_engine, 'metrics')

    # Read metrics tably by timestamp
    start_ind = 1
    end_ind = 3
    query_start = start_time + (start_ind * delta)
    query_end = start_time + (end_ind * delta)
    rewards_df = db_metrics_table.read("reward", start_time=query_start, end_time=query_end)
    q_df = db_metrics_table.read("q", start_time=query_start, end_time=query_end)

    # Ensure the correct entries are in the read DFs
    assert len(rewards_df) == end_ind - start_ind + 1
    assert len(q_df) == end_ind - start_ind + 1
    for i in range(start_ind, end_ind + 1):
        assert rewards_df.iloc[i - start_ind]["time"] == pd.Timestamp(start_time + (i * delta))
        assert rewards_df.iloc[i - start_ind]["value"] == 2 * i
        assert q_df.iloc[i - start_ind]["time"] == pd.Timestamp(start_time + (i * delta))
        assert q_df.iloc[i - start_ind]["value"] == i

def test_db_metrics_read_by_step(tsdb_engine: Engine, db_metrics_table: MetricsTable):
    start_time = dt.datetime(2023, 7, 13, 6, tzinfo=pytz.UTC)
    curr_time = deepcopy(start_time)
    delta = dt.timedelta(hours=1)
    for i in range(5):
        db_metrics_table.write(
            agent_step=i,
            metric="q",
            value=i,
            timestamp=curr_time.isoformat()
        )
        db_metrics_table.write(
            agent_step=i,
            metric="reward",
            value=2*i,
            timestamp=curr_time.isoformat()
        )
        curr_time += delta
    db_metrics_table.blocking_sync()

    # ensure metrics table exists
    assert table_exists(tsdb_engine, 'metrics')

    # Read metrics tably by agent_step
    start_step = None
    end_step = 3
    rewards_df = db_metrics_table.read("reward", step_start=start_step, step_end=end_step)
    q_df = db_metrics_table.read("q", step_start=start_step, step_end=end_step)

    # Ensure the correct entries are in the read DFs
    assert len(rewards_df) == end_step + 1
    assert len(q_df) == end_step + 1
    for i in range(end_step + 1):
        assert rewards_df.iloc[i]["agent_step"] == i
        assert rewards_df.iloc[i]["value"] == 2 * i
        assert q_df.iloc[i]["agent_step"] == i
        assert q_df.iloc[i]["value"] == i

def test_db_metrics_read_by_metric(tsdb_engine: Engine, db_metrics_table: MetricsTable):
    start_time = dt.datetime(2023, 7, 13, 6, tzinfo=pytz.UTC)
    curr_time = deepcopy(start_time)
    delta = dt.timedelta(hours=1)
    steps = 5
    for i in range(steps):
        db_metrics_table.write(
            agent_step=i,
            metric="reward",
            value=2*i,
            timestamp=curr_time.isoformat()
        )
        db_metrics_table.write(
            agent_step=i,
            metric="q",
            value=i,
            timestamp=curr_time.isoformat()
        )
        curr_time += delta
    db_metrics_table.blocking_sync()

    # ensure metrics table exists
    assert table_exists(tsdb_engine, 'metrics')

    # Read metrics table by metric
    rewards_df = db_metrics_table.read("reward")
    q_df = db_metrics_table.read("q")

    # Ensure the correct entries are in the read DFs
    assert len(rewards_df) == steps
    assert len(q_df) == steps
    for i in range(steps):
        assert rewards_df.iloc[i]["agent_step"] == i
        assert rewards_df.iloc[i]["value"] == 2 * i
        assert q_df.iloc[i]["agent_step"] == i
        assert q_df.iloc[i]["value"] == i

def test_pandas_metrics_read_by_metric(pandas_metrics_table: PandasMetricsTable):
    start_time = dt.datetime(2023, 7, 13, 6, tzinfo=pytz.UTC)
    curr_time = deepcopy(start_time)
    delta = dt.timedelta(hours=1)
    steps = 5
    for i in range(steps):
        pandas_metrics_table.write(
            agent_step=i,
            metric="reward",
            value=2*i,
            timestamp=curr_time.isoformat()
        )
        pandas_metrics_table.write(
            agent_step=i,
            metric="q",
            value=i,
            timestamp=curr_time.isoformat()
        )
        curr_time += delta
    pandas_metrics_table.close()

    # Read metrics table by metric
    rewards_df = pandas_metrics_table.read("reward")
    q_df = pandas_metrics_table.read("q")

    # Ensure the correct entries are in the read DFs
    assert len(rewards_df) == steps
    assert len(q_df) == steps
    for i in range(steps):
        assert rewards_df.iloc[i]["agent_step"] == i
        assert rewards_df.iloc[i]["value"] == 2 * i
        assert q_df.iloc[i]["agent_step"] == i
        assert q_df.iloc[i]["value"] == i

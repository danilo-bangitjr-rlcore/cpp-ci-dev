import datetime as dt
from copy import deepcopy

import pandas as pd
import pytest
import pytz
from sqlalchemy import Engine

from corerl.eval.metrics import MetricsDBConfig, MetricsTable
from corerl.utils.buffered_sql_writer import WatermarkSyncConfig


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


# ---------------------------------------------------------------------------- #
#                                  Wide Tests                                  #
# ---------------------------------------------------------------------------- #

@pytest.fixture()
def db_metrics_table_wide(
    tsdb_engine: Engine,
    tsdb_tmp_db_name: str,
):
    port = tsdb_engine.url.port
    assert port is not None

    metrics_db_cfg = MetricsDBConfig(
        enabled=True,
        narrow_format=False,  # Use wide format
        table_name='metrics_wide',
        drivername='postgresql+psycopg2',
        username='postgres',
        password='password',
        ip='localhost',
        port=port,
        db_name=tsdb_tmp_db_name,
        table_schema='public',
        watermark_cfg=WatermarkSyncConfig('watermark', 1, 10),
    )

    metrics_table = MetricsTable(metrics_db_cfg)

    yield metrics_table

    metrics_table.close()

@pytest.fixture()
def populated_metrics_table_wide(db_metrics_table_wide: MetricsTable):
    start_time = dt.datetime(2023, 7, 13, 6, tzinfo=pytz.UTC)
    curr_time = deepcopy(start_time)
    delta = dt.timedelta(hours=1)
    for i in range(5):
        db_metrics_table_wide.write(
            agent_step=i,
            metric="q",
            value=i,
            timestamp=curr_time.isoformat(),
        )
        db_metrics_table_wide.write(
            agent_step=i,
            metric="reward",
            value=2*i,
            timestamp=curr_time.isoformat(),
        )
        curr_time += delta
        db_metrics_table_wide.blocking_sync()

    # add two reward values without associated q
    db_metrics_table_wide.write(
        agent_step=5,
        metric="reward",
        value=2*5,
        timestamp=curr_time.isoformat(),
    )
    db_metrics_table_wide.write(
        agent_step=6,
        metric="reward",
        value=2*6,
        timestamp=curr_time.isoformat(),
    )

    db_metrics_table_wide.blocking_sync()
    return db_metrics_table_wide, start_time, delta


def test_db_metrics_write_wide(
    tsdb_engine: Engine,
    populated_metrics_table_wide: tuple[MetricsTable, dt.datetime, dt.timedelta],
):
    _, start_time, delta = populated_metrics_table_wide

    with tsdb_engine.connect() as conn:
        metrics_df = pd.read_sql_table('metrics_wide', con=conn)

    assert len(metrics_df) == 6

    # Create expected dataframe for wide format
    agent_steps = [0, 1, 2, 3, 4, 6]
    times = [
        pd.Timestamp(start_time),
        pd.Timestamp(start_time + delta),
        pd.Timestamp(start_time + (2 * delta)),
        pd.Timestamp(start_time + (3 * delta)),
        pd.Timestamp(start_time + (4 * delta)),
        pd.Timestamp(start_time + (5 * delta)),
    ]
    q_values = [0.0, 1.0, 2.0, 3.0, 4.0, None] # no final q value
    reward_values = [0.0, 2.0, 4.0, 6.0, 8.0, 11.0] # final reward is mean(10, 12)=11
    expected_df = pd.DataFrame({
        'time': times,
        'agent_step': agent_steps,
        'q': q_values,
        'reward': reward_values,
    })

    pd.testing.assert_frame_equal(
        metrics_df,
        expected_df,
    )


def test_db_metrics_read_by_time_wide(
    tsdb_engine: Engine,
    populated_metrics_table_wide: tuple[MetricsTable, dt.datetime, dt.timedelta],
):
    db_metrics_table, start_time, delta = populated_metrics_table_wide

    with tsdb_engine.connect() as conn:
        metrics_df = pd.read_sql_table('metrics_wide', con=conn)

    # Check that all rows were written (6 rows total)
    assert len(metrics_df) == 6

    # Read metrics table by timestamp - should return wide format
    start_ind = 1
    end_ind = 3
    query_start = start_time + (start_ind * delta)
    query_end = start_time + (end_ind * delta)

    # In wide format, we read by metric but get time-filtered results
    rewards_df = db_metrics_table.read("reward", start_time=query_start, end_time=query_end)
    q_df = db_metrics_table.read("q", start_time=query_start, end_time=query_end)

    # Expected data for time range (agent_steps 1, 2, 3)
    expected_times = [
        pd.Timestamp(start_time + delta),
        pd.Timestamp(start_time + (2 * delta)),
        pd.Timestamp(start_time + (3 * delta)),
    ]
    expected_reward_df = pd.DataFrame({
        'time': expected_times,
        'reward': [2.0, 4.0, 6.0],
    })
    expected_q_df = pd.DataFrame({
        'time': expected_times,
        'q': [1.0, 2.0, 3.0],
    })

    pd.testing.assert_frame_equal(rewards_df, expected_reward_df)
    pd.testing.assert_frame_equal(q_df, expected_q_df)

def test_db_metrics_read_by_step_wide(
    populated_metrics_table_wide: tuple[MetricsTable, dt.datetime, dt.timedelta],
):
    db_metrics_table, _, _ = populated_metrics_table_wide

    # Read metrics table by agent_step - wide format returns metric columns
    start_step = None
    end_step = 3
    rewards_df = db_metrics_table.read("reward", step_start=start_step, step_end=end_step)
    q_df = db_metrics_table.read("q", step_start=start_step, step_end=end_step)

    # Expected data for agent_steps 0, 1, 2, 3
    expected_reward_df = pd.DataFrame({
        'agent_step': [0, 1, 2, 3],
        'reward': [0.0, 2.0, 4.0, 6.0],
    })
    expected_q_df = pd.DataFrame({
        'agent_step': [0, 1, 2, 3],
        'q': [0.0, 1.0, 2.0, 3.0],
    })

    pd.testing.assert_frame_equal(rewards_df, expected_reward_df)
    pd.testing.assert_frame_equal(q_df, expected_q_df)

def test_db_metrics_read_by_metric_wide(
    populated_metrics_table_wide: tuple[MetricsTable, dt.datetime, dt.timedelta],
):
    db_metrics_table, start_time, delta = populated_metrics_table_wide

    # Read all data for each metric
    rewards_df = db_metrics_table.read("reward")
    q_df = db_metrics_table.read("q")

    # Expected data for all rewards (6 entries including aggregated final entry)
    expected_times = [
            pd.Timestamp(start_time),
            pd.Timestamp(start_time + delta),
            pd.Timestamp(start_time + (2 * delta)),
            pd.Timestamp(start_time + (3 * delta)),
            pd.Timestamp(start_time + (4 * delta)),
            pd.Timestamp(start_time + (5 * delta)),
    ]
    expected_reward_df = pd.DataFrame({
        'time': expected_times,
        'agent_step': [0, 1, 2, 3, 4, 6],
        'reward': [0.0, 2.0, 4.0, 6.0, 8.0, 11.0],  # final is mean(10, 12)=11
    })
    # Expected data for all q values (5 entries)
    expected_q_df = pd.DataFrame({
        'time': expected_times,
        'agent_step': [0, 1, 2, 3, 4, 6],
        'q': [0.0, 1.0, 2.0, 3.0, 4.0, None],
    })

    pd.testing.assert_frame_equal(rewards_df, expected_reward_df)
    pd.testing.assert_frame_equal(q_df, expected_q_df)

def test_disconnect_between_writes_wide(tsdb_engine: Engine, db_metrics_table_wide: MetricsTable):
    # 1. Start the database and connect (handled by fixture)
    # 2. Write a few metrics
    db_metrics_table_wide.write(agent_step=0, metric="q", value=1.0, timestamp="2023-07-13T06:00:00+00:00")
    db_metrics_table_wide.write(agent_step=0, metric="reward", value=2.0, timestamp="2023-07-13T06:00:00+00:00")
    db_metrics_table_wide.blocking_sync()

    # 3. Simulate an accidental disconnect
    tsdb_engine.dispose()

    # 4. Attempt to write more metrics (should be buffered, not written immediately)
    db_metrics_table_wide.write(agent_step=1, metric="q", value=3.0, timestamp="2023-07-13T07:00:00+00:00")
    db_metrics_table_wide.write(agent_step=1, metric="reward", value=4.0, timestamp="2023-07-13T07:00:00+00:00")

    # 5. Sync and verify all metrics are accessible
    db_metrics_table_wide.blocking_sync()
    with tsdb_engine.connect() as conn:
        metrics_df = pd.read_sql_table('metrics_wide', con=conn)

    # Expected data: 2 rows (one per agent_step) with aggregated values
    expected_df = pd.DataFrame({
        'time': [
            pd.Timestamp("2023-07-13T06:00:00+00:00"),
            pd.Timestamp("2023-07-13T07:00:00+00:00"),
        ],
        'agent_step': [0, 1],
        'q': [1.0, 3.0],
        'reward': [2.0, 4.0],
    })

    pd.testing.assert_frame_equal(metrics_df, expected_df)

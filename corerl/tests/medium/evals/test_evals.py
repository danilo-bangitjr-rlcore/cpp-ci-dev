import datetime as dt
from copy import deepcopy

import pandas as pd
import pytz
from lib_sql.inspection import table_exists
from lib_utils.time import now_iso
from sqlalchemy import Engine

from corerl.eval.evals.static import StaticEvalsTable
from tests.infrastructure.config import load_config


def test_db_eval_writer(tsdb_engine: Engine, evals_table: StaticEvalsTable):
    eval_out = {"Q": [1, 2, 3]}
    evals_table.write(
        agent_step=0,
        evaluator="q_eval",
        value=eval_out,
        timestamp=now_iso(),
    )
    evals_table.flush()

    # ensure evals table exists
    assert table_exists(tsdb_engine, 'evals')

    # Ensure the entry written above exists in the evals table
    with tsdb_engine.connect() as conn:
        evaluations = pd.read_sql_table('evals', con=conn)
        assert len(evaluations) == 1

        entry = evaluations.iloc[0]
        assert entry["agent_step"] == 0
        assert entry["evaluator"] == "q_eval"

        val = entry["value"]
        assert "Q" in val
        assert val["Q"] == [1,2,3]

def test_db_evals_read_by_time(tsdb_engine: Engine, evals_table: StaticEvalsTable):
    start_time = dt.datetime(2023, 7, 13, 6, tzinfo=pytz.UTC)
    curr_time = deepcopy(start_time)
    delta = dt.timedelta(hours=1)
    steps = 5
    samples = 5
    for i in range(steps):
        state_v = {"state_v": [i*j for j in range(samples)]}
        evals_table.write(
            agent_step=i,
            evaluator="state_v",
            value=state_v,
            timestamp=curr_time.isoformat(),
        )
        curr_time += delta
    evals_table.flush()

    # ensure evals table exists
    assert table_exists(tsdb_engine, 'evals')

    # Read evals table by timestamp
    start_ind = 1
    query_start = start_time + (start_ind * delta)
    state_v_df = evals_table.read("state_v", start_time=query_start)

    # Ensure the correct entries are in the read DFs
    assert len(state_v_df) == steps - start_ind
    for i in range(start_ind, steps):
        assert state_v_df.iloc[i - start_ind]["time"] == pd.Timestamp(start_time + (i * delta))
        assert state_v_df.iloc[i - start_ind]["value"] == {"state_v": [i*j for j in range(samples)]}

def test_db_evals_read_by_step(tsdb_engine: Engine, evals_table: StaticEvalsTable):
    start_time = dt.datetime(2023, 7, 13, 6, tzinfo=pytz.UTC)
    curr_time = deepcopy(start_time)
    delta = dt.timedelta(hours=1)
    steps = 5
    samples = 5
    for i in range(steps):
        state_v = {"state_v": [i*j for j in range(samples)]}
        evals_table.write(
            agent_step=i,
            evaluator="state_v",
            value=state_v,
            timestamp=curr_time.isoformat(),
        )
        curr_time += delta
    evals_table.flush()

    # ensure evals table exists
    assert table_exists(tsdb_engine, 'evals')

    # Read evals table by agent step
    start_step = 2
    end_step = 3
    state_v_df = evals_table.read("state_v", step_start=start_step, step_end=end_step)

    # Ensure the correct entries are in the read DFs
    assert len(state_v_df) == end_step - start_step + 1
    for i in range(start_step, end_step + 1):
        assert state_v_df.iloc[i - start_step]["agent_step"] == i
        assert state_v_df.iloc[i - start_step]["value"] == {"state_v": [i*j for j in range(samples)]}

def test_db_evals_read_by_eval(tsdb_engine: Engine, evals_table: StaticEvalsTable):
    start_time = dt.datetime(2023, 7, 13, 6, tzinfo=pytz.UTC)
    curr_time = deepcopy(start_time)
    delta = dt.timedelta(hours=1)
    steps = 5
    samples = 5
    for i in range(steps):
        state_v = {"state_v": [i*j for j in range(samples)]}
        evals_table.write(
            agent_step=i,
            evaluator="state_v",
            value=state_v,
            timestamp=curr_time.isoformat(),
        )
        curr_time += delta
    evals_table.flush()

    # ensure evals table exists
    assert table_exists(tsdb_engine, 'evals')

    # Read evals table by evaluator
    state_v_df = evals_table.read("state_v")

    # Ensure the correct entries are in the read DFs
    assert len(state_v_df) == steps
    for i in range(steps):
        assert state_v_df.iloc[i]["time"] == pd.Timestamp(start_time + (i * delta))
        assert state_v_df.iloc[i]["agent_step"] == i
        assert state_v_df.iloc[i]["value"] == {"state_v": [i*j for j in range(samples)]}


# ============================================================================
# Integration Edge Cases Tests
# ============================================================================


def test_connection_failure_recovery(tsdb_engine: Engine, evals_table: StaticEvalsTable):
    """
    Test behavior when database connection is lost and recovered.

    Verifies that connection disposal doesn't cause crashes and that
    SQLAlchemy can automatically re-establish connections.
    """
    # Write some data successfully first
    evals_table.write(0, "test_eval", {"data": "initial"})
    evals_table.flush()

    assert table_exists(tsdb_engine, "evals")
    initial_df = evals_table.read("test_eval")
    assert len(initial_df) == 1

    # Writes should still succeed when buffered
    evals_table.write(1, "test_eval", {"data": "buffered_write"})

    # Simulate connection issues by disposing the engine connection pool
    tsdb_engine.dispose()

    # This should either succeed or fail gracefully without crashing
    evals_table.read("test_eval")

    # SQLAlchemy should automatically reconnect
    final_df = evals_table.read("test_eval")
    assert len(final_df) >= 1


def test_buffered_writer_consistency(evals_table: StaticEvalsTable):
    """
    Test that buffered writes maintain consistency during failures.

    Verifies that batch writes either succeed completely or fail atomically,
    never leaving the database in a partial/inconsistent state.
    """
    # Write initial data successfully
    evals_table.write(0, "consistency_test", {"value": "initial"})
    evals_table.flush()

    initial_df = evals_table.read("consistency_test")
    assert len(initial_df) == 1
    assert initial_df.iloc[0]["value"] == {"value": "initial"}

    # Start a batch of writes that should be atomic
    evals_table.write(1, "consistency_test", {"value": "batch1"})
    evals_table.write(2, "consistency_test", {"value": "batch2"})
    evals_table.write(3, "consistency_test", {"value": "batch3"})

    # Simulate connection failure before flush
    evals_table.engine.dispose()

    final_df = evals_table.read("consistency_test")

    # Should have either 1 (just initial) or 4 (all data) records
    # Never 2 or 3 records (partial batch would indicate inconsistency)
    assert len(final_df) in [1, 4], f"Inconsistent state: {len(final_df)} records found"


def test_engine_disposal_and_recreation(
    tsdb_engine: Engine,
    tsdb_tmp_db_name: str,
    basic_config_path: str,
):
    """
    Test engine disposal and recreation scenarios.

    Verifies that multiple StaticEvalsTable instances can be created with
    the same configuration and that data persists across engine recreation.
    """
    port = tsdb_engine.url.port
    assert port is not None

    basic_config = load_config(
        basic_config_path,
        overrides={
            "evals.db_name": tsdb_tmp_db_name,
            "evals.port": port,
        },
    )

    table1 = StaticEvalsTable(basic_config.evals)

    table1.write(0, "disposal_test", {"data": "first_instance"})
    table1.flush()

    df1 = table1.read("disposal_test")
    assert len(df1) == 1

    table1.close()

    # Create a new instance with same config (engine recreation)
    table2 = StaticEvalsTable(basic_config.evals)

    # New instance should be able to read existing data
    df2 = table2.read("disposal_test")
    assert len(df2) == 1
    assert df2.iloc[0]["value"] == {"data": "first_instance"}

    table2.write(1, "disposal_test", {"data": "second_instance"})
    table2.flush()

    df3 = table2.read("disposal_test")
    assert len(df3) == 2

    table2.close()

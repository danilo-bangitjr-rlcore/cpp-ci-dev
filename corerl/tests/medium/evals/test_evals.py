import datetime as dt
from copy import deepcopy

import pandas as pd
import pytz
from lib_sql.inspection import table_exists
from lib_utils.time import now_iso
from sqlalchemy import Engine

from corerl.eval.evals.static import StaticEvalsTable


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

import pandas as pd
from sqlalchemy import Engine

from corerl.eval.evals import EvalDBConfig, EvalWriter
from corerl.sql_logging.sql_logging import table_exists
from corerl.utils.time import now_iso


def test_db_eval_writer(tsdb_engine: Engine, tsdb_tmp_db_name: str):
    port = tsdb_engine.url.port
    assert port is not None

    eval_writer_cfg = EvalDBConfig(enabled=True,
                                   port=port,
                                   db_name=tsdb_tmp_db_name,
                                   lo_wm=0)
    eval_writer = EvalWriter(eval_writer_cfg, high_watermark=1)
    eval_out = {"Q": [1, 2, 3]}
    eval_writer.write(agent_step=0,
                      evaluator="q_eval",
                      value=eval_out,
                      timestamp=now_iso())

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

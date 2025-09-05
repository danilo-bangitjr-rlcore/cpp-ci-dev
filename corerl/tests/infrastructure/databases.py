import pytest
from sqlalchemy import Engine

from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.db.data_writer import DataWriter
from corerl.eval.evals import EvalsTable
from corerl.eval.metrics import MetricsTable
from tests.infrastructure.config import load_config


@pytest.fixture()
def data_reader_writer(
    tsdb_engine: Engine,
    tsdb_tmp_db_name: str,
    basic_config_path: str,
):
    port = tsdb_engine.url.port
    assert port is not None

    basic_config = load_config(basic_config_path, overrides={
        'infra.db.db_name': tsdb_tmp_db_name,
        'infra.db.port': port,
    })
    db = basic_config.env.db
    reader = DataReader(db_cfg=db)
    writer = DataWriter(cfg=db)
    yield (reader, writer)
    reader.close()
    writer.close()


@pytest.fixture()
def metrics_table(
    tsdb_engine: Engine,
    tsdb_tmp_db_name: str,
    basic_config_path: str,
):
    port = tsdb_engine.url.port
    assert port is not None

    basic_config = load_config(basic_config_path, overrides={
        'metrics.db_name': tsdb_tmp_db_name,
        'metrics.port': port,
    })
    table = MetricsTable(basic_config.metrics)
    yield table
    table.close()


@pytest.fixture()
def evals_table(
    tsdb_engine: Engine,
    tsdb_tmp_db_name: str,
    basic_config_path: str,
):
    port = tsdb_engine.url.port
    assert port is not None

    basic_config = load_config(basic_config_path, overrides={
        'evals.db_name': tsdb_tmp_db_name,
        'evals.port': port,
    })
    table = EvalsTable(basic_config.evals)
    yield table
    table.close()

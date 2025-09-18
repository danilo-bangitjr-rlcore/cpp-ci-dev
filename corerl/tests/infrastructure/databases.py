import pytest
from sqlalchemy import Engine

from corerl.data_pipeline.db.data_reader import DataReader
from corerl.data_pipeline.db.data_writer import DataWriter
from corerl.eval.evals.factory import create_evals_writer
from corerl.eval.metrics.factory import create_metrics_writer
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
    table = create_metrics_writer(basic_config.metrics)
    yield table
    table.close()


@pytest.fixture()
def wide_metrics_table(
    tsdb_engine: Engine,
    tsdb_tmp_db_name: str,
    basic_config_path: str,
):
    port = tsdb_engine.url.port
    assert port is not None

    basic_config = load_config(basic_config_path, overrides={
        'metrics.db_name': tsdb_tmp_db_name,
        'metrics.port': port,
        'metrics.narrow_format': False,
        'metrics.table_name': 'metrics_wide',
    })
    table = create_metrics_writer(basic_config.metrics)
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
    table = create_evals_writer(basic_config.evals)
    yield table
    table.close()

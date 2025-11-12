import logging
from unittest.mock import Mock

import pytest
from _pytest.logging import LogCaptureFixture
from asyncua import Node
from asyncua.ua import VariantType
from lib_sql.engine import get_sql_engine
from lib_sql.inspection import column_exists, table_exists
from sqlalchemy import Engine

from coreio.communication.opc_communication import NodeData
from coreio.communication.sql_communication import SQL_Manager
from coreio.utils.config_schemas import DBConfigAdapter, InfraConfigAdapter

logger = logging.getLogger(__name__)

TABLE_NAME = "test_table"
DB_NAME = "test_db"

@pytest.fixture(scope="module")
def opc_sample_nodes():
    node = Mock(spec=Node)
    return {
        "ns=2;i=1": NodeData(node=node, name="sensor1", var_type=VariantType.Double),
        "ns=2;i=2": NodeData(node=node, name="sensor2", var_type=VariantType.Int32),
    }

@pytest.fixture(scope="function")
def sample_db_config(tsdb_engine: Engine):
    return InfraConfigAdapter(
        db = DBConfigAdapter(
            drivername = "postgresql+psycopg2",
            username = "postgres",
            password = "password",
            ip = "localhost",
            port = tsdb_engine.url.port or 0,
            schema = "public",
            db_name = DB_NAME, # needs to be the same db_name for all our tests
        ),
    )

def test_schema_creation(
    tsdb_engine: Engine,
    sample_db_config: InfraConfigAdapter,
    opc_sample_nodes: dict[str, NodeData],
    caplog: LogCaptureFixture,
):
    """CoreIO creates new database table from NodeData."""
    with caplog.at_level(logging.INFO):
        _ = SQL_Manager(sample_db_config, TABLE_NAME, opc_sample_nodes)

    # Using our own engine instead of `tsdb_engine`, because that changes the db_name every function
    engine = get_sql_engine(db_data=sample_db_config.db, db_name=sample_db_config.db.db_name)
    assert table_exists(engine, table_name=TABLE_NAME, schema=sample_db_config.db.schema)
    for node in opc_sample_nodes.values():
        assert column_exists(
            engine,
            table_name=TABLE_NAME,
            column_name=node.name.lower(),
            schema=sample_db_config.db.schema)

def test_existing_schema(
    tsdb_engine: Engine,
    sample_db_config: InfraConfigAdapter,
    opc_sample_nodes: dict[str, NodeData],
    caplog: LogCaptureFixture,
):
    """CoreIO detects existing table and columns."""

    with caplog.at_level(logging.INFO):
        _ = SQL_Manager(sample_db_config, TABLE_NAME, opc_sample_nodes)

    for node in opc_sample_nodes.values():
        assert f"{node.name} found" in caplog.text

def test_incompatible_types(
        tsdb_engine: Engine,
        sample_db_config: InfraConfigAdapter,
        caplog: LogCaptureFixture,
):
    """CoreIO raises an error when incompatible type is found in existing table"""

    node = Mock(spec=Node)
    opc_sample_nodes = {
        "ns=2;i=1": NodeData(node=node, name="sensor1", var_type=VariantType.Int32), # Changed to incompatible type
        "ns=2;i=2": NodeData(node=node, name="sensor2", var_type=VariantType.Int32),
    }

    with caplog.at_level(logging.INFO):
        with pytest.raises(TypeError):
            _ = SQL_Manager(sample_db_config, TABLE_NAME, opc_sample_nodes)

    print(caplog.text)

def test_add_column(
    tsdb_engine: Engine,
    sample_db_config: InfraConfigAdapter,
    caplog: LogCaptureFixture,
):
    """CoreIO adds a column to existing table when there is a new node."""

    node = Mock(spec=Node)
    opc_sample_nodes = {
        "ns=2;i=3": NodeData(node=node, name="sensor3", var_type=VariantType.Int32),
    }

    # Using our own engine instead of `tsdb_engine`, because that changes the db_name every function
    engine = get_sql_engine(db_data=sample_db_config.db, db_name=sample_db_config.db.db_name)
    with caplog.at_level(logging.INFO):
        _ = SQL_Manager(sample_db_config, TABLE_NAME, opc_sample_nodes)

    for col in ["sensor1", "sensor2", "sensor3"]:
        assert column_exists(
            engine,
            table_name=TABLE_NAME,
            column_name=col,
            schema=sample_db_config.db.schema)

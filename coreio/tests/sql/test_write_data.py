import logging
from datetime import datetime
from unittest.mock import Mock

import pytest
from asyncua import Node
from asyncua.ua import VariantType
from lib_sql.engine import get_sql_engine
from lib_utils.time import now_iso
from sqlalchemy import Engine, text

from coreio.communication.opc_communication import NodeData
from coreio.communication.sql_communication import SQL_Manager
from coreio.utils.config_schemas import DBConfigAdapter, InfraConfigAdapter

# Setup clock, wait, see writes
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

# write once
def test_write_once(
    tsdb_engine: Engine,
    sample_db_config: InfraConfigAdapter,
    opc_sample_nodes: dict[str, NodeData],
):
    sql_communication = SQL_Manager(sample_db_config, TABLE_NAME, opc_sample_nodes)
    time = now_iso()
    nodes_name_val = {"sensor1": 0.123, "sensor2": 42}
    sql_communication.write_to_sql(nodes_name_val, time)

    engine = get_sql_engine(db_data=sample_db_config.db, db_name=sample_db_config.db.db_name)
    query = f"SELECT * FROM {sample_db_config.db.schema}.{TABLE_NAME}"
    with engine.connect() as conn:
        result = conn.execute(text(query))
        result_rows = result.fetchall()

    assert len(result_rows) == 1
    row = result_rows[0]
    assert row[0] == datetime.fromisoformat(time)
    assert row[1] == nodes_name_val["sensor1"]
    assert row[2] == nodes_name_val["sensor2"]

def test_second_write(
    tsdb_engine: Engine,
    sample_db_config: InfraConfigAdapter,
    opc_sample_nodes: dict[str, NodeData],
):
    sql_communication = SQL_Manager(sample_db_config, TABLE_NAME, opc_sample_nodes)
    time = now_iso()
    nodes_name_val = {"sensor1": 0.123, "sensor2": 42}
    sql_communication.write_to_sql(nodes_name_val, time)

    engine = get_sql_engine(db_data=sample_db_config.db, db_name=sample_db_config.db.db_name)
    query = f"SELECT * FROM {sample_db_config.db.schema}.{TABLE_NAME}"
    with engine.connect() as conn:
        result = conn.execute(text(query))
        result_rows = result.fetchall()

    assert len(result_rows) == 2
    for row in result_rows:
        assert isinstance(row[0], datetime)
        assert row[1] == nodes_name_val["sensor1"]
        assert row[2] == nodes_name_val["sensor2"]

def test_partial_write(
    tsdb_engine: Engine,
    sample_db_config: InfraConfigAdapter,
    opc_sample_nodes: dict[str, NodeData],
):
    sql_communication = SQL_Manager(sample_db_config, TABLE_NAME, opc_sample_nodes)
    time = now_iso()
    nodes_name_val = {"sensor2": 42}
    sql_communication.write_to_sql(nodes_name_val, time)

    # Get most recent row
    engine = get_sql_engine(db_data=sample_db_config.db, db_name=sample_db_config.db.db_name)
    query = f"SELECT * FROM {sample_db_config.db.schema}.{TABLE_NAME} ORDER BY time DESC LIMIT 1"
    with engine.connect() as conn:
        result = conn.execute(text(query))
        result_rows = result.fetchall()

    assert len(result_rows) == 1
    for row in result_rows:
        assert isinstance(row[0], datetime)
        assert row[1] is None
        assert row[2] == nodes_name_val["sensor2"]

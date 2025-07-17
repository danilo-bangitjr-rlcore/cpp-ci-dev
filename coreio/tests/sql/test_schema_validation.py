import logging
import time
from collections.abc import Container
from unittest.mock import Mock

import docker
import pytest
from _pytest.logging import LogCaptureFixture
from asyncua import Node
from asyncua.ua import VariantType
from lib_utils.sql_logging.sql_logging import column_exists, get_sql_engine, table_exists
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from test.infrastructure.networking import get_free_port

from coreio.communication.opc_communication import NodeData
from coreio.communication.sql_communication import SQL_Manager
from coreio.utils.config_schemas import InfraConfigAdapter

logger = logging.getLogger(__name__)


TABLE_NAME = "test_table"

@pytest.fixture(scope="module")
def tsdb_port():
    """
    Gets a free port from localhost that the server can listen on
    instead of assuming any particular one will be free
    """
    return get_free_port('localhost')

@pytest.fixture(scope="module")
def timescaledb_container(tsdb_port: int):
    client = docker.from_env()

    container = client.containers.run(
        "timescale/timescaledb:2.19.1-pg17",
        environment={
            "POSTGRES_DB": "test_db",
            "POSTGRES_USER": "test_user",
            "POSTGRES_PASSWORD": "test_password",
        },
        ports={'5432/tcp': tsdb_port},
        tmpfs={'/var/lib/postgresql/data': ''},
        detach=True,
        remove=True,
        command="postgres -c shared_preload_libraries=timescaledb",
    )

    _wait_for_db_ready(tsdb_port)

    yield container

    container.stop()

def _wait_for_db_ready(tsdb_port: int, max_retries: int=30):
    conn_str = f"postgresql://test_user:test_password@localhost:{tsdb_port}/test_db"
    engine = create_engine(conn_str)

    for _ in range(max_retries):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return
        except OperationalError:
            time.sleep(1)

    raise TimeoutError(f"Database not ready after {max_retries} seconds")

@pytest.fixture(scope="module")
def opc_sample_nodes():
    node = Mock(spec=Node)
    return {
        "ns=2;i=1": NodeData(node=node, name="sensor1", var_type=VariantType.Double),
        "ns=2;i=2": NodeData(node=node, name="sensor2", var_type=VariantType.Int32),
    }

@pytest.fixture(scope="module")
def mock_infra_cfg(tsdb_port: int):
    cfg = Mock(spec=InfraConfigAdapter)
    cfg.db.drivername = "postgresql+psycopg2"
    cfg.db.username = "test_user"
    cfg.db.password = "test_password"
    cfg.db.ip = "localhost"
    cfg.db.port = tsdb_port
    cfg.db.schema = "public"
    cfg.db.db_name = "test_db"
    return cfg

def test_schema_creation(
        timescaledb_container: Container,
        mock_infra_cfg: Mock,
        opc_sample_nodes: dict[str, NodeData],
        caplog: LogCaptureFixture,
):
    """CoreIO creates new database table from NodeData."""

    with caplog.at_level(logging.INFO):
        _ = SQL_Manager(mock_infra_cfg, TABLE_NAME, opc_sample_nodes)

    # Verify
    engine = get_sql_engine(db_data=mock_infra_cfg.db, db_name=mock_infra_cfg.db.db_name)
    assert table_exists(engine, table_name=TABLE_NAME, schema=mock_infra_cfg.db.schema)
    for node in opc_sample_nodes.values():
        assert column_exists(
            engine,
            table_name=TABLE_NAME,
            column_name=node.name.lower(),
            schema=mock_infra_cfg.db.schema)


def test_existing_schema(
        timescaledb_container: Container,
        mock_infra_cfg: Mock,
        opc_sample_nodes: dict[str, NodeData],
        caplog: LogCaptureFixture,
):
    """CoreIO detects existing table and columns."""

    with caplog.at_level(logging.INFO):
        _ = SQL_Manager(mock_infra_cfg, TABLE_NAME, opc_sample_nodes)

    for node in opc_sample_nodes.values():
        assert f"{node.name} found" in caplog.text

def test_incompatible_types(
        timescaledb_container: Container,
        mock_infra_cfg: Mock,
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
            _ = SQL_Manager(mock_infra_cfg, TABLE_NAME, opc_sample_nodes)

    print(caplog.text)

def test_add_column(
        timescaledb_container: Container,
        mock_infra_cfg: Mock,
        caplog: LogCaptureFixture,
):
    """CoreIO adds a column to existing table when there is a new node."""

    node = Mock(spec=Node)
    opc_sample_nodes = {
        "ns=2;i=3": NodeData(node=node, name="sensor3", var_type=VariantType.Int32),
    }

    engine = get_sql_engine(db_data=mock_infra_cfg.db, db_name=mock_infra_cfg.db.db_name)
    with caplog.at_level(logging.INFO):
        _ = SQL_Manager(mock_infra_cfg, TABLE_NAME, opc_sample_nodes)

    for col in ["sensor1", "sensor2", "sensor3"]:
        assert column_exists(
            engine,
            table_name=TABLE_NAME,
            column_name=col,
            schema=mock_infra_cfg.db.schema)

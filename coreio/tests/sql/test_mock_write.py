from unittest.mock import MagicMock, Mock, patch

import pytest
from asyncua import Node
from asyncua.ua import VariantType
from sqlalchemy import text

from coreio.communication.opc_communication import NodeData
from coreio.communication.sql_communication import SQL_Manager
from coreio.utils.config_schemas import InfraConfigAdapter


@pytest.fixture
def mock_cfg():
    cfg = Mock(spec=InfraConfigAdapter)
    cfg.db.drivername = "postgresql+psycopg2"
    cfg.db.username = "test_user"
    cfg.db.password = "test_password"
    cfg.db.ip = "localhost"
    cfg.db.port = 5432
    cfg.db.schema = "public"
    cfg.db.db_name = "test_db"
    return cfg

@pytest.fixture(scope="module")
def opc_sample_nodes():
    node = Mock(spec=Node)
    return {
        "ns=2;i=1": NodeData(node=node, name="sensor1", var_type=VariantType.Double),
        "ns=2;i=2": NodeData(node=node, name="sensor2", var_type=VariantType.Int32),
    }

@pytest.fixture
def sql_manager(mock_cfg: Mock, opc_sample_nodes: dict[str, NodeData]):
    with (patch("coreio.communication.sql_communication.get_sql_engine", return_value=MagicMock()),
          patch.object(SQL_Manager, '_ensure_db_schema')):
        manager = SQL_Manager(
            cfg=mock_cfg,
            table_name="test_table",
            nodes_to_persist=dict(opc_sample_nodes),
        )
        manager.engine = Mock()
        return manager

class TestWriteNodes:

    def test_write_to_sql_no_engine(self, sql_manager: Mock):
        sql_manager.engine = None

        with patch('coreio.communication.sql_communication.logger') as mock_logger:
            sql_manager.write_to_sql({"sensor1": 1.0}, "2023-01-01T00:00:00Z")
            mock_logger.error.assert_called_once_with("SQL engine is not initialized")

    def test_write_to_sql_no_data(self, sql_manager: SQL_Manager):
        with patch('coreio.communication.sql_communication.logger') as mock_logger:
            sql_manager.write_to_sql({}, "2023-01-01T00:00:00Z")
            mock_logger.warning.assert_called_once_with("No data provided to write_to_sql")

    def test_write_to_sql_no_valid_columns(self, sql_manager: SQL_Manager):
        with patch('coreio.communication.sql_communication.logger') as mock_logger:
            sql_manager.write_to_sql({"invalid_node": 1.0}, "2023-01-01T00:00:00Z")
            mock_logger.warning.assert_called_once_with("No valid columns found in provided data")

    def test_write_to_sql_filters_and_inserts(self, sql_manager: SQL_Manager):
        connection = Mock()
        with patch('coreio.communication.sql_communication.TryConnectContextManager') as mock_context:
            mock_context.return_value.__enter__.return_value = connection
            sql_manager.write_to_sql({"sensor1": 1.0, "invalid": 2.0, "sensor2": 42}, "2023-01-01T00:00:00Z")

            assert connection.execute.called
            args, _ = connection.execute.call_args
            data = args[1]
            assert data["sensor1"] == 1.0
            assert data["sensor2"] == 42
            assert "invalid" not in data

    @patch('coreio.communication.sql_communication.now_iso')
    def test_write_to_sql_uses_default_timestamp(self, mock_now_iso: Mock, sql_manager: SQL_Manager):
        mock_now_iso.return_value = "2023-01-01T12:00:00Z"
        mock_connection = Mock()

        with patch('coreio.communication.sql_communication.TryConnectContextManager') as mock_context:
            mock_context.return_value.__enter__.return_value = mock_connection

            sql_manager.write_to_sql({"sensor1": 1.0}, None)

            args, _ = mock_connection.execute.call_args
            executed_data = args[1]
            assert executed_data["timestamp"] == "2023-01-01T12:00:00Z"

    def test_write_to_sql_uses_provided_timestamp(self, sql_manager: SQL_Manager):
        mock_connection = Mock()

        with patch('coreio.communication.sql_communication.TryConnectContextManager') as mock_context:
            mock_context.return_value.__enter__.return_value = mock_connection

            timestamp = "2023-06-15T14:30:00Z"
            sql_manager.write_to_sql({"sensor1": 1.0}, timestamp)

            args, _ = mock_connection.execute.call_args
            executed_data = args[1]
            assert executed_data["timestamp"] == timestamp

    def test_write_to_sql_case_insensitive_keys(self, sql_manager: SQL_Manager):
        mock_connection = Mock()

        with patch('coreio.communication.sql_communication.TryConnectContextManager') as mock_context:
            mock_context.return_value.__enter__.return_value = mock_connection

            data = {"Sensor1": 1.0, "SENSOR2": "hello"}
            sql_manager.write_to_sql(data, "2023-01-01T00:00:00Z")

            args, _ = mock_connection.execute.call_args
            executed_data = args[1]
            assert executed_data["sensor1"] == 1.0
            assert executed_data["sensor2"] == "hello"

    def test_write_to_sql_commits_transaction(self, sql_manager: SQL_Manager):
        mock_connection = Mock()

        with patch('coreio.communication.sql_communication.TryConnectContextManager') as mock_context:
            mock_context.return_value.__enter__.return_value = mock_connection

            sql_manager.write_to_sql({"sensor1": 1.0}, "2023-01-01T00:00:00Z")

            mock_connection.commit.assert_called_once()

    def test_write_to_sql_handles_database_exception(self, sql_manager: SQL_Manager):
        mock_connection = Mock()
        mock_connection.execute.side_effect = Exception("Database error")

        with patch('coreio.communication.sql_communication.TryConnectContextManager') as mock_context:
            mock_context.return_value.__enter__.return_value = mock_connection
            with patch('coreio.communication.sql_communication.logger') as mock_logger:

                sql_manager.write_to_sql({"sensor1": 1.0}, "2023-01-01T00:00:00Z")

                mock_logger.error.assert_called_once_with("Failed to write nodes: Database error")

    def test_insert_sql_generates_correct_query(self, sql_manager: SQL_Manager):
        expected_sql = text("""
            INSERT INTO public.test_table
            (time, sensor1, sensor2)
            VALUES (TIMESTAMP WITH TIME ZONE :timestamp, :sensor1, :sensor2);
        """)

        actual_sql = sql_manager._insert_sql([node.name for node in sql_manager.nodes_to_persist.values()])

        assert str(actual_sql).strip() == str(expected_sql).strip()

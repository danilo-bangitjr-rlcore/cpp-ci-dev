"""Unit tests for TelemetryManager class with mocked dependencies."""

from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
import yaml
from coretelemetry.services import DBConfig, SqlReader, TelemetryManager
from fastapi import HTTPException


@pytest.fixture
def mock_sql_reader() -> Mock:
    """Fully mocked SqlReader."""
    return Mock(spec=SqlReader)


@pytest.fixture
def manager(mock_sql_reader: Mock) -> TelemetryManager:
    """TelemetryManager with mocked SqlReader."""
    manager = TelemetryManager()
    manager.sql_reader = mock_sql_reader
    return manager


@pytest.fixture
def mock_yaml_config() -> dict:
    """Sample YAML config structure."""
    return {"metrics": {"table_name": "test_metrics_table"}}


class TestConfigManagement:
    """Tests for configuration getter/setter methods."""

    def test_get_set_db_config(self) -> None:
        """Test database configuration getter and setter."""
        manager = TelemetryManager()
        new_config = DBConfig(
            username="new_user",
            password="new_pass",
            ip="192.168.1.1",
            port=5433,
            db_name="new_db",
        )

        result = manager.set_db_config(new_config)

        assert result == new_config
        assert manager.get_db_config() == new_config

    def test_get_set_config_path(self) -> None:
        """Test config path getter and setter."""
        manager = TelemetryManager()
        new_path = Path("/tmp/configs")

        result = manager.set_config_path(new_path)

        assert result == new_path
        assert manager.get_config_path() == new_path

    def test_refresh_clears_cache(self, manager: TelemetryManager) -> None:
        """Test refresh clears sql_reader and cache."""
        manager.sql_reader = Mock()
        manager.metrics_table_cache = {"agent1": "table1"}

        manager.refresh()

        assert manager.sql_reader is None
        assert manager.metrics_table_cache == {}


class TestConnectionTesting:
    """Tests for database connection testing."""

    def test_test_db_connection_success(
        self, manager: TelemetryManager, mock_sql_reader: Mock,
    ) -> None:
        """Test database connection returns True on success."""
        mock_sql_reader.test_connection.return_value = True

        result = manager.test_db_connection()

        assert result is True
        mock_sql_reader.test_connection.assert_called_once()

    def test_test_db_connection_failure(
        self, manager: TelemetryManager, mock_sql_reader: Mock,
    ) -> None:
        """Test database connection returns False on failure."""
        mock_sql_reader.test_connection.return_value = False

        result = manager.test_db_connection()

        assert result is False


class TestGetMetricsTableName:
    """Tests for _get_metrics_table_name private method."""

    def test_get_metrics_table_name_cache_hit(self, manager: TelemetryManager) -> None:
        """Test cached table name is returned."""
        manager.metrics_table_cache = {"agent1": "cached_table"}

        result = manager._get_metrics_table_name("agent1")

        assert result == "cached_table"

    def test_get_metrics_table_name_yaml_parse(
        self, manager: TelemetryManager, mock_yaml_config: dict, tmp_path: Path,
    ) -> None:
        """Test YAML is parsed correctly and result is cached."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir

        yaml_content = yaml.dump(mock_yaml_config)

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("pathlib.Path.exists", return_value=True):
                result = manager._get_metrics_table_name("test_agent")

        assert result == "test_metrics_table"
        assert manager.metrics_table_cache["test_agent"] == "test_metrics_table"

    def test_get_metrics_table_name_file_not_found(
        self, manager: TelemetryManager, tmp_path: Path,
    ) -> None:
        """Test HTTPException 500 when YAML file doesn't exist."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir

        with pytest.raises(HTTPException) as exc_info:
            manager._get_metrics_table_name("nonexistent_agent")

        assert exc_info.value.status_code == 500
        assert "Configuration file not found" in exc_info.value.detail

    def test_get_metrics_table_name_yaml_parse_error(
        self, manager: TelemetryManager, tmp_path: Path,
    ) -> None:
        """Test HTTPException 500 when YAML parsing fails."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir

        with patch("builtins.open", mock_open(read_data="invalid: yaml: content:")):
            with patch("pathlib.Path.exists", return_value=True):
                with pytest.raises(HTTPException) as exc_info:
                    manager._get_metrics_table_name("test_agent")

        assert exc_info.value.status_code == 500
        assert "Failed to parse configuration file" in exc_info.value.detail

    def test_get_metrics_table_name_missing_key(
        self, manager: TelemetryManager, tmp_path: Path,
    ) -> None:
        """Test HTTPException 500 when table_name key is missing."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir

        yaml_content = yaml.dump({"other_key": "value"})

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("pathlib.Path.exists", return_value=True):
                with pytest.raises(HTTPException) as exc_info:
                    manager._get_metrics_table_name("test_agent")

        assert exc_info.value.status_code == 500
        assert "table_name' not found" in exc_info.value.detail


class TestGetTelemetryData:
    """Tests for get_telemetry_data method."""

    def test_get_telemetry_data_reserved_column(
        self, manager: TelemetryManager,
    ) -> None:
        """Test 'time' as metric raises HTTPException 400."""
        with pytest.raises(HTTPException) as exc_info:
            manager.get_telemetry_data("agent1", "time", None, None)

        assert exc_info.value.status_code == 400
        assert "reserved column" in exc_info.value.detail

    def test_get_telemetry_data_table_not_found(
        self, manager: TelemetryManager, mock_sql_reader: Mock, tmp_path: Path,
    ) -> None:
        """Test HTTPException 404 when table doesn't exist."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir
        manager.metrics_table_cache = {"agent1": "metrics_table"}
        mock_sql_reader.table_exists.return_value = False

        with pytest.raises(HTTPException) as exc_info:
            manager.get_telemetry_data("agent1", "temperature", None, None)

        assert exc_info.value.status_code == 404
        assert "Table" in exc_info.value.detail

    def test_get_telemetry_data_column_not_found(
        self, manager: TelemetryManager, mock_sql_reader: Mock, tmp_path: Path,
    ) -> None:
        """Test HTTPException 404 when column doesn't exist."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir
        manager.metrics_table_cache = {"agent1": "metrics_table"}
        mock_sql_reader.table_exists.return_value = True
        mock_sql_reader.column_exists.return_value = False

        with pytest.raises(HTTPException) as exc_info:
            manager.get_telemetry_data("agent1", "temperature", None, None)

        assert exc_info.value.status_code == 404
        assert "Column" in exc_info.value.detail

    def test_get_telemetry_data_connection_failure(
        self, manager: TelemetryManager, mock_sql_reader: Mock, tmp_path: Path,
    ) -> None:
        """Test HTTPException 503 on connection failure."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir
        manager.metrics_table_cache = {"agent1": "metrics_table"}
        mock_sql_reader.table_exists.return_value = True
        mock_sql_reader.column_exists.return_value = True
        mock_sql_reader.build_query.return_value = ("SELECT * FROM table", {})
        mock_sql_reader.execute_query.side_effect = Exception("Database error")

        with pytest.raises(HTTPException) as exc_info:
            manager.get_telemetry_data("agent1", "temperature", None, None)

        assert exc_info.value.status_code == 503

    def test_get_telemetry_data_success(
        self, manager: TelemetryManager, mock_sql_reader: Mock, tmp_path: Path,
    ) -> None:
        """Test successful telemetry data retrieval."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir
        manager.metrics_table_cache = {"agent1": "metrics_table"}
        mock_sql_reader.table_exists.return_value = True
        mock_sql_reader.column_exists.return_value = True
        mock_sql_reader.build_query.return_value = ("SELECT * FROM table", {})
        mock_sql_reader.execute_query.return_value = [
            ("2024-01-01", 25.5),
            ("2024-01-02", 26.0),
        ]

        result = manager.get_telemetry_data("agent1", "temperature", None, None)

        assert len(result) == 2
        assert result[0]["value"] == 25.5

    def test_get_telemetry_data_with_start_time(
        self, manager: TelemetryManager, mock_sql_reader: Mock, tmp_path: Path,
    ) -> None:
        """Test telemetry data with start_time parameter."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir
        manager.metrics_table_cache = {"agent1": "metrics_table"}
        mock_sql_reader.table_exists.return_value = True
        mock_sql_reader.column_exists.return_value = True
        mock_sql_reader.build_query.return_value = ("SELECT * FROM table", {})
        mock_sql_reader.execute_query.return_value = [
            ("2024-01-01", 25.5),
        ]

        result = manager.get_telemetry_data(
            "agent1", "temperature", "2024-01-01", None,
        )

        assert len(result) == 1
        mock_sql_reader.execute_query.assert_called_once()

    def test_get_telemetry_data_with_time_range(
        self, manager: TelemetryManager, mock_sql_reader: Mock, tmp_path: Path,
    ) -> None:
        """Test telemetry data with start_time and end_time."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir
        manager.metrics_table_cache = {"agent1": "metrics_table"}
        mock_sql_reader.table_exists.return_value = True
        mock_sql_reader.column_exists.return_value = True
        mock_sql_reader.build_query.return_value = ("SELECT * FROM table", {})
        mock_sql_reader.execute_query.return_value = [
            ("2024-01-01", 25.5),
            ("2024-01-02", 26.0),
        ]

        result = manager.get_telemetry_data(
            "agent1", "temperature", "2024-01-01", "2024-01-02",
        )

        assert len(result) == 2
        mock_sql_reader.execute_query.assert_called_once()

    def test_get_telemetry_data_empty_result(
        self, manager: TelemetryManager, mock_sql_reader: Mock, tmp_path: Path,
    ) -> None:
        """Test telemetry data raises 404 when no data found."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir
        manager.metrics_table_cache = {"agent1": "metrics_table"}
        mock_sql_reader.table_exists.return_value = True
        mock_sql_reader.column_exists.return_value = True
        mock_sql_reader.build_query.return_value = ("SELECT * FROM table", {})
        mock_sql_reader.execute_query.return_value = []

        with pytest.raises(HTTPException) as exc_info:
            manager.get_telemetry_data("agent1", "temperature", None, None)

        assert exc_info.value.status_code == 404


class TestGetAvailableMetrics:
    """Tests for get_available_metrics method."""

    def test_get_available_metrics_success(
        self, manager: TelemetryManager, mock_sql_reader: Mock, tmp_path: Path,
    ) -> None:
        """Test successful retrieval of available metrics."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir
        manager.metrics_table_cache = {"agent1": "metrics_table"}
        mock_sql_reader.table_exists.return_value = True
        mock_sql_reader.get_column_names.return_value = [
            "time",
            "temperature",
            "pressure",
        ]

        result = manager.get_available_metrics("agent1")

        assert result == {"agent_id": "agent1", "data": ["temperature", "pressure"]}
        assert "time" not in result["data"]

    def test_get_available_metrics_table_not_found(
        self, manager: TelemetryManager, mock_sql_reader: Mock, tmp_path: Path,
    ) -> None:
        """Test HTTPException 404 when table doesn't exist."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir
        manager.metrics_table_cache = {"agent1": "metrics_table"}
        mock_sql_reader.table_exists.return_value = False

        with pytest.raises(HTTPException) as exc_info:
            manager.get_available_metrics("agent1")

        assert exc_info.value.status_code == 404
        assert "Table" in exc_info.value.detail

    def test_get_available_metrics_connection_failure(
        self, manager: TelemetryManager, mock_sql_reader: Mock, tmp_path: Path,
    ) -> None:
        """Test HTTPException 503 on connection failure."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir
        manager.metrics_table_cache = {"agent1": "metrics_table"}
        mock_sql_reader.table_exists.return_value = True
        mock_sql_reader.get_column_names.side_effect = Exception("Database error")

        with pytest.raises(HTTPException) as exc_info:
            manager.get_available_metrics("agent1")

        assert exc_info.value.status_code == 503

    def test_get_available_metrics_only_time_column(
        self, manager: TelemetryManager, mock_sql_reader: Mock, tmp_path: Path,
    ) -> None:
        """Test raises 404 when only time column exists."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir
        manager.metrics_table_cache = {"agent1": "metrics_table"}
        mock_sql_reader.table_exists.return_value = True
        mock_sql_reader.get_column_names.return_value = ["time"]

        with pytest.raises(HTTPException) as exc_info:
            manager.get_available_metrics("agent1")

        assert exc_info.value.status_code == 404

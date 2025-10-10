"""Unit tests for TelemetryManager class with mocked dependencies."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from coretelemetry.agent_metrics_api.exceptions import (
    ColumnNotFoundError,
    ConfigFileNotFoundError,
    ConfigParseError,
    DatabaseConnectionError,
    NoDataFoundError,
    NoMetricsAvailableError,
    ReservedColumnError,
    TableNotFoundError,
)
from coretelemetry.agent_metrics_api.services import TelemetryManager
from coretelemetry.utils.sql import DBConfig, SqlReader


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

    def test_clear_cache_clears_cache(self, manager: TelemetryManager) -> None:
        """Test clear_cache clears sql_reader and cache."""
        manager.sql_reader = Mock()
        manager.metrics_table_cache = {"agent1": "table1"}

        manager.clear_cache()

        assert manager.sql_reader is None
        assert manager.metrics_table_cache == {}


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

        # Create actual YAML file
        yaml_file = config_dir / "test_agent.yaml"
        yaml_file.write_text(yaml.dump(mock_yaml_config))

        result = manager._get_metrics_table_name("test_agent")

        assert result == "test_metrics_table"
        assert manager.metrics_table_cache["test_agent"] == "test_metrics_table"

    def test_get_metrics_table_name_file_not_found(
        self, manager: TelemetryManager, tmp_path: Path,
    ) -> None:
        """Test ConfigFileNotFoundError when YAML file doesn't exist."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir

        with pytest.raises(ConfigFileNotFoundError):
            manager._get_metrics_table_name("nonexistent_agent")

    def test_get_metrics_table_name_yaml_parse_error(
        self, manager: TelemetryManager, tmp_path: Path,
    ) -> None:
        """Test ConfigParseError when YAML parsing fails."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir

        # Create actual YAML file with invalid syntax
        yaml_file = config_dir / "test_agent.yaml"
        yaml_file.write_text("invalid: yaml: content:")

        with pytest.raises(ConfigParseError):
            manager._get_metrics_table_name("test_agent")

    def test_get_metrics_table_name_missing_key(
        self, manager: TelemetryManager, tmp_path: Path,
    ) -> None:
        """Test ConfigParseError when table_name key is missing."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir

        # Create actual YAML file with wrong structure
        yaml_file = config_dir / "test_agent.yaml"
        yaml_file.write_text(yaml.dump({"other_key": "value"}))

        with pytest.raises(ConfigParseError):
            manager._get_metrics_table_name("test_agent")


class TestGetTelemetryData:
    """Tests for get_telemetry_data method."""

    def test_get_telemetry_data_reserved_column(
        self, manager: TelemetryManager,
    ) -> None:
        """Test 'time' as metric raises ReservedColumnError."""
        with pytest.raises(ReservedColumnError):
            manager.get_telemetry_data("agent1", "time", None, None)

    def test_get_telemetry_data_table_not_found(
        self, manager: TelemetryManager, mock_sql_reader: Mock, tmp_path: Path,
    ) -> None:
        """Test TableNotFoundError when table doesn't exist."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir
        manager.metrics_table_cache = {"agent1": "metrics_table"}
        mock_sql_reader.table_exists.return_value = False

        with pytest.raises(TableNotFoundError):
            manager.get_telemetry_data("agent1", "temperature", None, None)

    def test_get_telemetry_data_column_not_found(
        self, manager: TelemetryManager, mock_sql_reader: Mock, tmp_path: Path,
    ) -> None:
        """Test ColumnNotFoundError when column doesn't exist."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir
        manager.metrics_table_cache = {"agent1": "metrics_table"}
        mock_sql_reader.table_exists.return_value = True
        mock_sql_reader.column_exists.return_value = False

        with pytest.raises(ColumnNotFoundError):
            manager.get_telemetry_data("agent1", "temperature", None, None)

    def test_get_telemetry_data_connection_failure(
        self, manager: TelemetryManager, mock_sql_reader: Mock, tmp_path: Path,
    ) -> None:
        """Test DatabaseConnectionError on connection failure."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir
        manager.metrics_table_cache = {"agent1": "metrics_table"}
        mock_sql_reader.table_exists.return_value = True
        mock_sql_reader.column_exists.return_value = True
        mock_sql_reader.build_query.return_value = ("SELECT * FROM table", {})
        mock_sql_reader.execute_query.side_effect = Exception("Database error")

        with pytest.raises(DatabaseConnectionError):
            manager.get_telemetry_data("agent1", "temperature", None, None)

    def test_get_telemetry_data_empty_result(
        self, manager: TelemetryManager, mock_sql_reader: Mock, tmp_path: Path,
    ) -> None:
        """Test NoDataFoundError when no data found."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir
        manager.metrics_table_cache = {"agent1": "metrics_table"}
        mock_sql_reader.table_exists.return_value = True
        mock_sql_reader.column_exists.return_value = True
        mock_sql_reader.build_query.return_value = ("SELECT * FROM table", {})
        mock_sql_reader.execute_query.return_value = []

        with pytest.raises(NoDataFoundError):
            manager.get_telemetry_data("agent1", "temperature", None, None)


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
        """Test TableNotFoundError when table doesn't exist."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir
        manager.metrics_table_cache = {"agent1": "metrics_table"}
        mock_sql_reader.table_exists.return_value = False

        with pytest.raises(TableNotFoundError):
            manager.get_available_metrics("agent1")

    def test_get_available_metrics_connection_failure(
        self, manager: TelemetryManager, mock_sql_reader: Mock, tmp_path: Path,
    ) -> None:
        """Test DatabaseConnectionError on connection failure."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir
        manager.metrics_table_cache = {"agent1": "metrics_table"}
        mock_sql_reader.table_exists.return_value = True
        mock_sql_reader.get_column_names.side_effect = Exception("Database error")

        with pytest.raises(DatabaseConnectionError):
            manager.get_available_metrics("agent1")

    def test_get_available_metrics_only_time_column(
        self, manager: TelemetryManager, mock_sql_reader: Mock, tmp_path: Path,
    ) -> None:
        """Test NoMetricsAvailableError when only time column exists."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        manager.config_path = config_dir
        manager.metrics_table_cache = {"agent1": "metrics_table"}
        mock_sql_reader.table_exists.return_value = True
        mock_sql_reader.get_column_names.return_value = ["time"]

        with pytest.raises(NoMetricsAvailableError):
            manager.get_available_metrics("agent1")

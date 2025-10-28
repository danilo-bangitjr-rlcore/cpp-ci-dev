"""Integration tests for AgentMetricsManager with real PostgreSQL database."""

from pathlib import Path

import pytest
import yaml
from coretelemetry.agent_metrics_api.exceptions import (
    ColumnNotFoundError,
    NoDataFoundError,
    ReservedColumnError,
    TableNotFoundError,
)
from coretelemetry.agent_metrics_api.services import AgentMetricsManager
from coretelemetry.utils.sql import DBConfig

pytest_plugins = [
    "test.infrastructure.networking",
    "test.infrastructure.utils.tsdb",
]


@pytest.fixture
def manager_real(db_config_from_engine: DBConfig, sample_config_dir: Path) -> AgentMetricsManager:
    """Real AgentMetricsManager with real database and config."""
    manager = AgentMetricsManager()
    manager.set_db_config(db_config_from_engine)
    manager.set_config_path(sample_config_dir)
    return manager


class TestGetTelemetryDataIntegration:
    """Integration tests for get_telemetry_data with real database."""

    @pytest.mark.timeout(30)
    def test_get_telemetry_data_latest_value(
        self, manager_real: AgentMetricsManager, sample_metrics_table: tuple[str, str],
    ):
        """Test retrieving latest value without time parameters."""
        _schema_name, _table_name = sample_metrics_table

        result = manager_real.get_telemetry_data("test_agent", "temperature", None, None)

        # Should return only 1 row (latest non-NULL value)
        assert len(result) == 1
        assert "timestamp" in result[0]
        assert "value" in result[0]
        assert result[0]["value"] == 27.5  # Latest non-NULL temperature

    @pytest.mark.timeout(30)
    def test_get_telemetry_data_time_range(
        self, manager_real: AgentMetricsManager, sample_metrics_table: tuple[str, str],
    ):
        """Test retrieving data with start and end time."""
        _schema_name, _table_name = sample_metrics_table

        result = manager_real.get_telemetry_data(
            "test_agent", "temperature", "2024-01-01 10:00:00+00", "2024-01-01 11:30:00+00",
        )

        # Should return 2 rows
        assert len(result) == 2
        assert result[0]["value"] == 26.0  # Descending order
        assert result[1]["value"] == 25.5

    @pytest.mark.timeout(30)
    def test_get_telemetry_data_filters_nulls(
        self, manager_real: AgentMetricsManager, sample_metrics_table: tuple[str, str],
    ):
        """Test that NULL values are filtered out."""
        _schema_name, _table_name = sample_metrics_table

        result = manager_real.get_telemetry_data(
            "test_agent", "temperature", "2024-01-01 10:00:00+00", "2024-01-01 14:00:00+00",
        )

        # Should return 3 rows (excludes 13:00 with NULL temperature)
        assert len(result) == 3
        # Verify no NULL values
        for row in result:
            assert row["value"] is not None

    @pytest.mark.timeout(30)
    def test_get_telemetry_data_transforms_correctly(
        self, manager_real: AgentMetricsManager, sample_metrics_table: tuple[str, str],
    ):
        """Test data transformation to dict structure."""
        _schema_name, _table_name = sample_metrics_table

        result = manager_real.get_telemetry_data("test_agent", "pressure", None, None)

        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "timestamp" in result[0]
        assert "value" in result[0]
        assert isinstance(result[0]["value"], float)

    @pytest.mark.timeout(30)
    def test_get_telemetry_data_time_reserved_word(self, manager_real: AgentMetricsManager):
        """Test 'time' as metric raises ReservedColumnError."""
        with pytest.raises(ReservedColumnError):
            manager_real.get_telemetry_data("test_agent", "time", None, None)

    @pytest.mark.timeout(30)
    def test_get_telemetry_data_table_not_found(
        self, manager_real: AgentMetricsManager, sample_config_dir: Path,
    ):
        """Test TableNotFoundError when table doesn't exist."""
        # Create config with nonexistent table
        bad_config = {"metrics": {"table_name": "nonexistent_table_xyz"}}
        config_file = sample_config_dir / "bad_agent.yaml"
        with open(config_file, "w") as f:
            yaml.dump(bad_config, f)

        with pytest.raises(TableNotFoundError):
            manager_real.get_telemetry_data("bad_agent", "temperature", None, None)

    @pytest.mark.timeout(30)
    def test_get_telemetry_data_column_not_found(
        self, manager_real: AgentMetricsManager, sample_metrics_table: tuple[str, str],
    ):
        """Test ColumnNotFoundError when column doesn't exist."""
        with pytest.raises(ColumnNotFoundError):
            manager_real.get_telemetry_data("test_agent", "nonexistent_column", None, None)

    @pytest.mark.timeout(30)
    def test_get_telemetry_data_empty_result(
        self, manager_real: AgentMetricsManager, sample_metrics_table: tuple[str, str],
    ):
        """Test NoDataFoundError when query returns no data."""
        with pytest.raises(NoDataFoundError):
            # Query for a future time range with no data
            manager_real.get_telemetry_data("test_agent", "temperature", "2025-01-01", "2025-01-02")


class TestGetAvailableMetricsIntegration:
    """Integration tests for get_available_metrics with real database."""

    @pytest.mark.timeout(30)
    def test_get_available_metrics_returns_columns(
        self, manager_real: AgentMetricsManager, sample_metrics_table: tuple[str, str],
    ):
        """Test retrieving available metrics excludes 'time'."""
        _schema_name, _table_name = sample_metrics_table

        result = manager_real.get_available_metrics("test_agent")

        assert result["agent_id"] == "test_agent"
        assert "time" not in result["data"]
        assert "temperature" in result["data"]
        assert "pressure" in result["data"]
        assert len(result["data"]) == 2


class TestCaching:
    """Integration tests for YAML config caching."""

    @pytest.mark.timeout(30)
    def test_yaml_cache_works(
        self, manager_real: AgentMetricsManager, sample_metrics_table: tuple[str, str],
    ):
        """Test that second call uses cached table name."""
        _schema_name, table_name = sample_metrics_table

        # Verify cache starts empty
        assert "test_agent" not in manager_real.metrics_table_cache

        # First call
        manager_real.get_telemetry_data("test_agent", "temperature", None, None)

        # Verify cache is populated
        assert "test_agent" in manager_real.metrics_table_cache
        assert table_name == manager_real.metrics_table_cache["test_agent"]

        # Second call should use cache
        result = manager_real.get_telemetry_data("test_agent", "pressure", None, None)

        assert len(result) >= 1

    @pytest.mark.timeout(30)
    def test_clear_cache_clears_yaml_cache(
        self, manager_real: AgentMetricsManager, sample_metrics_table: tuple[str, str],
    ):
        """Test that clear_cache clears the YAML config cache."""
        _schema_name, _table_name = sample_metrics_table

        # Populate cache
        manager_real.get_telemetry_data("test_agent", "temperature", None, None)
        assert "test_agent" in manager_real.metrics_table_cache

        # Clear cache
        manager_real.clear_cache()

        # Cache should be cleared
        assert manager_real.metrics_table_cache == {}

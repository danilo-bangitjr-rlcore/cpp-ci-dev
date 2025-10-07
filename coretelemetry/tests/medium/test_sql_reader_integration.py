"""Integration tests for SqlReader with real PostgreSQL database."""

import pytest
from coretelemetry.services import DBConfig, SqlReader
from sqlalchemy import Engine

pytest_plugins = [
    "test.infrastructure.networking",
    "test.infrastructure.utils.tsdb",
]


@pytest.fixture
def sql_reader_real(db_config_from_engine: DBConfig) -> SqlReader:
    """Real SqlReader with actual database connection."""
    return SqlReader(db_config_from_engine)


class TestTableExistsIntegration:
    """Integration tests for table_exists with real database."""

    @pytest.mark.timeout(10)
    def test_table_exists_real_table(self, sql_reader_real: SqlReader, sample_metrics_table: tuple[str, str]):
        """Test table_exists returns True for real table."""
        _schema_name, table_name = sample_metrics_table

        result = sql_reader_real.table_exists(table_name)

        assert result is True

    @pytest.mark.timeout(10)
    def test_table_exists_nonexistent(self, sql_reader_real: SqlReader):
        """Test table_exists returns False for nonexistent table."""
        result = sql_reader_real.table_exists("nonexistent_table_xyz")

        assert result is False


class TestColumnExistsIntegration:
    """Integration tests for column_exists with real database."""

    @pytest.mark.timeout(10)
    def test_column_exists_real_column(self, sql_reader_real: SqlReader, sample_metrics_table: tuple[str, str]):
        """Test column_exists returns True for real column."""
        _schema_name, table_name = sample_metrics_table

        result = sql_reader_real.column_exists(table_name, "temperature")

        assert result is True

    @pytest.mark.timeout(10)
    def test_column_exists_nonexistent(self, sql_reader_real: SqlReader, sample_metrics_table: tuple[str, str]):
        """Test column_exists returns False for nonexistent column."""
        _schema_name, table_name = sample_metrics_table

        result = sql_reader_real.column_exists(table_name, "nonexistent_column_xyz")

        assert result is False


class TestExecuteQueryIntegration:
    """Integration tests for execute_query with real database."""

    @pytest.mark.timeout(10)
    def test_execute_query_all_rows(self, sql_reader_real: SqlReader, sample_metrics_table: tuple[str, str]):
        """Test query returns all rows without filters."""
        _schema_name, table_name = sample_metrics_table

        query, params = sql_reader_real.build_query(
            table_name=table_name,
            column_name="temperature",
            start_time=None,
            end_time=None,
            time_col=True,
            not_null=False,
        )

        result = sql_reader_real.execute_query(query, params)

        # Should return only 1 row due to LIMIT 1 (latest value)
        assert len(result) == 1

    @pytest.mark.timeout(10)
    def test_execute_query_with_start_time(self, sql_reader_real: SqlReader, sample_metrics_table: tuple[str, str]):
        """Test query with start_time filter."""
        _schema_name, table_name = sample_metrics_table

        query, params = sql_reader_real.build_query(
            table_name=table_name,
            column_name="temperature",
            start_time="2024-01-01 11:00:00+00",
            end_time=None,
            time_col=True,
            not_null=False,
        )

        result = sql_reader_real.execute_query(query, params)

        # Should return 3 rows (11:00, 12:00, 13:00 but 13:00 has NULL temp)
        assert len(result) >= 2

    @pytest.mark.timeout(10)
    def test_execute_query_with_time_range(self, sql_reader_real: SqlReader, sample_metrics_table: tuple[str, str]):
        """Test query with start_time and end_time range."""
        _schema_name, table_name = sample_metrics_table

        query, params = sql_reader_real.build_query(
            table_name=table_name,
            column_name="temperature",
            start_time="2024-01-01 10:00:00+00",
            end_time="2024-01-01 11:30:00+00",
            time_col=True,
            not_null=False,
        )

        result = sql_reader_real.execute_query(query, params)

        # Should return 2 rows (10:00 and 11:00)
        assert len(result) == 2

    @pytest.mark.timeout(10)
    def test_execute_query_not_null_filter(self, sql_reader_real: SqlReader, sample_metrics_table: tuple[str, str]):
        """Test query excludes NULL values with not_null=True."""
        _schema_name, table_name = sample_metrics_table

        query, params = sql_reader_real.build_query(
            table_name=table_name,
            column_name="temperature",
            start_time="2024-01-01 10:00:00+00",
            end_time="2024-01-01 14:00:00+00",
            time_col=True,
            not_null=True,
        )

        result = sql_reader_real.execute_query(query, params)

        # Should return 3 rows (excludes 13:00 with NULL temperature)
        assert len(result) == 3
        # Verify no NULL values
        for row in result:
            assert row[1] is not None


class TestGetColumnNamesIntegration:
    """Integration tests for get_column_names with real database."""

    @pytest.mark.timeout(10)
    def test_get_column_names_real_table(self, sql_reader_real: SqlReader, sample_metrics_table: tuple[str, str]):
        """Test get_column_names retrieves actual columns."""
        _schema_name, table_name = sample_metrics_table

        result = sql_reader_real.get_column_names(table_name)

        assert "time" in result
        assert "temperature" in result
        assert "pressure" in result
        assert len(result) == 3


class TestConnectionIntegration:
    """Integration tests for test_connection with real database."""

    @pytest.mark.timeout(10)
    def test_test_connection_success(self, sql_reader_real: SqlReader, tsdb_engine: Engine):
        """Test connection to real database succeeds."""
        result = sql_reader_real.test_connection()

        assert result is True

from datetime import datetime
from pathlib import Path

import yaml
from coretelemetry.agent_metrics_api.exceptions import (
    ColumnNotFoundError,
    ConfigFileNotFoundError,
    ConfigParseError,
    DatabaseConnectionError,
    NoDataFoundError,
    NoMetricsAvailableError,
    ReservedColumnError,
    ResultTooLargeError,
    TableNotFoundError,
)
from coretelemetry.utils.sql import DBConfig, SqlReader

MAX_RESULT_ROWS = 5000


class AgentMetricsManager:

    def __init__(self):
        self.db_config = DBConfig()
        self.config_path = Path("clean/")
        self.metrics_table_cache: dict[str, str] = {}
        self.sql_reader: SqlReader | None = None

    # Private helper methods
    def _get_metrics_table_name(self, agent_id: str) -> str:
        if agent_id in self.metrics_table_cache:
            return self.metrics_table_cache[agent_id]

        yaml_file_path = self.config_path / f"{agent_id}.yaml"

        if not yaml_file_path.exists():
            raise ConfigFileNotFoundError(
                f"Configuration file not found for agent '{agent_id}': {yaml_file_path}",
                agent_id=agent_id,
                config_path=str(yaml_file_path),
            )

        try:
            with open(yaml_file_path) as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            raise ConfigParseError(
                f"Failed to parse configuration file for agent '{agent_id}': {e!s}",
                agent_id=agent_id,
                config_path=str(yaml_file_path),
            ) from e

        table_name = (config_data or {}).get('metrics', {}).get('table_name')

        if not table_name:
            raise ConfigParseError(
                f"'metrics.table_name' not found in configuration for agent '{agent_id}'",
                agent_id=agent_id,
                config_path=str(yaml_file_path),
            )

        self.metrics_table_cache[agent_id] = table_name

        return table_name


    # Configuration methods
    def get_db_config(self) -> DBConfig:
        return self.db_config

    def set_db_config(self, config: DBConfig) -> DBConfig:
        self.db_config = config
        return self.db_config

    def get_config_path(self) -> Path:
        return self.config_path

    def set_config_path(self, path: Path) -> Path:
        self.config_path = path
        return self.config_path

    def clear_cache(self):
        """Clear all cached data including SQL reader and metrics table cache."""
        self.sql_reader = None
        self.metrics_table_cache = {}

    def test_db_connection(self) -> bool:
        if self.sql_reader is None:
            self.sql_reader = SqlReader(self.db_config)
        return self.sql_reader.test_connection()

    # Metrics methods
    def get_telemetry_data(
        self,
        agent_id: str,
        metric: str,
        start_time: str | None,
        end_time: str | None,
    ):
        if self.sql_reader is None:
            self.sql_reader = SqlReader(self.db_config)

        if metric.lower() == "time":
            raise ReservedColumnError(
                "'time' is a reserved column and cannot be used as a metric",
                metric=metric,
            )

        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()

        table_name = self._get_metrics_table_name(agent_id)

        if not self.sql_reader.table_exists(table_name):
            raise TableNotFoundError(
                f"Table '{table_name}' not found in database",
                table_name=table_name,
                agent_id=agent_id,
            )

        if not self.sql_reader.column_exists(table_name, metric):
            raise ColumnNotFoundError(
                f"Column '{metric}' not found in table '{table_name}'",
                column=metric,
                table_name=table_name,
                agent_id=agent_id,
            )

        query, params = self.sql_reader.build_query(
            table_name, metric, start_time, end_time, time_col=True, not_null=True,
        )

        try:
            raw_data = self.sql_reader.execute_query(query, params)
        except Exception as e:
            raise DatabaseConnectionError(
                f"Database connection failed: {e!s}",
                agent_id=agent_id,
            ) from e

        if not raw_data:
            time_range = f" from {start_time} to {end_time}" if start_time or end_time else ""
            raise NoDataFoundError(
                f"No data found in table '{table_name}' for column '{metric}'{time_range}",
                table_name=table_name,
                metric=metric,
                agent_id=agent_id,
            )

        if len(raw_data) > MAX_RESULT_ROWS:
            raise ResultTooLargeError(
                f"Result exceeded maximum length of {MAX_RESULT_ROWS} rows",
                max_rows=MAX_RESULT_ROWS,
                actual_rows=len(raw_data),
            )

        return [{"timestamp": row[0], "value": float(row[1])} for row in raw_data]

    def get_available_metrics(self, agent_id: str) -> dict:
        if self.sql_reader is None:
            self.sql_reader = SqlReader(self.db_config)

        table_name = self._get_metrics_table_name(agent_id)

        if not self.sql_reader.table_exists(table_name):
            raise TableNotFoundError(
                f"Table '{table_name}' not found in database",
                table_name=table_name,
                agent_id=agent_id,
            )

        try:
            all_columns = self.sql_reader.get_column_names(table_name)
        except Exception as e:
            raise DatabaseConnectionError(
                f"Failed to retrieve columns: {e!s}",
                agent_id=agent_id,
                table_name=table_name,
            ) from e

        metrics = [col for col in all_columns if col.lower() != "time"]

        if not metrics:
            raise NoMetricsAvailableError(
                f"No metrics available for agent '{agent_id}'",
                agent_id=agent_id,
                table_name=table_name,
            )

        return {"agent_id": agent_id, "data": metrics}


# Create singleton instance
agent_metrics_manager = AgentMetricsManager()


def get_agent_metrics_manager() -> AgentMetricsManager:
    return agent_metrics_manager

from datetime import datetime
from pathlib import Path

import yaml
from coretelemetry.utils.sql import DBConfig, SqlReader
from fastapi import HTTPException

MAX_RESULT_ROWS = 5000


class TelemetryManager:

    def __init__(self):
        self.db_config = DBConfig()
        self.config_path = Path("clean/")
        self.metrics_table_cache: dict[str, str] = {}
        self.sql_reader: SqlReader | None = None

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

    # Private helper methods
    def _get_metrics_table_name(self, agent_id: str) -> str:
        if agent_id in self.metrics_table_cache:
            return self.metrics_table_cache[agent_id]

        yaml_file_path = self.config_path / f"{agent_id}.yaml"

        if not yaml_file_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Configuration file not found for agent '{agent_id}': {yaml_file_path}",
            )

        try:
            with open(yaml_file_path) as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse configuration file for agent '{agent_id}': {e!s}",
            ) from e

        table_name = (config_data or {}).get('metrics', {}).get('table_name')

        if not table_name:
            raise HTTPException(
                status_code=500,
                detail=f"'metrics.table_name' not found in configuration for agent '{agent_id}'",
            )

        self.metrics_table_cache[agent_id] = table_name

        return table_name

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
            raise HTTPException(status_code=400, detail="'time' is a reserved column and cannot be used as a metric")

        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()

        table_name = self._get_metrics_table_name(agent_id)

        if not self.sql_reader.table_exists(table_name):
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found in database")

        if not self.sql_reader.column_exists(table_name, metric):
            raise HTTPException(status_code=404, detail=f"Column '{metric}' not found in table '{table_name}'")

        query, params = self.sql_reader.build_query(
            table_name, metric, start_time, end_time, time_col=True, not_null=True,
        )

        try:
            raw_data = self.sql_reader.execute_query(query, params)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Database connection failed: {e!s}") from e

        if not raw_data:
            time_range = f" from {start_time} to {end_time}" if start_time or end_time else ""
            raise HTTPException(
                status_code=404,
                detail=f"No data found in table '{table_name}' for column '{metric}'{time_range}",
            )

        if len(raw_data) > MAX_RESULT_ROWS:
            raise HTTPException(status_code=413, detail=f"Result exceeded maximum length of {MAX_RESULT_ROWS} rows")

        return [{"timestamp": row[0], "value": float(row[1])} for row in raw_data]

    def get_available_metrics(self, agent_id: str) -> dict:
        if self.sql_reader is None:
            self.sql_reader = SqlReader(self.db_config)

        table_name = self._get_metrics_table_name(agent_id)

        if not self.sql_reader.table_exists(table_name):
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found in database")

        try:
            all_columns = self.sql_reader.get_column_names(table_name)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to retrieve columns: {e!s}") from e

        metrics = [col for col in all_columns if col.lower() != "time"]

        if not metrics:
            raise HTTPException(status_code=404, detail=f"No metrics available for agent '{agent_id}'")

        return {"agent_id": agent_id, "data": metrics}


# Create singleton instance
telemetry_manager = TelemetryManager()


def get_telemetry_manager() -> TelemetryManager:
    return telemetry_manager

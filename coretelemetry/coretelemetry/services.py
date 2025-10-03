from pathlib import Path

import yaml
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
class DBConfig:
    drivername: str = 'postgresql+psycopg2'
    username: str = 'postgres'
    password: str = 'password'
    ip: str = 'localhost'
    port: int = 5432
    db_name: str = 'postgres'
    schema: str = 'public'


class TelemetryManager:

    def __init__(self):
        self.db_config = DBConfig()
        self.config_path = Path("clean/")
        self.metrics_table_cache: dict[str, str] = {}

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

    # Private helper methods
    def _get_metrics_table_name(self, agent_id: str) -> str:
        if agent_id in self.metrics_table_cache:
            return self.metrics_table_cache[agent_id]

        yaml_file_path = self.config_path / f"{agent_id}.yaml"

        if not yaml_file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_file_path}")

        with open(yaml_file_path) as f:
            config_data = yaml.safe_load(f)

        table_name = (config_data or {}).get('metrics', {}).get('table_name')

        if not table_name:
            raise KeyError(f"'metrics.table_name' not found in {yaml_file_path}")

        self.metrics_table_cache[agent_id] = table_name

        return table_name

    # Telemetry data methods (to be implemented)
    async def get_telemetry_data(
        self,
        agent_id: str,
        metric: str,
        from_date: str,
        to: str,
    ) -> dict:
        # Placeholder implementation - will be replaced with actual SQL queries
        return {
            "agent_id": agent_id,
            "metric": metric,
            "from_date": from_date,
            "to": to,
            "data": [],
        }

    async def get_available_metrics(self, agent_id: str) -> list[str]:
        # Placeholder implementation
        return []


# Create singleton instance
telemetry_manager = TelemetryManager()


def get_telemetry_manager() -> TelemetryManager:
    return telemetry_manager

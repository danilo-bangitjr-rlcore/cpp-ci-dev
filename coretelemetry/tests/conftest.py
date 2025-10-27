from pathlib import Path

import pytest
import yaml
from coretelemetry.utils.sql import DBConfig
from sqlalchemy import Engine, text

# Enable pytest plugins for database testing
pytest_plugins = [
    "test.infrastructure.networking",
    "test.infrastructure.utils.tsdb",
]


@pytest.fixture
def sample_fixture():
    """Sample fixture for future use."""
    return {"status": "ok"}


@pytest.fixture
def sample_metrics_table(tsdb_engine: Engine):
    """Create a sample metrics table with test data for integration tests."""
    schema_name = "public"
    table_name = "test_metrics_wide"

    # Create table
    create_query = f"""
    CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (
        time TIMESTAMPTZ NOT NULL,
        temperature DOUBLE PRECISION,
        pressure DOUBLE PRECISION
    );
    """
    with tsdb_engine.connect() as conn:
        conn.execute(text(create_query))

        # Insert test data
        insert_query = f"""
        INSERT INTO {schema_name}.{table_name} (time, temperature, pressure) VALUES
        ('2024-01-01 10:00:00+00', 25.5, 101.3),
        ('2024-01-01 11:00:00+00', 26.0, 101.5),
        ('2024-01-01 12:00:00+00', 27.5, 102.0),
        ('2024-01-01 13:00:00+00', NULL, 103.0);
        """
        conn.execute(text(insert_query))
        conn.commit()

    yield (schema_name, table_name)


@pytest.fixture
def db_config_from_engine(tsdb_engine: Engine):
    """Create DBConfig from tsdb_engine for testing."""
    return DBConfig(
        drivername="postgresql+psycopg2",
        username="postgres",
        password="password",
        ip="localhost",
        port=tsdb_engine.url.port or 5432,
        db_name=tsdb_engine.url.database or "postgres",
        schema="public",
    )


@pytest.fixture
def sample_config_dir(tmp_path: Path, sample_metrics_table: tuple[str, str]):
    """Create temporary config directory with YAML files for testing."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    _schema_name, table_name = sample_metrics_table

    # Create agent YAML config
    agent_config = {"metrics": {"table_name": table_name.replace("_wide", "") }}

    config_file = config_dir / "test_agent.yaml"
    with open(config_file, "w") as f:
        yaml.dump(agent_config, f)

    return config_dir

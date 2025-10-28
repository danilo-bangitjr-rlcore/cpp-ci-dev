from copy import deepcopy
from typing import Any

import pytest
from lib_config.loader import direct_load_config
from sqlalchemy import Engine

from corerl.config import MainConfig
from corerl.configs.data_pipeline.db.data_writer import TagDBConfig


@pytest.fixture
def basic_config_path():
    return "tests/infrastructure/configs/basic_config.yaml"


def load_config(path: str, overrides: dict[str, Any]):
    cfg = direct_load_config(MainConfig, config_name=path, overrides=overrides)
    assert isinstance(cfg, MainConfig), f"Failed to load MainConfig at path {path}"
    return cfg


@pytest.fixture
def basic_config(
    basic_config_path: str,
):
    return load_config(basic_config_path, overrides={})


class ConfigBuilder:
    def __init__(self, base_config: MainConfig):
        self._base_config = base_config
        self._overrides: dict[str, Any] = {}

    def with_override(self, key: str, value: Any) -> "ConfigBuilder":
        self._overrides[key] = value
        return self

    def with_overrides(self, overrides: dict[str, Any]) -> "ConfigBuilder":
        self._overrides.update(overrides)
        return self

    def build(self) -> MainConfig:
        config = deepcopy(self._base_config)
        for key, value in self._overrides.items():
            self._set_nested_attr(config, key, value)

        return config

    def _set_nested_attr(self, obj: Any, key: str, value: Any) -> None:
        keys = key.split(".")
        current = obj

        # Navigate to the parent of the target attribute
        for k in keys[:-1]:
            if hasattr(current, k):
                current = getattr(current, k)
            else:
                raise AttributeError(f"Object has no attribute '{k}' in path '{key}'")

        # Set the final attribute
        final_key = keys[-1]
        if hasattr(current, final_key):
            setattr(current, final_key, value)
        else:
            raise AttributeError(f"Object has no attribute '{final_key}' in path '{key}'")


@pytest.fixture
def config_builder(basic_config: MainConfig) -> ConfigBuilder:
    return ConfigBuilder(basic_config)


def create_config_with_overrides(
    base_config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> MainConfig:
    base_config_path = base_config_path or "tests/infrastructure/configs/basic_config.yaml"
    overrides = overrides or {}

    # Load base config
    base_config = direct_load_config(MainConfig, config_name=base_config_path)
    assert isinstance(base_config, MainConfig), (
        f"Failed to load MainConfig from '{base_config_path}'. "
        f"Got {type(base_config).__name__} instead. "
        f"Check if the config file exists and is valid."
    )

    # Apply overrides using ConfigBuilder
    builder = ConfigBuilder(base_config)
    for key, value in overrides.items():
        builder.with_override(key, value)

    return builder.build()


@pytest.fixture()
def test_db_config(tsdb_engine: Engine, tsdb_tmp_db_name: str):
    port = tsdb_engine.url.port
    assert port is not None

    return TagDBConfig(
        drivername="postgresql+psycopg2",
        username="postgres",
        password="password",
        ip="localhost",
        port=port,
        db_name=tsdb_tmp_db_name,
        table_name="tags",
        table_schema="public",
    )

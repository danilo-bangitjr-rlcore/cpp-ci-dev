from typing import Any

import pytest
from lib_config.loader import direct_load_config
from sqlalchemy import Engine

from corerl.config import MainConfig
from corerl.data_pipeline.db.data_reader import TagDBConfig
from corerl.environment.async_env.async_env import AsyncEnvConfig
from corerl.eval.evals import EvalDBConfig
from corerl.eval.metrics import MetricsDBConfig


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


@pytest.fixture()
def test_db_config(tsdb_engine: Engine, tsdb_tmp_db_name: str) :
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
        table_schema='public',
    )

@pytest.fixture(scope="function")
def offline_cfg(test_db_config: TagDBConfig) -> MainConfig:
    cfg = direct_load_config(
        MainConfig,
        config_name='tests/medium/offline_training/assets/offline_config.yaml',
    )
    assert isinstance(cfg, MainConfig)

    if cfg.agent.critic.buffer.name == 'mixed_history_buffer':
        cfg.agent.critic.buffer.online_weight = 0.0

    if cfg.agent.policy.buffer.name == 'mixed_history_buffer':
        cfg.agent.policy.buffer.online_weight = 0.0

    assert isinstance(cfg.env, AsyncEnvConfig)
    cfg.env.db = test_db_config

    assert isinstance(cfg.metrics, MetricsDBConfig)
    cfg.metrics.port = test_db_config.port
    cfg.metrics.db_name = test_db_config.db_name

    assert isinstance(cfg.evals, EvalDBConfig)
    cfg.evals.port = test_db_config.port
    cfg.evals.db_name = test_db_config.db_name

    return cfg

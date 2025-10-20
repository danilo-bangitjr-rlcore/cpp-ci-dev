
import pytest
from corerl.configs.data_pipeline.db.data_writer import TagDBConfig
from lib_config.loader import direct_load_config
from sqlalchemy import Engine

from coreoffline.utils.config import OfflineMainConfig


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
        table_schema='public',
    )


@pytest.fixture(scope="function")
def offline_cfg(test_db_config: TagDBConfig):
    cfg = direct_load_config(
        OfflineMainConfig,
        config_name='tests/infrastructure/assets/offline_config.yaml',
    )


    assert isinstance(cfg, OfflineMainConfig)

    if cfg.agent.critic.buffer.name == 'mixed_history_buffer':
        cfg.agent.critic.buffer.online_weight = 0.0

    if cfg.agent.policy.buffer.name == 'mixed_history_buffer':
        cfg.agent.policy.buffer.online_weight = 0.0

    cfg.env.db = test_db_config

    cfg.metrics.port = test_db_config.port
    cfg.metrics.db_name = test_db_config.db_name

    cfg.evals.port = test_db_config.port
    cfg.evals.db_name = test_db_config.db_name

    return cfg

from typing import TYPE_CHECKING

from lib_config.config import MISSING, computed, config

from corerl.configs.sql_logging.sql_engine import SQLEngineConfig

if TYPE_CHECKING:
    from corerl.config import MainConfig


@config()
class EvalDBConfig(SQLEngineConfig):
    table_name: str = 'evals'
    enabled: bool = False
    table_schema: str = MISSING
    db_name: str = MISSING

    # Buffered writer config
    low_watermark: int = 1
    high_watermark: int = 256

    @computed('table_schema')
    @classmethod
    def _table_schema(cls, cfg: 'MainConfig'):
        return cfg.infra.db.schema

    @computed('db_name')
    @classmethod
    def _db_name(cls, cfg: 'MainConfig'):
        return cfg.infra.db.db_name

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

import pandas as pd
from lib_config.config import MISSING, computed, config

from corerl.sql_logging.sql_logging import SQLEngineConfig

if TYPE_CHECKING:
    from corerl.config import MainConfig

log = logging.getLogger(__name__)


class EvalsWriterProtocol(Protocol):
    def write(self, agent_step: int, evaluator: str, value: object, timestamp: str | None = None) -> None:
        ...

    def read(
        self,
        evaluator: str,
        step_start: int | None = None,
        step_end: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        ...

    def flush(self) -> None:
        ...

    def close(self) -> None:
        ...


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

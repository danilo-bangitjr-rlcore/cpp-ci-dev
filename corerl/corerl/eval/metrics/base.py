import logging
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, SupportsFloat

import pandas as pd
from lib_config.config import MISSING, computed, config

from corerl.sql_logging.sql_logging import SQLEngineConfig

if TYPE_CHECKING:
    from corerl.config import MainConfig

log = logging.getLogger(__name__)


class MetricsWriterProtocol(Protocol):
    def write(self, agent_step: int, metric: str, value: SupportsFloat, timestamp: str | None = None) -> None:
        ...

    def write_dict(
        self,
        values: Mapping[str, SupportsFloat | Mapping[str, Any]],
        agent_step: int,
        prefix: str = '',
    ) -> None:
        ...

    def read(
        self,
        metric: str,
        step_start: int | None = None,
        step_end: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        prefix_match: bool = False,
    ) -> pd.DataFrame:
        ...

    def flush(self) -> None:
        ...

    def close(self) -> None:
        ...


@config()
class MetricsDBConfig(SQLEngineConfig):
    table_name: str = 'metrics'
    enabled: bool = False
    narrow_format: bool = True
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
    def _dbname(cls, cfg: 'MainConfig'):
        return cfg.infra.db.db_name

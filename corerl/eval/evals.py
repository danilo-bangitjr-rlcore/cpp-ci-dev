import json
import logging
from typing import Literal, NamedTuple, Protocol

from sqlalchemy import Connection, Engine, text

from corerl.configs.config import config
from corerl.configs.group import Group
from corerl.utils.buffered_sql_writer import BufferedWriter, BufferedWriterConfig
from corerl.utils.time import now_iso

log = logging.getLogger(__name__)


class EvalWriterProtocol(Protocol):
    def write(self, agent_step: int, evaluator: str, value: object, timestamp: str | None = None):
        ...
    def close(self)-> None:
        ...


class _EvalPoint(NamedTuple):
    timestamp: str
    agent_step: int
    evaluator: str
    value: object # jsonb


@config()
class EvalDBConfig(BufferedWriterConfig):
    name: Literal['db'] = 'db'
    db_name: str = 'postgres'
    table_name: str = 'evals'
    table_schema: str = 'public'
    lo_wm: int = 10
    enabled: bool = False


class EvalWriter(BufferedWriter[_EvalPoint]):
    def __init__(
        self,
        cfg: EvalDBConfig,
        high_watermark: int = 256,
    ):
        super().__init__(cfg, cfg.lo_wm, high_watermark)
        self.cfg = cfg
        self._has_built = False

        self._eng: Engine | None = None
        self._conn: Connection | None = None


    def _create_table_sql(self):
        return text(f"""
            CREATE TABLE {self.cfg.table_schema}.{self.cfg.table_name} (
                time TIMESTAMP WITH time zone NOT NULL,
                agent_step INTEGER NOT NULL,
                evaluator text NOT NULL,
                value jsonb NOT NULL
            );
            SELECT create_hypertable('{self.cfg.table_name}', 'time', chunk_time_interval => INTERVAL '1d');
            CREATE INDEX evaluator_idx ON {self.cfg.table_name} (evaluator);
            ALTER TABLE {self.cfg.table_name} SET (
                timescaledb.compress,
                timescaledb.compress_segmentby='evaluator'
            );
        """)


    def _insert_sql(self):
        return text(f"""
            INSERT INTO {self.cfg.table_schema}.{self.cfg.table_name}
            (time, agent_step, evaluator, value)
            VALUES (TIMESTAMP :timestamp, :agent_step, :evaluator, :value)
        """)


    def write(self, agent_step: int, evaluator: str, value: object, timestamp: str | None = None):
        if not self.cfg.enabled:
            return

        point = _EvalPoint(
            timestamp=timestamp or now_iso(),
            agent_step=agent_step,
            evaluator=evaluator,
            value=json.dumps(value),
        )

        try:
            self._write(point)
        except Exception:
            log.exception(f'Failed to write evaluation output: {evaluator} {value}')


evals_group = Group[
    [], EvalWriterProtocol,
]()

evals_group.dispatcher(EvalWriter)

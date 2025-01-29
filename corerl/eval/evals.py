import json
import logging
import os
import shutil
from collections import defaultdict
from typing import Literal, NamedTuple, Protocol

import pandas as pd
from pydantic import Field
from sqlalchemy import Connection, Engine, text
from typing_extensions import Annotated

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


@config()
class PandasEvalsConfig:
    name: Literal['pandas'] = 'pandas'
    output_path: str = 'eval_outputs'
    buffer_size: int = 256  # Number of points to buffer before writing


class PandasEvalWriter():
    def __init__(
            self,
            cfg: PandasEvalsConfig
        ):
        self.buffer = defaultdict(list)  # Temporary buffer for points
        self.output_path = cfg.output_path
        self.buffer_size = cfg.buffer_size

        if os.path.exists(self.output_path):
            logging.warning("Output path for evaluation outputs already exists. "
                            "Existing files will be overwritten.")
            shutil.rmtree(self.output_path)

        os.makedirs(self.output_path, exist_ok=True)

    def write(self, agent_step: int, evaluator: str, value: object, timestamp: str | None = None):
        point = _EvalPoint(
            timestamp=timestamp or now_iso(),
            agent_step=agent_step,
            evaluator=evaluator,
            value=json.dumps(value),
        )

        self.buffer[evaluator].append(point)

        if len(self.buffer[evaluator]) >= self.buffer_size:
            self._flush_evaluator(evaluator)

    def _flush_evaluator(self, evaluator: str):
        """Write buffered points for a specific evaluator to CSV"""
        if not self.buffer[evaluator]:
            return

        data = defaultdict(list)
        for point in self.buffer[evaluator]:
            data['timestamp'].append(point.timestamp)
            data['agent_step'].append(point.agent_step)
            data['evaluator'].append(point.evaluator)
            data['value'].append(point.value)

        df = pd.DataFrame(data)
        file_path = f'{self.output_path}/{evaluator}.csv'

        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, index=False)

        self.buffer[evaluator].clear()

    def close(self):
        # Flush any remaining points
        for evaluator in list(self.buffer.keys()):
            self._flush_evaluator(evaluator)


EvaluatorsConfig = Annotated[
    EvalDBConfig | PandasEvalsConfig,
    Field(discriminator='name')
]


evals_group = Group[
    [], EvalWriterProtocol,
]()

evals_group.dispatcher(EvalWriter)
evals_group.dispatcher(PandasEvalWriter)

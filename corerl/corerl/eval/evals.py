import json
import logging
from datetime import UTC, datetime
from typing import NamedTuple

import pandas as pd
from lib_config.config import config
from lib_utils.sql_logging.connect_engine import TryConnectContextManager
from lib_utils.sql_logging.utils import SQLColumn, create_tsdb_table_query
from lib_utils.time import now_iso
from pydantic import Field
from sqlalchemy import text

from corerl.utils.buffered_sql_writer import BufferedWriter, BufferedWriterConfig, WatermarkSyncConfig

log = logging.getLogger(__name__)


class _EvalPoint(NamedTuple):
    timestamp: str
    agent_step: int
    evaluator: str
    value: object # jsonb


@config()
class EvalDBConfig(BufferedWriterConfig):
    table_name: str = 'evals'
    enabled: bool = False
    watermark_cfg: WatermarkSyncConfig = Field(
        default_factory=lambda: WatermarkSyncConfig('watermark', 1, 256),
    )


class EvalsTable(BufferedWriter[_EvalPoint]):
    def __init__(
        self,
        cfg: EvalDBConfig,
    ):
        super().__init__(cfg)
        self.cfg = cfg

    def _create_table_sql(self):
        return create_tsdb_table_query(
            schema=self.cfg.table_schema,
            table=self.cfg.table_name,
            columns=[
                SQLColumn(name='time', type='TIMESTAMP WITH TIME ZONE', nullable=False),
                SQLColumn(name='agent_step', type='INTEGER', nullable=False),
                SQLColumn(name='evaluator', type='TEXT', nullable=False),
                SQLColumn(name='value', type='jsonb', nullable=False),
            ],
            partition_column='evaluator',
            index_columns=['evaluator'],
        )

    def write(self, agent_step: int, evaluator: str, value: object, timestamp: str | None = None):
        if not self.cfg.enabled:
            return

        value = value if isinstance(value, str) else json.dumps(value)
        point = _EvalPoint(
            timestamp=timestamp or now_iso(),
            agent_step=agent_step,
            evaluator=evaluator,
            value=value,
        )

        try:
            self._write(point)
        except Exception:
            log.exception(f'Failed to write evaluation output: {evaluator} {value}')

    def _execute_read(self, stmt: str) -> pd.DataFrame:
        assert self.engine is not None
        with TryConnectContextManager(self.engine) as connection:
            return pd.read_sql(sql=text(stmt), con=connection)


    def _read_by_eval(self, evaluator: str) -> pd.DataFrame:
        stmt = f"""
            SELECT
                time,
                agent_step,
                value
            FROM {self.cfg.table_name}
            WHERE
                evaluator='{evaluator}';
        """

        df = self._execute_read(stmt)
        df["time"] = pd.to_datetime(df["time"])
        df["agent_step"] = df["agent_step"].astype(int)

        return df

    def _read_by_step(
        self,
        evaluator: str,
        step_start: int | None,
        step_end: int | None,
    ) -> pd.DataFrame:
        stmt = f"""
            SELECT
                agent_step,
                value
            FROM {self.cfg.table_name}
            WHERE
                evaluator='{evaluator}'
        """

        if step_start is not None:
            stmt += f" AND agent_step>='{step_start}'"

        if step_end is not None:
            stmt += f" AND agent_step<='{step_end}'"

        stmt += ";"

        df = self._execute_read(stmt)
        df["agent_step"] = df["agent_step"].astype(int)

        return df

    def _read_by_time(
        self,
        evaluator: str,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> pd.DataFrame:
        stmt = f"""
            SELECT
                time,
                value
            FROM {self.cfg.table_name}
            WHERE
                evaluator='{evaluator}'
        """

        if start_time is not None:
            if start_time.tzinfo is None:
                start_tz = start_time.astimezone().tzinfo
                log.warning(f"naive start_time passed, assuming {start_tz} and converting to UTC")
                start_time = start_time.replace(tzinfo=start_tz).astimezone(UTC)
            stmt += f" AND time>='{start_time.isoformat()}'"

        if end_time is not None:
            if end_time.tzinfo is None:
                end_tz = end_time.astimezone().tzinfo
                log.warning(f"naive end_time passed, assuming {end_tz} and converting to UTC")
                end_time = end_time.replace(tzinfo=end_tz).astimezone(UTC)
            stmt += f" AND time<='{end_time.isoformat()}'"

        stmt += ";"

        df = self._execute_read(stmt)
        df["time"] = pd.to_datetime(df["time"])

        return df

    def read(
        self,
        evaluator: str,
        step_start: int | None = None,
        step_end: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        # Make sure all EvalPoint objects in buffer have been written to DB
        self.blocking_sync()

        if start_time is not None or end_time is not None:
            return self._read_by_time(evaluator, start_time, end_time)
        if step_start is not None or step_end is not None:
            return self._read_by_step(evaluator, step_start, step_end)
        return self._read_by_eval(evaluator)

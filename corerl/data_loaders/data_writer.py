from omegaconf import DictConfig
from datetime import datetime, UTC
from corerl.sql_logging.sql_logging import get_sql_engine
from sqlalchemy import text, Engine
from corerl.data_loaders.utils import try_connect


class DataWriter:
    def __init__(self, db_cfg: DictConfig, db_name: str, sensor_table_name: str, commit_every: int) -> None:
        db_data = dict(db_cfg)
        self.engine: Engine = get_sql_engine(db_data=db_data, db_name=db_name)
        self.sensor_table_name = sensor_table_name
        self.host = "localhost"
        self.commit_every = commit_every
        self._writes = 0
        self.connection = try_connect(self.engine)

    def write(
        self,
        timestamp: datetime,
        name: str,
        val: float,
        host: str | None = None,
        id: str | None = None,
        quality: str | None = None,
    ) -> None:
        if host is None:
            host = self.host
        if id is None:
            id = name
        if quality is None:
            quality = "The operation succeeded. StatusGood (0x0)"

        assert timestamp.tzinfo == UTC
        ts = timestamp.isoformat()
        
        jsonb_str = f'{{"val": {val}}}'
        insert_stmt = f"""
            INSERT INTO {self.sensor_table_name}
            (time, host, id, name, \"Quality\", fields)
            VALUES (TIMESTAMP '{ts}', '{host}', '{id}', '{name}', '{quality}', '{jsonb_str}');
        """

        self.connection.execute(text(insert_stmt))

        self._writes += 1
        if self._writes % self.commit_every == 0:
            self.commit()

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

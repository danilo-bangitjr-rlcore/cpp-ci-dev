from omegaconf import DictConfig
from datetime import datetime, UTC
from corerl.sql_logging.sql_logging import get_sql_engine
from sqlalchemy import text, Engine


class DataWriter:
    def __init__(
        self,
        db_cfg: DictConfig,
        db_name: str,
        sensor_table_name: str,
    ):
        # db_data = OmegaConf.to_container(db_cfg)
        db_data = dict(db_cfg)
        self.engine: Engine = get_sql_engine(db_data=db_data, db_name=db_name)
        self.ts_format = "%Y-%m-%d %H:%M:%S%z"
        self.sensor_table_name = sensor_table_name
        self.host = "localhost"

    def write(
        self,
        timestamp: datetime,
        name: str,
        val: float,
        host: str | None = None,
        id: str | None = None,
        quality: str | None = None,
    ):
        if host is None:
            host = self.host
        if id is None:
            id = name
        if quality is None:
            quality = "The operation succeeded. StatusGood (0x0)"

        assert timestamp.tzinfo == UTC
        ts: str = timestamp.strftime(self.ts_format)
        jsonb_str = f"'{{\"val\": {val}}}'"
        insert_stmt = f"""
            INSERT INTO {self.sensor_table_name}
            (time, host, id, name, \"Quality\", fields)
            VALUES (TIMESTAMP \'{ts}\', \'{host}\', \'{id}\', \'{name}\', \'{quality}\', {jsonb_str});
        """

        with self.engine.connect() as connection:
            connection.execute(text(insert_stmt))
            connection.commit()

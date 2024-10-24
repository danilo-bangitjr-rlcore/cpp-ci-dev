from omegaconf import OmegaConf, DictConfig
from datetime import datetime, UTC
from corerl.sql_logging.sql_logging import get_sql_engine

class DataWriter:
    def __init__(
        self,
        db_cfg: DictConfig,
        db_name: str,
    ):
        # db_data = OmegaConf.to_container(db_cfg)
        db_data = dict(db_cfg)
        self.engine = get_sql_engine(db_data=db_data, db_name=db_name)

    def write(self, timestamp: datetime, name: str, val: float):
        assert timestamp.tzinfo == UTC
        ...

from omegaconf import DictConfig
from datetime import datetime, UTC

class DataWriter:
    def __init__(
        self,
        username: str,
        password: str,
        db_name: str,
        ip: str = "localhost",
        port: int = 5432,
        drivername: str = "postgresql+psycopg2",
    ): ...

    def write(self, timestamp: datetime, name: str, val: float):
        assert timestamp.tzinfo == UTC
        ...

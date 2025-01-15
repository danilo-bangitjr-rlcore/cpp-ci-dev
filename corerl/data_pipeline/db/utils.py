import logging
import time
from types import TracebackType

from sqlalchemy import Connection, Engine

logger = logging.getLogger(__name__)


def try_connect(engine: Engine, backoff_seconds: int = 5, max_tries: int = 5) -> Connection:
    connection = None
    tries = 0
    while not connection is not None:
        if tries >= max_tries:
            raise Exception("sql engine connection failed")
        try:
            connection = engine.connect()
        except Exception:
            logger.warning(f"failed to connect sql engine, retrying in {backoff_seconds} seconds...")
            time.sleep(backoff_seconds)
        tries += 1

    return connection


class TryConnectContextManager(object):
    def __init__(self, engine: Engine, backoff_seconds: int = 5, max_tries: int = 5):
        self.engine = engine
        self.backoff_seconds = backoff_seconds
        self.max_tries = max_tries
        self.conn = None

    def __enter__(self):
        self.conn = try_connect(self.engine, self.backoff_seconds, self.max_tries)
        return self.conn

    def __exit__(
        self, exc_type: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None
    ):
        if self.conn:
            self.conn.close()

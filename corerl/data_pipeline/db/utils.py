import time
import logging
from sqlalchemy import Engine, Connection

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

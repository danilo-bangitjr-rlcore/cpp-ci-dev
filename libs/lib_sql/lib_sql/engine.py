import logging
import time
from typing import Protocol

import sqlalchemy
from sqlalchemy import URL, Engine

from lib_sql.database import maybe_create_database, maybe_drop_database

logger = logging.getLogger(__name__)

class SQLEngineConfigProtocol(Protocol):
    drivername: str
    username: str
    password: str
    ip: str
    port: int

def try_create_engine(url_object: URL, backoff_seconds: int = 5, max_tries: int = 5) -> Engine:
    engine: Engine | None = None
    tries = 0
    while engine is None:
        if tries >= max_tries:
            raise Exception("sql engine creation failed")
        try:
            engine = sqlalchemy.create_engine(url_object, pool_recycle=280, pool_pre_ping=True)
        except Exception:
            logger.warning(
                "failed to create sql engine, retrying in %s seconds...", backoff_seconds,
            )
            time.sleep(backoff_seconds)
        tries += 1
    return engine


def get_sql_engine(
    db_data: SQLEngineConfigProtocol, db_name: str, force_drop: bool = False,
    backoff_seconds: int = 5, max_tries: int = 5,
) -> Engine:
    url_object = sqlalchemy.URL.create(
        drivername=db_data.drivername,
        username=db_data.username,
        password=db_data.password,
        host=db_data.ip,
        port=db_data.port,
        database=db_name,
    )
    logger.debug("creating sql engine...")
    engine = try_create_engine(url_object=url_object, backoff_seconds=backoff_seconds, max_tries=max_tries)

    if force_drop:
        maybe_drop_database(engine.url)

    maybe_create_database(engine.url, backoff_seconds=1, max_tries=1)

    return engine

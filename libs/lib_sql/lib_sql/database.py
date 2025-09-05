import logging
import time

from sqlalchemy import URL
from sqlalchemy_utils import create_database, database_exists, drop_database

logger = logging.getLogger(__name__)

def maybe_drop_database(conn_url: URL) -> None:
    if not database_exists(conn_url):
        return
    drop_database(conn_url)


def maybe_create_database(
    conn_url: URL, backoff_seconds: int = 5, max_tries: int = 5,
) -> None:
    if database_exists(conn_url):
        return

    success = False
    tries = 0
    while not success:
        if tries >= max_tries:
            raise Exception("database creation failed")
        try:
            if database_exists(conn_url):
                success = True
            else:
                create_database(conn_url)
                success = True
        except Exception as e:
            logger.warning(
                "failed to create database, retrying in %s seconds...", backoff_seconds,
            )
            logger.error(e, exc_info=True)
            time.sleep(backoff_seconds)
        tries += 1

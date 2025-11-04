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
    conn_url: URL, backoff_seconds: float = 5, max_tries: int = 5,
) -> None:
    # Attempt database creation with retry logic that handles concurrent creation
    for attempt in range(max_tries):
        try:
            # Check if database already exists (handles idempotency)
            if database_exists(conn_url):
                return

            # Attempt to create the database
            create_database(conn_url)
            return  # Success!

        except Exception as e:
            # Check if database was created by another process during our attempt
            # This handles the race condition in parallel test execution
            try:
                if database_exists(conn_url):
                    return  # Another process created it, we're done
            except Exception:
                pass  # Ignore errors checking existence, will retry

            # If this was our last attempt, raise with context
            if attempt == max_tries - 1:
                raise Exception("database creation failed") from e

            # Log and retry with backoff
            logger.warning(
                "failed to create database (attempt %d/%d), retrying in %s seconds...",
                attempt + 1, max_tries, backoff_seconds,
            )
            logger.error(e, exc_info=True)
            time.sleep(backoff_seconds)

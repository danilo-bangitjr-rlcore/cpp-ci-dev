import os


def get_coreio_sqlite_path() -> str:
    """
    Get the path to the CoreIO SQLite database.
    This is used to store and retrieve configuration data.
    """
    # Check if the environment variable is set
    coreio_sqlite_path = os.environ.get("COREIO_SQLITE_DB_PATH", None)
    if not coreio_sqlite_path:
        # defer to the docker volume bound path
        coreio_sqlite_path = "/app/coreio-data/sqlite.db"
    return coreio_sqlite_path

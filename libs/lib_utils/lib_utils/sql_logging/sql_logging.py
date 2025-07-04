from __future__ import annotations

import logging
import time
from typing import Protocol

import sqlalchemy
from sqlalchemy import URL, Column, DateTime, Engine, MetaData, Table, inspect
from sqlalchemy.sql import func
from sqlalchemy_utils import create_database, database_exists, drop_database

logger = logging.getLogger(__name__)

class SQLEngineConfigProtocol(Protocol):
    drivername: str
    username: str
    password: str
    ip: str
    port: int

def get_sql_engine(db_data: SQLEngineConfigProtocol, db_name: str, force_drop: bool = False) -> Engine:
    url_object = sqlalchemy.URL.create(
        drivername=db_data.drivername,
        username=db_data.username,
        password=db_data.password,
        host=db_data.ip,
        port=db_data.port,
        database=db_name,
    )
    logger.debug("creating sql engine...")
    engine = try_create_engine(url_object=url_object)

    if force_drop:
        maybe_drop_database(engine.url)

    maybe_create_database(engine.url)

    return engine


def try_create_engine(url_object: URL, backoff_seconds: int = 5, max_tries: int = 5) -> Engine:
    engine = None
    tries = 0
    while engine is None:
        if tries >= max_tries:
            raise Exception("sql engine creation failed")
        try:
            engine = sqlalchemy.create_engine(url_object, pool_recycle=280, pool_pre_ping=True)
        except Exception:
            logger.warning(f"failed to create sql engine, retrying in {backoff_seconds} seconds...")
            time.sleep(backoff_seconds)
        tries += 1

    return engine

def maybe_drop_database(conn_url: URL) -> None:
    if not database_exists(conn_url):
        return
    drop_database(conn_url)

def maybe_create_database(conn_url: URL, backoff_seconds: int = 5, max_tries: int = 5) -> None:
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
            logger.warning(f"failed to create database, retrying in {backoff_seconds} seconds...")
            logger.error(e, exc_info=True)
            time.sleep(backoff_seconds)
        tries += 1


def create_column(name: str, dtype: str, primary_key: bool = False) -> Column:
    # TODO: support onupdate
    if dtype == "DateTime":
        col = Column(
            name,
            DateTime(timezone=True),
            server_default=func.now(),
            primary_key=primary_key,
        )
    else:
        dtype_obj = getattr(sqlalchemy, dtype)
        col = Column(name, dtype_obj, nullable=False, primary_key=primary_key)

    return col


def create_table(metadata: MetaData, schema: dict) -> Table:
    """
    schema like:

        name: critic_weights
        columns:
            id: Integer
            ts: DateTime
            network: BLOB
        primary_keys: [id]
        autoincrement: True

    """
    # TODO: test support compount primary keys

    cols = []
    for key in schema["columns"]:
        if key in schema["primary_keys"]:
            primary_key = True
        else:
            primary_key = False

        col = create_column(
            name=key, dtype=schema["columns"][key], primary_key=primary_key,
        )
        cols.append(col)

    return Table(schema["name"], metadata, *cols)



def create_tables(metadata: MetaData, engine: Engine, schemas: dict) -> None:
    for table_name, values in schemas.items():
        create_table(
            metadata=metadata, schema={"name": table_name, **values},
        )

    metadata.create_all(engine, checkfirst=True)

def table_exists(engine: Engine, table_name: str, schema: str = 'public') -> bool:
    iengine = inspect(engine)
    existing_tables = iengine.get_table_names(schema)

    return table_name in existing_tables

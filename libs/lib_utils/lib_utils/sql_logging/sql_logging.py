from __future__ import annotations

import logging
import time
from typing import Protocol

import sqlalchemy
from sqlalchemy import (
    CHAR,
    URL,
    BigInteger,
    Double,
    Engine,
    Float,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    inspect,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.types import TypeEngine
from sqlalchemy_utils import create_database, database_exists, drop_database

from lib_utils.list import find

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

def table_exists(engine: Engine, table_name: str, schema: str = 'public') -> bool:
    iengine = inspect(engine)
    existing_tables = iengine.get_table_names(schema)

    return table_name in existing_tables

def column_exists(engine: Engine, table_name: str, column_name: str,  schema: str = 'public') -> bool:
    try:
        iengine = inspect(engine)
        columns = iengine.get_columns(table_name, schema=schema)
        return any(col['name'] == column_name for col in columns)
    except SQLAlchemyError:
        return False

def get_column_type(engine: Engine, table_name: str, column_name: str,  schema: str = 'public'):
    assert column_exists(engine, table_name, column_name, schema), "SQL Error, column not found"
    iengine = inspect(engine)
    columns = iengine.get_columns(table_name, schema=schema)
    column = find(lambda col: col["name"] == column_name, columns)
    assert column is not None # Check for pyright, since we already checked that column exists
    return column["type"]


def are_types_compatible(existing_type: TypeEngine, expected_type: TypeEngine):
    if type(existing_type) is type(expected_type):
        return True

    existing_class = type(existing_type)
    expected_class = type(expected_type)

    safe_conversions = {

        # Numeric widening (safe)
        SmallInteger: (Integer, BigInteger, Float, Double, Numeric),
        Integer: (BigInteger, Float, Double, Numeric),
        BigInteger: (Float, Numeric, Double),
        Float: (Numeric, Double),
        Double: (Numeric,),

        # String widening (safe)
        String: (Text,),  # VARCHAR to TEXT
        CHAR: (String, Text),

        # Add more as needed ...
    }

    # Check if conversion is explicitly safe
    if existing_class in safe_conversions:
        return expected_class in safe_conversions[existing_class]

    return False

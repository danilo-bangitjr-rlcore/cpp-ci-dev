import hashlib
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NewType

from sqlalchemy.sql import text


@dataclass
class SQLColumn:
    name: str
    type: str
    nullable: bool = False
    primary: bool = False


def create_tsdb_table_query(
    schema: str,
    table: str,
    columns: list[SQLColumn],
    partition_column: str | None = None,
    index_columns: list[str] | None = None,
    time_column: str = 'time',
    chunk_time_interval: str = '1d',
):
    # Strict requirement: TSDB tables must have a time column
    if not columns:
        raise ValueError("TSDB table creation requires at least one column to be defined")

    if not any(col.name == time_column for col in columns):
        raise ValueError(f"TSDB table creation requires a '{time_column}' column to be defined in the schema")

    if index_columns is None:
        index_columns = []

    schema_builder = ''
    if schema != 'public':
        schema_builder = f'CREATE SCHEMA IF NOT EXISTS {schema};'

    schema_table = schema + '.' + table

    # Build columns and determine primary key columns
    cols_list: list[str] = []
    primary_key_columns: list[str] = []

    # Find user-defined primary key columns
    for col in columns:
        nullable_clause = "NOT NULL" if not col.nullable else ""
        cols_list.append(f'{col.name} {col.type} {nullable_clause}')
        if col.primary:
            primary_key_columns.append(col.name)

    # Only create primary key if columns are explicitly marked as primary
    cols = ', '.join(cols_list)
    primary_key = f"PRIMARY KEY ({', '.join(primary_key_columns)})" if primary_key_columns else ""

    idxs = '\n'.join([
        f'CREATE INDEX IF NOT EXISTS {table}_{col}_idx ON {schema_table} ({col});'
        for col in index_columns
    ])

    cti = chunk_time_interval

    # Create hypertable and compression (we're guaranteed to have a time column)
    hypertable_sql = f"""
        SELECT create_hypertable(
            '{schema_table}', '{time_column}',
            if_not_exists => TRUE,
            chunk_time_interval => INTERVAL '{cti}'
        );"""
    compress_sql = f"""
        ALTER TABLE {schema_table} SET (
            timescaledb.compress,
            timescaledb.compress_segmentby='{partition_column if partition_column is not None else ""}'
        );"""

    table_sql = f"""
        {schema_builder}
        CREATE TABLE {schema_table} (
            {cols}{f',\n {primary_key}' if primary_key else ''}
        );
        {hypertable_sql}
        {idxs}
        {compress_sql}
    """

    return text(table_sql)

def add_column_to_table_query(
    schema: str,
    table: str,
    column: SQLColumn,
):
    return text(f"""ALTER TABLE {schema}.{table}
         ADD COLUMN {column.name} {column.type}
         {"NOT NULL" if not column.nullable else ""};""")


def create_sqlite_table_query(
    schema: str,
    table: str,
    columns: list[SQLColumn],
    index_columns: list[str] | None = None,
):
    """Create a SQLite table with the given columns and indexes.

    Note: SQLite doesn't support schemas like PostgreSQL, so the schema parameter is ignored.
    partition_column, time_column, and chunk_time_interval are ignored for SQLite compatibility.
    """
    if index_columns is None:
        index_columns = []

    # SQLite doesn't support schemas, use table name directly
    table_name = table

    # Build columns list from schema
    cols_list = []

    for col in columns:
        nullable_clause = "NOT NULL" if not col.nullable else ""
        cols_list.append(f'{col.name} {col.type} {nullable_clause}')

    cols = ', '.join(cols_list)

    idxs = '\n'.join([
        f'CREATE INDEX IF NOT EXISTS {table}_{col}_idx ON {table_name} ({col});'
        for col in index_columns
    ])

    return text(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {cols}
        );
        {idxs}
    """)


#######################
### Name sanitation ###
#######################

SanitizedName = NewType("SanitizedName", str)

class ColumnMapper:
    def __init__(self, columns: list[str]):
        clean_names = _clean_names_with_hash_disambiguation(columns)
        self.name_to_pg: dict[str, SanitizedName] = dict(zip(columns, clean_names, strict=True))
        self.pg_to_name: dict[SanitizedName, str] = {v: k  for k, v in self.name_to_pg.items()}

def _clean_names_with_hash_disambiguation(names: list[str]):
    # Sanitize all names
    cleaned_names = [_sanitize_key(name) for name in names]

    # Find duplicates
    seen = defaultdict(list)
    for i, cleaned in enumerate(cleaned_names):
        seen[cleaned].append(i)

    result = cleaned_names.copy()

    # Append hash to duplicates
    for cleaned, indices in seen.items():
        if len(indices) > 1:
            for idx in indices:
                original = names[idx]
                hash_suffix = _get_short_hash(original)
                result[idx] = f"{cleaned}_{hash_suffix}"

    # Cast to type SanitizedName
    return [SanitizedName(res) for res in result]


def _get_short_hash(text: str, length: int = 4):
    """Generate a short _deterministic_ hash from text"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:length]

def _sanitize_key(name: str):
    # remove non alpha-numeric characters and spaces
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', name)
    # Replace multiple consecutive underscores with single underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # lowercase
    return sanitized.lower()

def sanitize_keys(dict_points: Sequence[dict]):
    def _sanitize_dict_keys(d: dict[str, Any]) -> dict[str, Any]:
        return {
            _sanitize_key(key): value
            for key, value in d.items()
        }

    return [_sanitize_dict_keys(point) for point in dict_points]

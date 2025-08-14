import re
from dataclasses import dataclass
from typing import Any, NewType

from sqlalchemy.sql import text


@dataclass
class SQLColumn:
    name: str
    type: str
    nullable: bool = False


def create_tsdb_table_query(
    schema: str,
    table: str,
    columns: list[SQLColumn],
    partition_column: str | None,
    index_columns: list[str],
    time_column: str = 'time',
    chunk_time_interval: str = '1d',
):
    schema_builder = ''
    if schema != 'public':
        schema_builder = f'CREATE SCHEMA IF NOT EXISTS {schema};'

    schema_table = schema + '.' + table
    cols = ', '.join([
        f'{col.name} {col.type} {"NOT NULL" if not col.nullable else ""}'
        for col in columns
    ])

    idxs = '\n'.join([
        f'CREATE INDEX IF NOT EXISTS {table}_{col}_idx ON {schema_table} ({col});'
        for col in index_columns
    ])

    cti = chunk_time_interval
    return text(f"""
        {schema_builder}
        CREATE TABLE {schema_table} (
            {cols}
        );
        SELECT create_hypertable(
            '{schema_table}', '{time_column}',
            if_not_exists => TRUE,
            chunk_time_interval => INTERVAL '{cti}'
        );
        {idxs}
        ALTER TABLE {schema_table} SET (
            timescaledb.compress,
            timescaledb.compress_segmentby='{partition_column if partition_column is not None else ""}'
        );
    """)

def add_column_to_table_query(
    schema: str,
    table: str,
    column: SQLColumn,
):
    return text(f"""ALTER TABLE {schema}.{table}
         ADD COLUMN {column.name} {column.type}
         {"NOT NULL" if not column.nullable else ""};""")


#######################
### Name sanitation ###
#######################

SanitizedName = NewType("SanitizedName", str)

class ColumnMapper:
    def __init__(self, columns: list[str]):
        self.name_to_pg: dict[str, SanitizedName] = {name: sanitize_key(name) for name in columns}
        self.pg_to_name: dict[SanitizedName, str] = {v: k  for k, v in self.name_to_pg.items()}

def sanitize_key(name: str):
    # remove non alpha-numeric characters and spaces
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', name)
    # Replace multiple consecutive underscores with single underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # lowercase
    sanitized = sanitized.lower()

    return SanitizedName(sanitized)

def sanitize_keys(dict_points: list[dict]):

    def _sanitize_dict_keys(d: dict[str, Any]):
        keys = list(d.keys())
        for key in keys:
            sanitized_key = sanitize_key(key)
            if sanitized_key != key:
                d[sanitized_key] = d.pop(key)

    # Sanitize the dictionary keys
    for point in dict_points:
        _sanitize_dict_keys(point)

    return dict_points

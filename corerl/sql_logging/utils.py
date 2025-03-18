from dataclasses import dataclass

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
    partition_column: str,
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
        SELECT create_hypertable('{schema_table}', '{time_column}', chunk_time_interval => INTERVAL '{cti}');
        {idxs}
        ALTER TABLE {schema_table} SET (
            timescaledb.compress,
            timescaledb.compress_segmentby='{partition_column}'
        );
    """)

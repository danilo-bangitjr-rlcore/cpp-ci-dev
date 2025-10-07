from pathlib import Path
from typing import Any

import pandas as pd
from corerl.data_pipeline.db.data_writer import DataWriter, TagDBConfig
from sqlalchemy import Connection, Engine, text


def get_offline_data_writer(engine: Engine, infra_overrides: dict[str, object]) -> DataWriter:
    drivername = engine.url.drivername
    username = engine.url.username
    password = engine.url.password
    ip = engine.url.host
    port = engine.url.port
    assert drivername is not None
    assert username is not None
    assert password is not None
    assert ip is not None
    assert port is not None

    db_cfg = TagDBConfig(
        drivername=drivername,
        username=username,
        password=password,
        ip=ip,
        port=port,
        db_name=str(infra_overrides['infra.db.db_name']),
        table_schema=str(infra_overrides['infra.db.schema']),
        enabled=True,
    )
    return DataWriter(db_cfg)

def read_offline_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "Timestamp"})
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
    df = df.set_index("Timestamp")
    return df.dropna()

def column_to_sql_tups(column: pd.Series) -> list[tuple[Any, ...]]:
    """
    Converting pd.Series into (timestamp, tag_name, value) tuples
    """
    tag = column.name
    df = column.to_frame()
    df["Tag"] = [tag] * len(df)
    return list(df.itertuples(index=True, name=None))

def get_active_branch(base: Path = Path('.')) -> str:
    refs = (base / '.git/HEAD').read_text()

    for ref in refs.splitlines():
        if ref.startswith('ref: '):
            return ref.partition('refs/heads/')[2]

    raise Exception('Was unable to determine active branch')


def add_retention_policy(conn: Connection, table_name: str, schema: str, days: int):
    try:
        conn.execute(text(f"SELECT add_retention_policy('{schema}.{table_name}', INTERVAL '{days}d');"))
        conn.commit()
    except Exception:
        ...

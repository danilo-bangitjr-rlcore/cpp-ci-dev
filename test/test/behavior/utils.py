from pathlib import Path
from typing import Any

import pandas as pd
from corerl.data_pipeline.db.data_writer import DataWriter, TagDBConfig
from sqlalchemy import Engine


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
        table_schema=str(infra_overrides['infra.db.schema'])
    )
    return DataWriter(db_cfg)

def read_offline_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "Timestamp"})
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
    df = df.set_index("Timestamp")
    df = df.dropna()
    return df

def column_to_sql_tups(column: pd.Series) -> list[tuple[Any, ...]]:
    """
    Converting pd.Series into (timestamp, tag_name, value) tuples
    """
    tag = column.name
    df = column.to_frame()
    df["Tag"] = [tag] * len(df)
    return list(df.itertuples(index=True, name=None))

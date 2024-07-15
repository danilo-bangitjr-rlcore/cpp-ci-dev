import sqlalchemy
from sqlalchemy import Engine, MetaData
from sqlalchemy import Table, Column, DateTime, ARRAY
from sqlalchemy.sql import func
from sqlalchemy_utils import database_exists, create_database

def get_sql_engine(db_data: dict, db_name: str) -> Engine:
    url_object = sqlalchemy.URL.create(
        "mysql+pymysql",
        username=db_data["username"],
        password=db_data["password"],
        host=db_data["ip"],
        port=db_data["port"],
        database=db_name,
    )
    engine = sqlalchemy.create_engine(url_object)

    if not database_exists(engine.url):
        create_database(engine.url)

    return engine

def create_column(name: str, dtype: str, primary_key: bool=False) -> Column:
    # TODO: support onupdate
    if dtype == "DateTime":
        col = Column(name, DateTime(timezone=True), server_default=func.now(), primary_key=primary_key)
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
            name=key,
            dtype=schema["columns"][key],
            primary_key=primary_key
        )
        cols.append(col)

    table = Table(
        schema["name"],
        metadata,
        *cols
    )

    return table


def create_tables(metadata: MetaData, engine: Engine, schemas: dict) -> None:
    for table_name in schemas:
        create_table(metadata=metadata, schema={"name": table_name, **schemas[table_name]})
    
    metadata.create_all(engine, checkfirst=True)
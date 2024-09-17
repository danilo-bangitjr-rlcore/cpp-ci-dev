import sqlalchemy
import corerl.utils.dict as dict_u

from typing import Any
from collections.abc import MutableMapping
from sqlalchemy import Engine, MetaData
from sqlalchemy import Table, Column, DateTime
from sqlalchemy.sql import func
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy_utils import database_exists, drop_database, create_database
from sqlalchemy import select
from omegaconf import OmegaConf
from corerl.sql_logging.base_schema import (
    Base,
    Run,
    HParam,
)
from sqlalchemy.orm import Session

import logging
from sqlalchemy import inspect
# from sqlalchemy.ext.declarative.clsregistry import _ModuleMarker

logger = logging.getLogger(__name__)


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
            name=key, dtype=schema["columns"][key], primary_key=primary_key
        )
        cols.append(col)

    table = Table(schema["name"], metadata, *cols)

    return table


def create_tables(metadata: MetaData, engine: Engine, schemas: dict) -> None:
    for table_name in schemas:
        create_table(
            metadata=metadata, schema={"name": table_name, **schemas[table_name]}
        )

    metadata.create_all(engine, checkfirst=True)

def is_sane_database(engine):
    """
    adapted from stackoverflow:
      https://stackoverflow.com/questions/30428639/check-database-schema-matches-sqlalchemy-models-on-application-startup
    Check whether the current database matches the models declared in model base.

    Currently we check that all tables exist with all columns. What is not checked

    * Column types are not verified

    * Relationships are not verified at all (TODO)

    :param Base: Declarative Base for SQLAlchemy models to check

    :param session: SQLAlchemy session bound to an engine

    :return: True if all declared models have corresponding tables and columns.
    """
    iengine = inspect(engine)

    errors = False

    exisiting_tables = iengine.get_table_names()

    # Go through all SQLAlchemy models
    for table in Base.metadata.sorted_tables:

        if table.name in exisiting_tables:
            # Check all columns are found
            # Looks like [{'default': "nextval('sanity_check_test_id_seq'::regclass)", 'autoincrement': True, 'nullable': False, 'type': INTEGER(), 'name': 'id'}]

            exisiting_table_cols = [c["name"] for c in iengine.get_columns(table.name)]
            for column in table.columns:
                # Assume normal flat column
                if not column.key in exisiting_table_cols:
                    logger.warning("Schema declares column %s which does not exist in table %s", column.key, table.name)
                    errors = True
        else:
            logger.warning("Schema declares table %s which does not exist in database %s", table, engine)
            errors = True

    return not errors

# utils
def setup_sql_logging(cfg, restart_db=False):
    logger.info("Setting up sql db...")

    con_cfg = cfg.agent.buffer.con_cfg
    flattened_cfg = prep_cfg_for_db(cfg, to_remove=[])
    db_name = cfg.agent.buffer.db_name
    engine = get_sql_engine(con_cfg, db_name=db_name)
    
    if restart_db:
        drop_database(engine.url)
        create_database(engine.url)    
        
    Base.metadata.create_all(engine) # create tables

    # check if there is a problem with an existing db schema    
    while not is_sane_database(engine):
        try:
            db_version = int(db_name.split('_v')[-1]) + 1
            base_db_name = db_name.split('_v')[0]
        except:
            db_version = 2
            base_db_name = db_name

        db_name = f'{base_db_name}_v{db_version}'
        logger.warning(f'Trying db with name {db_name}...')
        logger.warning(f'To avoid this change the db_name in your config!')
        engine = get_sql_engine(con_cfg, db_name=db_name)

        Base.metadata.create_all(engine)
        
    with Session(engine) as session:
        run = Run(
            hparams=[HParam(name=name, val=val) for name, val in flattened_cfg.items()]
        )
        session.add(run)
        session.commit()

        run_id = session.scalar(select(Run.run_id).order_by(Run.run_id.desc()))
        logger.info(f"{run_id=}")

        return session, run


def prep_cfg_for_db(cfg: Any, to_remove: list[str]) -> dict:
    cfg_dict = OmegaConf.to_container(cfg)
    assert isinstance(cfg_dict, MutableMapping)

    cgf_dict = dict_u.drop(cfg_dict, to_remove)
    return dict_u.flatten(cgf_dict)

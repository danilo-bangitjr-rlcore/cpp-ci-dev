from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any

import lib_utils.dict as dict_u
import sqlalchemy
from lib_config.config import MISSING, computed, config
from lib_utils.sql_logging.sql_logging import maybe_create_database, maybe_drop_database, try_create_engine
from sqlalchemy import Connection, Engine, inspect, select, text
from sqlalchemy.orm import Session
from sqlalchemy_utils import create_database, drop_database

from corerl.sql_logging.base_schema import (
    Base,
    HParam,
    Run,
)

if TYPE_CHECKING:
    from corerl.config import MainConfig

logger = logging.getLogger(__name__)

@config()
class SQLEngineConfig:
    drivername: str = MISSING
    username: str = MISSING
    password: str = MISSING
    ip: str = MISSING
    port: int = MISSING

    @computed('drivername')
    @classmethod
    def _drivername(cls, cfg: MainConfig):
        return cfg.infra.db.drivername

    @computed('username')
    @classmethod
    def _username(cls, cfg: MainConfig):
        return cfg.infra.db.username

    @computed('password')
    @classmethod
    def _password(cls, cfg: MainConfig):
        return cfg.infra.db.password

    @computed('ip')
    @classmethod
    def _ip(cls, cfg: MainConfig):
        return cfg.infra.db.ip

    @computed('port')
    @classmethod
    def _port(cls, cfg: MainConfig):
        return cfg.infra.db.port


def get_sql_engine(db_data: SQLEngineConfig, db_name: str, force_drop: bool = False) -> Engine:
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


def is_sane_database(engine: Engine):
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
            # Looks like [{
            #   'default': "nextval('sanity_check_test_id_seq'::regclass)",
            #   'autoincrement': True,
            #   'nullable': False,
            #   'type': INTEGER(),
            #   'name': 'id'
            # }]

            exisiting_table_cols = [c["name"] for c in iengine.get_columns(table.name)]
            for column in table.columns:
                # Assume normal flat column
                if column.key not in exisiting_table_cols:
                    logger.warning("Schema declares column %s which does not exist in table %s", column.key, table.name)
                    errors = True
        else:
            logger.warning("Schema declares table %s which does not exist in database %s", table, engine)
            errors = True

    return not errors

# utils
def setup_sql_logging(cfg: Any, restart_db: bool = False):
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
        except Exception:
            db_version = 2
            base_db_name = db_name

        db_name = f'{base_db_name}_v{db_version}'
        logger.warning(f'Trying db with name {db_name}...')
        logger.warning('To avoid this change the db_name in your config!')
        engine = get_sql_engine(con_cfg, db_name=db_name)

        Base.metadata.create_all(engine)

    with Session(engine) as session:
        run = Run(
            hparams=[HParam(name=name, val=val) for name, val in flattened_cfg.items()],
        )
        session.add(run)
        session.commit()

        run_id = session.scalar(select(Run.run_id).order_by(Run.run_id.desc()))
        logger.info(f"{run_id=}")

        return session, run


def prep_cfg_for_db(cfg: Any, to_remove: list[str]) -> dict:
    cfg_dict = dict_u.dataclass_to_dict(cfg)
    assert isinstance(cfg_dict, MutableMapping)

    cgf_dict = dict_u.drop(cfg_dict, to_remove)
    return dict_u.flatten(cgf_dict)


def add_retention_policy(conn: Connection, table_name: str, schema: str, days: int):
    try:
        conn.execute(text(f"SELECT add_retention_policy('{schema}.{table_name}', INTERVAL '{days}d');"))
        conn.commit()
    except Exception:
        ...

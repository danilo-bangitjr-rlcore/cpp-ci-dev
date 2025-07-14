import logging
from collections.abc import Sequence

from lib_utils.sql_logging.sql_logging import column_exists, get_sql_engine, table_exists

from coreio.utils.config_schemas import InfraConfigAdapter, TagConfigAdapter

logger = logging.getLogger(__name__)

class SQL_Manager:
    def __init__(self, cfg: InfraConfigAdapter, table_name: str):
        self.engine = get_sql_engine(db_data=cfg.db, db_name=cfg.db.db_name)
        self.schema = cfg.db.schema
        self.table_name = table_name
        self.time_column_name = "time"

    def ensure_table_exists(self, tag_configs: Sequence[TagConfigAdapter]):
        if table_exists(self.engine, self.table_name, schema=self.schema):
            # Verify table structure
            for tag in tag_configs:
                # To persist a tag in db, it simply has to have a non-None connection_id
                if tag.node_identifier is not None and tag.connection_id is not None:
                    self.ensure_column_exists(tag.name)
                    # ensure column exists

        else:
            # Create table
            ...

        # Do something with tag configs
        logger.debug(tag_configs)


    def ensure_column_exists(self, column_name: str):
        if column_exists(self.engine, table_name=self.table_name, column_name=column_name, schema=self.schema):
            # Check correct datatype
            ...
        else:
            # Create column
            ...

import logging
from os import name

from lib_utils.sql_logging.sql_logging import (
    are_types_compatible,
    column_exists,
    get_column_type,
    get_sql_engine,
    table_exists,
)
from lib_utils.sql_logging.utils import SQLColumn, create_tsdb_table_query
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import TypeEngine

from coreio.communication.opc_communication import NodeData
from coreio.utils.config_schemas import InfraConfigAdapter
from coreio.utils.opc_utils import OPC_TO_SQLALCHEMY_TYPE_MAP

logger = logging.getLogger(__name__)

class SQL_Manager:
    def __init__(self, cfg: InfraConfigAdapter, table_name: str, nodes_to_persist: dict[str, NodeData]):
        self.engine = get_sql_engine(db_data=cfg.db, db_name=cfg.db.db_name)
        self.schema = cfg.db.schema
        self.table_name = table_name
        self.time_column_name = "time"
        self.nodes_to_persist = nodes_to_persist

        self._ensure_db_schema()


    def ensure_db_schema(self, nodes_to_persist: dict[str, NodeData]):
        if table_exists(self.engine, self.table_name, schema=self.schema):
            logger.info(f"Table {self.schema}.{self.table_name} already exists!")
            # Verify table structure
            for node in self.nodes_to_persist.values():
                expected_type = self._get_sqlalchemy_type(node)
                # To persist a tag in db, it simply has to have a non-None connection_id
                self._ensure_column_exists(node.name, expected_type)
                    # ensure column exists
        else:
            logger.info(f"Table {self.schema}.{self.table_name} not found. Creating...")
            with TryConnectContextManager(self.engine) as connection:
                connection.execute(self._create_table_sql())
                connection.commit()


    def _ensure_column_exists(self, column_name: str, expected_type: TypeEngine):
        if column_exists(self.engine, table_name=self.table_name, column_name=column_name, schema=self.schema):
            # Check correct datatype
            existing_type = get_column_type(self.engine, self.table_name, column_name, self.schema)
            if type(existing_type) is type(expected_type):
                logger.info(f"> Column {column_name} found in db")
            elif are_types_compatible(existing_type, expected_type):
                logger.warning(f"Column {column_name} in table {self.schema}.{self.table_name} "
                    f"has expected type {type(expected_type)} but was of type {type(existing_type)}. "
                    "Type compatibility check OK")
            else:
                error_msg = (f"Column {column_name} in table {self.schema}.{self.table_name} "
                    f"has expected type {type(expected_type)} but was of type {type(existing_type)} "
                    "Type compatibility check FAIL")
                logger.error(error_msg)
                raise TypeError(error_msg)
        else:
            logger.info(f"Column {column_name} not found")
            logger.info("To do: add column to table")


    def _create_table_sql(self):
        columns: list[SQLColumn] = [SQLColumn(name="time", type="TIMESTAMP WITH TIME ZONE", nullable=False)]
        for node in self.nodes_to_persist.values():
            tsdb_type = self._get_tsdb_type(node)
            column = SQLColumn(name=name, type=tsdb_type, nullable=True)
            columns.append(column)

        return create_tsdb_table_query(
            schema=self.schema,
            table=self.table_name,
            columns=columns,
            partition_column=None,
            index_columns=['time'],
            chunk_time_interval='7d',
        )

    def _get_sqlalchemy_type(self, node: NodeData):
        return OPC_TO_SQLALCHEMY_TYPE_MAP[node.var_type]

    def _get_tsdb_type(self, node: NodeData):
        sqlalchemy_type = self._get_sqlalchemy_type(node)
        dialect = postgresql.dialect()
        return sqlalchemy_type.compile(dialect=dialect)

    def _get_tsdb_type(self, node: NodeData):
        sqlalchemy_type = self._get_sqlalchemy_type(node)
        return self._sqlalchemy_to_tsdb_type(sqlalchemy_type)

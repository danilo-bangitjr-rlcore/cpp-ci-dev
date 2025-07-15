import logging

from lib_utils.sql_logging.connect_engine import TryConnectContextManager
from lib_utils.sql_logging.sql_logging import (
    column_exists,
    get_all_columns,
    get_column_type,
    get_sql_engine,
    table_exists,
)
from lib_utils.sql_logging.utils import SQLColumn, add_column_to_table_query, create_tsdb_table_query
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import TypeEngine

from coreio.communication.opc_communication import NodeData
from coreio.utils.config_schemas import InfraConfigAdapter
from coreio.utils.opc_utils import OPC_TO_SQLALCHEMY_TYPE_MAP

logger = logging.getLogger(__name__)

class SQL_Manager:
    def __init__(
        self,
        cfg: InfraConfigAdapter,
        table_name: str,
        nodes_to_persist: dict[str, NodeData],
        warn_extra_cols: bool = True,
    ):

        self.engine = get_sql_engine(db_data=cfg.db, db_name=cfg.db.db_name)
        self.schema = cfg.db.schema
        self.table_name = table_name
        self.time_column_name = "time"
        self.nodes_to_persist = nodes_to_persist
        self._node_names_to_lower()

        self._ensure_db_schema()
        if warn_extra_cols:
            self._check_for_extra_columns()

    def _node_names_to_lower(self):
        for node in  self.nodes_to_persist.values():
            node.name = node.name.lower()

    def _ensure_db_schema(self):
        if table_exists(self.engine, self.table_name, schema=self.schema):
            logger.info(f"Table {self.schema}.{self.table_name} already exists!")
            logger.info("Validating table columns...")
            for node in self.nodes_to_persist.values():
                self._ensure_column_exists(node)
        else:
            logger.info(f"Table {self.schema}.{self.table_name} not found. Creating...")
            with TryConnectContextManager(self.engine) as connection:
                connection.execute(self._create_table_sql())
                connection.commit()

    def _ensure_column_exists(self, node: NodeData):
        if column_exists(self.engine, table_name=self.table_name, column_name=node.name, schema=self.schema):
            self._check_column_type(node)
        else:
            self._create_column_from_node(node)

    def _check_column_type(self,  node: NodeData):
        expected_type = self._get_sqlalchemy_type(node)
        existing_type = get_column_type(self.engine, self.table_name, node.name, self.schema)

        tsdb_expected_type = self._sqlalchemy_to_tsdb_type(expected_type)
        tsdb_existing_type = self._sqlalchemy_to_tsdb_type(existing_type)

        if tsdb_expected_type == tsdb_existing_type:
            logger.info(f"  Column {node.name} found in db")

        elif expected_type._type_affinity == existing_type._type_affinity:
            logger.warning(f"  Column {node.name} in table {self.schema}.{self.table_name} "
                f"has expected type {type(expected_type)} but is of type {type(existing_type)}. "
                "Type mismatch but compatible")
        else:
            error_msg = (f"  Column {node.name} in table {self.schema}.{self.table_name} "
                f"has expected type {type(expected_type)} but was of type {type(existing_type)}. "
                "Incompatible types!")

            logger.error(error_msg)
            raise TypeError(error_msg)

    def _create_column_from_node(self, node: NodeData):
        logger.info(f"Column {node.name} not found. Creating...")
        with TryConnectContextManager(self.engine) as connection:
            tsdb_type = self._get_tsdb_type(node)
            column = SQLColumn(name=node.name, type=tsdb_type, nullable=True)
            connection.execute(add_column_to_table_query(self.schema, self.table_name, column=column))
            connection.commit()

    def _create_table_sql(self):
        columns: list[SQLColumn] = [SQLColumn(name="time", type="TIMESTAMP WITH TIME ZONE", nullable=False)]
        for node in self.nodes_to_persist.values():
            tsdb_type = self._get_tsdb_type(node)
            column = SQLColumn(name=node.name, type=tsdb_type, nullable=True)
            columns.append(column)

        return create_tsdb_table_query(
            schema=self.schema,
            table=self.table_name,
            columns=columns,
            partition_column=None,
            index_columns=['time'],
            chunk_time_interval='7d',
        )

    def _check_for_extra_columns(self):
        existing_col_names = [col["name"] for col in get_all_columns(self.engine, self.table_name, schema=self.schema)]
        expected_col_names = [node.name for node in self.nodes_to_persist.values()] + [self.time_column_name]

        for existing_col in existing_col_names:
            if existing_col not in expected_col_names:
                logger.warning(f"Column {existing_col} found in database, not in tag config.")

    def _get_sqlalchemy_type(self, node: NodeData):
        return OPC_TO_SQLALCHEMY_TYPE_MAP[node.var_type]

    def _sqlalchemy_to_tsdb_type(self, sqlalchemy_type: TypeEngine):
        dialect = postgresql.dialect()
        return sqlalchemy_type.compile(dialect=dialect)

    def _get_tsdb_type(self, node: NodeData):
        sqlalchemy_type = self._get_sqlalchemy_type(node)
        return self._sqlalchemy_to_tsdb_type(sqlalchemy_type)


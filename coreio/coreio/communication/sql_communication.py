import logging
from typing import Any

from lib_utils.sql_logging.connect_engine import TryConnectContextManager
from lib_utils.sql_logging.sql_logging import (
    column_exists,
    get_all_columns,
    get_column_type,
    get_sql_engine,
    table_exists,
)
from lib_utils.sql_logging.utils import (
    ColumnMapper,
    SanitizedName,
    SQLColumn,
    add_column_to_table_query,
    create_tsdb_table_query,
)
from lib_utils.time import now_iso
from sqlalchemy import text
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

        column_names = [node.name for node in nodes_to_persist.values()]
        self.column_mapper = ColumnMapper(column_names)
        self.nodes_to_persist = {
            key: node.model_copy(update={"name": self.column_mapper.name_to_pg[node.name]})
            for key, node in nodes_to_persist.items()
        }

        self._ensure_db_schema()
        if warn_extra_cols:
            self._check_for_extra_columns()


    def _ensure_db_schema(self):
        """Ensure database table exists with required columns, create table/columns if necessary."""

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
        """Ensure column exists with expected (or compatible) type, create if necessary."""

        if column_exists(self.engine, table_name=self.table_name, column_name=node.name, schema=self.schema):
            self._check_column_type(node)
        else:
            logger.info(f"Column {node.name} not found. Creating...")
            self._create_column_from_node(node)

    def _check_column_type(self,  node: NodeData):
        """Check expected vs existing column type. Raise error if incompatible."""

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
        """Adds column to tsdb table based on NodeData"""

        with TryConnectContextManager(self.engine) as connection:
            tsdb_type = self._get_tsdb_type(node)
            column = SQLColumn(name=node.name, type=tsdb_type, nullable=True)
            connection.execute(add_column_to_table_query(self.schema, self.table_name, column=column))
            connection.commit()

    def _create_table_sql(self):
        """Sets up query to create tsdb hypertable"""

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
        """Check for database columns not present in the current tag configuration."""

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

    def _insert_sql(self, col_names_to_write: list[SanitizedName]):
        """Generates SQL query for inserting data into the tsdb table."""

        columns = ', '.join([f"{col}" for col in col_names_to_write])
        placeholders = ', '.join([f":{col}" for col in col_names_to_write])
        return text(f"""
            INSERT INTO {self.schema}.{self.table_name}
            (time, {columns})
            VALUES (TIMESTAMP WITH TIME ZONE :timestamp, {placeholders});
        """)

    # Write data to the tsdb table with the given timestamp.
    # This method expects data to be a dictionary where keys are node names
    # and values are their corresponding values.
    def write_to_sql(self, data: dict[str, Any], timestamp: str | None):
        """Write data to the tsdb table with the given timestamp."""

        if not self.engine:
            logger.error("SQL engine is not initialized")
            return

        if not data:
            logger.warning("No data provided to write_to_sql")
            return

        # Ensure all keys in input data are sanitized
        column_mapper = ColumnMapper(list(data.keys()))
        sanitized_data = {column_mapper.name_to_pg[k]: v for k, v in data.items()}

        # Ensure all keys in input data are only expected columns
        valid_columns = {node.name for node in self.nodes_to_persist.values()}
        filtered_data = {k: v for k, v in sanitized_data.items() if k in valid_columns}

        if not filtered_data:
            logger.warning("No valid columns found in provided data")
            return

        filtered_data[SanitizedName('timestamp')] = timestamp or now_iso()

        try:
            with TryConnectContextManager(self.engine, backoff_seconds=10, max_tries=99) as connection:
                col_names_to_write = [col for col in filtered_data.keys() if col != "timestamp"]
                sql = self._insert_sql(col_names_to_write)

                # Cast SanitizedName to str just for connection.execute
                connection.execute(sql, {str(k): v for k, v in filtered_data.items()})
                connection.commit()
        except Exception as exc:
            logger.error(f"Failed to write nodes: {exc}")

#!/usr/bin/env python
import asyncio
import csv
import logging
import sqlite3
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from numbers import Number

from asyncua import Client
from asyncua.ua.uaerrors import BadNodeIdUnknown, BadNodeIdExists, BadTypeMismatch


def insert_csv_into_db(path_to_sqlite, path_to_csv, csv_batch_size=4096) -> int:
    """
    Load the timescale database SQL CSV dump into a sqlitedb.
    CSV should ideally contain:
    - time: string, time as per RFC 3339
    - host: string, indicates source of OPC signal
    - id: string, OPC node namespace/identifier
    - name: string, identifier for the OPC variable
    - Quality: string, e.g. "The operation succeeded. StatusGood (0x0)"
    - fields: string, jsonb e.g. '{"val": 123.45}'

    Only `time`, `id`, `name`, and `fields` columns are used.
    """
    insert_stmt = "INSERT OR REPLACE INTO opcua (time, id, name, fields) VALUES (?, ?, ?, ?);"

    with open(path_to_csv, "r") as f:
        reader = csv.DictReader(f)
        counter = 0
        with sqlite3.connect(path_to_sqlite) as conn:
            csv_batch = []
            for row in reader:
                csv_batch.append([row["time"], row["id"], row["name"], row["fields"]])
                if len(csv_batch) >= csv_batch_size:
                    conn.executemany(insert_stmt, csv_batch)
                    counter += len(csv_batch)
                    csv_batch = []
            conn.executemany(insert_stmt, csv_batch)
            counter += len(csv_batch)
            conn.commit()
    return counter


def initialize_db(path_to_sqlite):
    """
    Initialize a sqlite database with one table `opcua` and indicies on time and id
    """

    sqlite_setup_stmts = [
        """
        CREATE TABLE IF NOT EXISTS opcua (
            time DATE NOT NULL,
            id text NOT NULL,
            name text NOT NULL,
            fields JSONB NOT NULL,
            PRIMARY KEY (time, id)
        );
        """,
        "CREATE INDEX IF NOT EXISTS opcua_time ON opcua (time);",
        "CREATE INDEX IF NOT EXISTS opcua_id ON opcua (id);",
    ]
    with sqlite3.connect(path_to_sqlite) as conn:
        for setup_stmt in sqlite_setup_stmts:
            conn.execute(setup_stmt)


def get_opc_node_data(path_to_db):
    """
    Given db, return a dictionary containing opc sensor data
    id : (name, start_time, end_time, count)
    """
    opc_nodes = {}
    with sqlite3.connect(path_to_db) as conn:
        cur = conn.execute("SELECT id, name, min(time), max(time), count(*) FROM opcua GROUP BY id;")
        for row in cur:
            opc_nodes[row[0]] = {"name": row[1], "start_time": row[2], "end_time": row[3], "count": row[4]}
    # populate with last value, used to infer datatype
    stmt = "SELECT json_extract(fields, '$.val') FROM opcua WHERE id = ? ORDER BY time DESC LIMIT 1;"
    for k, v in opc_nodes.items():
        with sqlite3.connect(args.db) as conn:
            cur = conn.execute(stmt, [k])
            v["last_val"] = cur.fetchone()[0]

    return opc_nodes


async def init_opc_simulation_folder(client: Client, folder_bname="OPCSimulation"):
    folder_node_id = f"ns=0;s={folder_bname}"
    try:
        folder = await client.nodes.objects.add_folder(folder_node_id, folder_bname)
    except BadNodeIdExists:
        # folder already exists
        folder = client.get_node(folder_node_id)
    return folder


def get_temporal_equivalent_value(conn: sqlite3.Connection, id: str, time: str = "now"):
    """
    Query sqlite3 using nearest sensor value following hierarchy:
        match month, day of month, time,
        match day of month, time,
        match time,
    """

    time_masks = ["0000-%m-%d %H:%M:%S", "0000-01-%d %H:%M:%S", "0000-01-01 %H:%M:%S"]

    select_sensors_mask_stmt = """
    SELECT
        json_extract(fields, '$.val'),
        id,
        time,
        name,
        abs(julianday(strftime(?, datetime(?)))
            - julianday(strftime(?, datetime(time)))) as md_delta
    FROM opcua
    WHERE
        id = ?
        AND md_delta < 1
    ORDER BY md_delta ASC
    LIMIT 1;
    """

    result = None
    for time_mask in time_masks:
        cur = conn.execute(select_sensors_mask_stmt, (time_mask, time, time_mask, id))
        result = cur.fetchone()
        if result:
            break

    return result


async def main(args: Namespace):
    _logger = logging.getLogger(__name__)

    _logger.info(f"Initializing db {args.db}")
    initialize_db(args.db)

    if args.csv:
        _logger.info(f"Loading {args.csv} into db...")
        num_rows = insert_csv_into_db(args.db, args.csv)
        _logger.info(f"Processed {num_rows} rows")

    # determine source data bounds, sample latest value to determine data type
    opc_nodes = get_opc_node_data(args.db)
    if not opc_nodes:
        _logger.warning("No data exists within db, load data from csv first!")
        return

    _logger.info(f"OPC connecting to {args.url}")
    async with Client(args.url) as client:
        folder = await init_opc_simulation_folder(client)

        for id, data in opc_nodes.items():
            node = client.get_node(id)
            try:
                _ = await node.read_browse_name()
            except BadNodeIdUnknown:
                # node does not exist in OPC server, create it
                if isinstance(data["last_val"], Number):
                    # TODO: assume that all numbers are floats until fields contains "DataType"
                    # set by the opcua driver
                    data["last_val"] = float(data["last_val"])
                node = await folder.add_variable(id, data["name"], data["last_val"])

        _logger.info(f"Beginning simulation loop with {len(opc_nodes)} variables")

        with sqlite3.connect(args.db) as conn:
            while True:
                for id, data in opc_nodes.items():
                    node = client.get_node(id)
                    result = get_temporal_equivalent_value(conn, id)
                    if not result:
                        _logger.warning(f"No temporal consistent sensor data found for {id}")
                        continue
                    try:
                        if isinstance(result[0], Number):
                            # TODO: assume that all numbers are floats until fields contains "DataType"
                            # set by the opcua driver
                            await node.write_value(float(result[0]))
                        else:
                            await node.write_value(result[0])
                    except BadTypeMismatch:
                        _logger.warning(
                            f"{id}: {result[0]} {type(result[0])} | "
                            + f"initialized as {data["last_val"]} {type(data["last_val"])}"
                        )


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--db", help="Path to sqlite3 db", default="opcua.db")
    parser.add_argument("-c", "--csv", help="Path to timescaledb CSV dump", default=None)
    parser.add_argument(
        "-u",
        "--url",
        help="URL/endpoint of OPC UA server",
        default="opc.tcp://admin@0.0.0.0:4840/rlcore/server/",
        metavar="URL",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set log level",
    )

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel))
    asyncio.run(main(args), debug=True)

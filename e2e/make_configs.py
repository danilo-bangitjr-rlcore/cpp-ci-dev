import pandas as pd
import sqlite3
from pathlib import Path
import argparse
import shutil
import logging

from opc_clients.opc_client import insert_csv_into_db, initialize_db


def read_from_sqlite(sqlite_path: Path, query: str):
    conn = sqlite3.connect(sqlite_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def generate_telegraf_conf(path, df_distinct):
    _logger = logging.getLogger(__name__)
    shutil.copyfile(path / "telegraf/base_telegraf.conf", path / "telegraf/generated_telegraf.conf")
    block = ""
    with open(path / "telegraf/generated_telegraf.conf", "a") as f:
        for row in df_distinct.itertuples():
            block += " " * 2 + "[[inputs.opcua.nodes]]\n"
            block += " " * 4 + f'namespace = "{row.ns}"\n'
            block += " " * 4 + f'identifier_type = "{row.id_type}"\n'
            block += " " * 4 + f'identifier = "{row.id_name}"\n'
            block += " " * 4 + 'name = "val"\n'
            block += " " * 4 + f'default_tags = {{ name = "{row.name}" }}\n'
            block += "\n"
        f.write(block)

    _logger.info(f"Generetad {path}/telegraf/generated_telegraf.conf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--db", help="Path to sqlite3 db", default="opc_clients/opcua.db")
    parser.add_argument("-c", "--csv", help="Path to timescaledb CSV dump", default=None)
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set log level",
    )

    # Parsing input for S3
    current_path = Path(__file__).parent.absolute()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel))
    _logger = logging.getLogger(__name__)
    _logger.info(f"Initializing db {args.db}")
    initialize_db(current_path / args.db)

    if args.csv:
        _logger.info(f"Loading {args.csv} into db...")
        num_rows = insert_csv_into_db(current_path / args.db, current_path / args.csv)
        _logger.info(f"Processed {num_rows} rows")

    # SQLite Query
    query = "SELECT DISTINCT id, name FROM opcua;"
    df_distinct = read_from_sqlite(current_path / args.db, query)

    df_distinct["ns"] = df_distinct["id"].apply(lambda x: x.split(";")[0].split("=")[1])
    df_distinct["id_type"] = df_distinct["id"].apply(lambda x: x.split(";")[1].split("=")[0])
    df_distinct["id_name"] = df_distinct["id"].apply(lambda x: x.split(";")[1].split("=")[1])

    _logger.info(f"Found {len(df_distinct)} distinct nodes")

    generate_telegraf_conf(current_path, df_distinct)

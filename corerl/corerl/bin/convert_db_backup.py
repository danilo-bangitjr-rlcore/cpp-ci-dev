#!/usr/bin/env python3

import argparse
import csv
from itertools import islice
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from corerl.utils.opc_connection import make_opc_node_id


def convert_table(
        path_file: Path,
        host: str = "RLCore",
        ns: int = 2,
        quality: str = "The operation succeeded. StatusGood (0x0)",
        output: Path | str | None = None,
        row_offset: int = 4,
        column_offset: int = 4,
        datatype_row: int = 1,
        tag_name_row: int = 3,
        processed_columns: tuple[str, ...] = ("time", "host", "id", "name", "Quality", "fields")):

    with open(path_file, mode="r", newline="") as infile:
        reader = csv.reader(infile)
        datatypes = []
        columns = []
        for i, line in enumerate(reader):
            if i == datatype_row:
                datatypes = line
            elif i == tag_name_row:
                columns = line
            elif i > tag_name_row and i > datatype_row:
                break

    dict_schema = {}
    for column, datatype in islice(zip(columns, datatypes, strict=True), column_offset, None):
        dict_schema[column] = datatype.capitalize()
    tag_names = dict_schema.keys()

    if output is None:
        out_file = Path(*path_file.parts[:-1]) / Path(f"proc_{path_file.parts[-1]}")
    else:
        out_file = Path(output)

    # Quick first pass through file to get the total number of lines for tqdm
    with open(path_file, mode="r", newline="") as infile:
        reader = csv.reader(infile)
        row_count = sum(1 for row in islice(reader, row_offset, None))

    with open(path_file, mode="r", newline="") as infile, \
         open(out_file, mode="w", newline="") as outfile:

        print(f"Reading from {path_file}")
        reader = csv.reader(infile)

        print(f"Writing to {out_file}")
        writer = csv.writer(outfile)

        writer.writerow(processed_columns)

        for line in tqdm(islice(reader, row_offset, None), total=row_count):
            time = line[3]
            for tag, val in zip(tag_names, islice(line, column_offset, None), strict=True):
                dtype = dict_schema[tag]
                if dtype == "Boolean":
                    val = val.lower() == "true"
                elif dtype == "Integer":
                    val = int(pd.to_numeric(val))
                else:
                    val = float(pd.to_numeric(val))

                writer.writerow([time,
                                 host,
                                 make_opc_node_id(tag, ns),
                                 tag,
                                 quality,
                                 {"val": val, "DataType": dtype}])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Name of the input csv file", type=str)
    parser.add_argument("-s", "--host", help="Name of the host OPC server", default="RLCore")
    parser.add_argument("-n", "--namespace", help="Number of the OPC namespace", default=2, type=int)
    parser.add_argument("-q", "--quality", help="Message when quality of data is good",
                        default="The operation succeeded. StatusGood (0x0)")
    parser.add_argument("-o", "--output",
                        help="Name for the output file. If none is given output will be `proc_[name_of_input_file]`")
    parser.add_argument("--row_offset", help="Row where the data begins (0-indexed)", default=4, type=int)
    parser.add_argument("--column_offset", help="Column where the data begins (0-indexed)", default=4, type=int)
    parser.add_argument("--datatype_row", help="Row where the datatypes are defined (0-indexed)", default=1, type=int)
    parser.add_argument("--tag_name_row", help="Row where the tag names are defined (0-indexed)", default=3, type=int)
    args = parser.parse_args()

    convert_table(
        path_file=Path(args.file),
        host=args.host,
        ns=args.namespace,
        quality=args.quality,
        output=args.output,
        row_offset=args.row_offset,
        column_offset=args.column_offset,
        datatype_row=args.datatype_row,
        tag_name_row=args.tag_name_row,)


if __name__ == "__main__":
    main()

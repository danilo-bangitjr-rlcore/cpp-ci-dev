import pandas as pd
import argparse

def create_table_from_tsv(file_path: str, table_name: str):
    """
    Reads a tab-delimited file and generates a CREATE TABLE SQL statement.

    Args:
        file_path (str): The path to the tab-delimited file.
        table_name (str): The name of the table to create.

    Returns:
        str: The CREATE TABLE SQL statement.
    """
    try:
        df = pd.read_csv(file_path, sep='\t', names=['tag', 'dtype']) 
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except Exception as e:
        return f"An error occurred: {e}"

    column_definitions = [
        "\"time\" timestamp with time zone NOT NULL"
    ]
    for _, row in df.iterrows():
        dtype = row['dtype']
        if dtype == 'number':
            dtype = 'real'
        column_definitions.append(f"\"{row['tag'].lower()}\" {dtype}") #use backticks for column names.

    create_table_sql = f"CREATE TABLE public.{table_name} (\n" + ",\n".join(column_definitions) + "\n);"

    return create_table_sql

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Path to input file'
    )

    parser.add_argument(
        '--table-name',
        type=str,
        default='scrubber4_wide',
        help='Name of the table to create'
    )

    args = parser.parse_args()

    print(create_table_from_tsv(args.input_file, table_name=args.table_name))

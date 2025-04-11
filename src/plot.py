import argparse
import os
import sqlite3

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--results", type=str, required=True)

args = parser.parse_args()

def read_meta_data(cursor: sqlite3.Cursor) -> dict:
    """
    Read metadata from the SQLite database.
    """
    table_name = '_metadata_'
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()

    result_dict = {}
    for row in rows:
        row_dict = {key: row[key] for key in row.keys()}
        row_id = row_dict.pop('id')
        result_dict[row_id] = row_dict
    return result_dict


def ids_for_value(key: str, value: str, metadata: dict) -> list:
    ids = []
    for id, row in metadata.items():
        if key in row and row[key] == value:
            ids.append(id)
    return ids

def unique_vals(key: str, metadata: dict) -> list:
    vals = []
    for _, row in metadata.items():
        if key in row and row[key] not in vals:
            vals.append(row[key])
    return vals


def get_rows_by_ids(cursor: sqlite3.Cursor, table_name: str, id_list: list[int]):
    # Convert id_list to tuple for SQL query
    # Use placeholders to prevent SQL injection
    placeholders = ','.join(['?'] * len(id_list))

    # Execute query filtering by IDs
    query = f"SELECT * FROM {table_name} WHERE id IN ({placeholders})"

    try:
        cursor.execute(query, id_list)
        results = cursor.fetchall()

        # Convert sqlite3.Row objects to dictionaries
        rows = [dict(row) for row in results]

        return rows

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return []

def group_by_id(rows: list[dict]):
    grouped = dict()
    for row in rows:
        if row['id'] not in grouped:
            grouped[row['id']] = []
        grouped[row['id']].append(row)
    return grouped

def plot_measurment_by_frame(
        grouped_rows: dict[int, list[dict]],
        ax: Axes,
        title: str,
    ):

    for _, group in grouped_rows.items():
        xs = [row['frame'] for row in group]
        ys = [row['measurement'] for row in group]

        ax.plot(xs, ys, color='blue', alpha=0.2)
        ax.set_title(title)


def main():
    conn = sqlite3.connect(args.results)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    metadata = read_meta_data(cursor)

    envs = unique_vals('env.name', metadata)
    fig, axs = plt.subplots(len(envs), 1, figsize=(10, 5 * len(envs)))
    for env_idx, env in enumerate(envs):
        ids = ids_for_value('env.name', env, metadata)
        rows = get_rows_by_ids(cursor, 'reward', ids)
        grouped_rows = group_by_id(rows)
        plot_measurment_by_frame(grouped_rows, ax=axs[env_idx], title=env)

    result_path = "/".join(args.results.split("/")[:-1])
    save_path = os.path.join(result_path, 'reward.png')
    plt.savefig(save_path)

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()

import argparse
import os
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--results", type=str, required=True)

args = parser.parse_args()

def plot_across(
        metadata: pd.DataFrame,
        data: pd.DataFrame,
        key: str,
        save_path: str,
    ):
    results = pd.merge(data, metadata, on='id') # joins metadata with data
    grouped_by_key = results.groupby(key)

    group_size = len(grouped_by_key)
    _, axs = plt.subplots(group_size, 1, figsize=(10, 5*group_size))

    ax_idx = 0
    for key_name, group_data in grouped_by_key:
        grouped_by_id = group_data.groupby('id') # iterate through runs
        for _, id_group_data in grouped_by_id:
            xs = id_group_data['frame']
            ys = id_group_data['measurement']

            axs[ax_idx].plot(xs, ys, color='blue', alpha=0.2)
            axs[ax_idx].set_title(key_name)

        ax_idx += 1

    plt.savefig(save_path)

def read_table(table_name: str):
    conn = sqlite3.connect(args.results)
    df = pd.read_sql(f'SELECT * FROM {table_name}', conn)
    conn.close()
    return df


def main():
    metadata = read_table('_metadata_')

    # REWARD
    result_path = "/".join(args.results.split("/")[:-1])
    save_path = os.path.join(result_path, 'reward.png')
    rewards = read_table('reward')
    plot_across(metadata, rewards, 'env.name', save_path)


    # CRITIC LOSS
    save_path = os.path.join(result_path, 'critic_loss.png')
    rewards = read_table('critic_loss')
    plot_across(metadata, rewards, 'env.name', save_path)


if __name__ == "__main__":
    main()

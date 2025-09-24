import datetime
import logging

import polars as pl
from corerl.data_pipeline.db.data_writer import DataWriter
from lib_config.loader import load_config

from coreoffline.config import LoadDataConfig

log = logging.getLogger(__name__)


def load_dataset(cfg: LoadDataConfig):
    """Loads dataset from CSV file and ingests into database based on config."""
    to_ingest = cfg.reward_tags + cfg.action_tags + cfg.input_tags
    writer = DataWriter(cfg.data_writer)
    df = pl.read_csv(cfg.csv_path, null_values=['', 'null', 'None', 'inf'])

    n_cols = len(df.columns)
    for i, col in enumerate(df.columns):
        if col in {'Date', ''}:
            continue

        if col not in to_ingest:
            continue

        print(f'Processing column: {col} ({i} / {n_cols} total columns)')
        for row in df.select(['Date', col]).iter_rows():
            t_str, v = row
            t = datetime.datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S')
            t = t.replace(tzinfo=datetime.UTC)
            tag = col
            if v is None:
                continue

            try:
                writer.write(t, tag, float(v))

            except Exception as exc:
                raise ValueError(f'Tried to cast {v} to float for column {col} but failed.') from exc

    writer.close()


@load_config(LoadDataConfig)
def main(cfg: LoadDataConfig):
    """Main function that loads data using the configuration."""
    print(f"Loading data from {cfg.csv_path}")

    load_dataset(cfg)
    print("Data loading completed successfully!")


if __name__ == '__main__':
    main()

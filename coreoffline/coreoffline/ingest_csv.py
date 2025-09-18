import datetime
from logging import getLogger
from pathlib import Path

import polars as pl
from corerl.data_pipeline.db.data_writer import DataWriter, TagDBConfig
from lib_config.config import MISSING, config, post_processor
from lib_config.loader import load_config
from pydantic import Field

logger = getLogger('cenovus')


@config()
class LoadDataConfig:
    # CSV file configuration
    csv_path: Path = MISSING

    # Tag configuration
    reward_tags: list[str] = Field(default_factory=list)
    action_tags: list[str] = Field(default_factory=list)
    input_tags: list[str] = Field(default_factory=list)

    # Database configuration
    data_writer: TagDBConfig = Field(default_factory=TagDBConfig)

    @post_processor
    def _validate(self, cfg: 'LoadDataConfig'):
        required_tags = ['reward_tags', 'action_tags', 'input_tags']
        for tag_type in required_tags:
            tags = getattr(self, tag_type)
            if not isinstance(tags, list):
                raise ValueError(f"{tag_type} must be a list")


def load_dataset(cfg: LoadDataConfig):
    """Loads dataset from CSV file and ingests into database based on config."""
    to_ingest = cfg.reward_tags + cfg.action_tags + cfg.input_tags
    writer = DataWriter(cfg.data_writer)
    df = pl.read_csv(cfg.csv_path, null_values=['', 'null', 'None', 'inf'])

    n_cols = len(df.columns)
    for i, col in enumerate(reversed(df.columns)):
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

            writer.write(t, tag, float(v))

    writer.close()


@load_config(LoadDataConfig)
def main(cfg: LoadDataConfig):
    """Main function that loads data using the configuration."""
    print(f"Loading data from {cfg.csv_path}")

    load_dataset(cfg)
    print("Data loading completed successfully!")


if __name__ == '__main__':
    main()

from typing import Any

import utils.data as utils

from corerl.data_pipeline.db.data_writer import DataWriter, TagDBConfig


def _load_dataset_from_s3() -> list[tuple[Any, ...]]:
    """
    Read csv files from s3, preprocess the data, and convert into (timestamp, tag, value) tuples
    """
    offline_files = utils.get_s3_files()
    columns = utils.load_csv_files(offline_files)
    utils.get_min_max(columns)
    sql_tups = utils.columns_to_sql_tups(columns)

    return sql_tups

def load_dataset(cfg: TagDBConfig):
    sql_tups = _load_dataset_from_s3()

    writer = DataWriter(
        cfg,
        low_watermark=1_000,
        high_watermark=10_000,
    )

    for sql_tup in sql_tups:
        writer.write(timestamp=sql_tup[0], name=sql_tup[2], val=sql_tup[1])

    writer.close()

if __name__ == '__main__':
    cfg = TagDBConfig(
        table_name='epcor_solar',
    )
    load_dataset(cfg)

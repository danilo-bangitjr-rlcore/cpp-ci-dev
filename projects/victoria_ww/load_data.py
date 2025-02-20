from typing import Any

import utils.data as utils

from corerl.configs.loader import load_config
from corerl.data_pipeline.db.data_writer import DataWriter, TagDBConfig


def _load_dataset_from_s3(dl_cfg: utils.VictoriaWWConfig) -> list[tuple[Any, ...]]:
    """
    Read csv files from s3, preprocess the data, and convert into (timestamp, tag, value) tuples
    """
    offline_files = utils.get_s3_files()
    columns = utils.load_excel_files(offline_files)
    utils.get_min_max(columns)
    last_timestamp = utils.get_last_timestamp(columns)
    sql_tups = utils.get_sql_tups(dl_cfg, columns, last_timestamp)

    return sql_tups

@load_config(utils.VictoriaWWConfig, base='projects/victoria_ww/configs', config_name='dl')
def load_dataset(dl_cfg: utils.VictoriaWWConfig, db_cfg: TagDBConfig):
    sql_tups = _load_dataset_from_s3(dl_cfg)

    writer = DataWriter(
        db_cfg,
        low_watermark=1_000,
        high_watermark=10_000,
    )

    for sql_tup in sql_tups:
        writer.write(timestamp=sql_tup[0], name=sql_tup[2], val=sql_tup[1])

    writer.close()

if __name__ == '__main__':
    db_cfg = TagDBConfig(
        table_name='victoria_ww',
    )
    load_dataset(db_cfg)

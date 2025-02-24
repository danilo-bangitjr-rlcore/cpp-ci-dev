import utils.data as utils

from corerl.config import DBConfig
from corerl.configs.loader import load_config
from corerl.data_pipeline.db.data_writer import DataWriter, TagDBConfig


def _load_dataset_from_s3(dl_cfg: utils.VictoriaWWConfig) -> list[utils.SQLEntry]:
    """
    Read excel files from s3, preprocess the data, and convert into (timestamp, tag, value) tuples
    """
    offline_files = utils.get_s3_files()
    columns = utils.load_excel_files(offline_files)
    last_timestamp = utils.get_last_timestamp(columns)
    sql_tups = utils.get_sql_tups(dl_cfg, columns, last_timestamp)

    return sql_tups

@load_config(utils.VictoriaWWConfig, base='projects/victoria_ww/configs', config_name='dl')
def load_dataset(dl_cfg: utils.VictoriaWWConfig, db_cfg: TagDBConfig):
    """
    Read offline data from s3 and write (timestamp, tag, value) tuples to TSDB
    """
    sql_tups = _load_dataset_from_s3(dl_cfg)

    writer = DataWriter(
        db_cfg,
        low_watermark=1_000,
        high_watermark=10_000,
    )

    for sql_tup in sql_tups:
        writer.write(timestamp=sql_tup.time, name=sql_tup.tag, val=sql_tup.val)

    writer.close()

if __name__ == '__main__':
    db_cfg = TagDBConfig(
        drivername=DBConfig.drivername,
        username=DBConfig.username,
        password=DBConfig.password,
        ip=DBConfig.ip,
        port=DBConfig.port,
        db_name='victoria_ww',
        table_name='tag_data',
    )
    load_dataset(db_cfg)

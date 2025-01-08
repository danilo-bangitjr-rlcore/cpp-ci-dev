from logging import getLogger
from cenovus.utils.data import get_file_paths, read_json, ReadFailure
from corerl.data_pipeline.db.data_writer import DataWriter, TagDBConfig


logger = getLogger('cenovus')


def _load_dataset_from_s3():
    for f in get_file_paths():
        raw = f.read_bytes()
        json = read_json(raw)

        if json is None:
            logger.warning(f'Failed to read {f.name}')
        else:
            yield f, json



def load_dataset(cfg: TagDBConfig):
    writer = DataWriter(
        cfg,
        low_watermark=1_000_000,
        high_watermark=10_000_000,
    )
    json_files = _load_dataset_from_s3()

    for f, tag_data in json_files:
        tag = f.name.removesuffix('.json')
        rows = zip(
            tag_data.data.timestamps,
            tag_data.data.values,
            strict=True,
        )
        for row in rows:
            t, v = row
            if isinstance(v, ReadFailure):
                continue

            writer.write(t, tag, v)

    writer.close()


if __name__ == '__main__':
    cfg = TagDBConfig(
        sensor_table_name='cenovus',
    )
    load_dataset(cfg)

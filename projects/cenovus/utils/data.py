from logging import getLogger
from pathlib import Path
from datetime import datetime
from cloudpathlib.enums import FileCacheMode
from pydantic import BaseModel, ValidationError
from cloudpathlib import S3Client, S3Path

logger = getLogger('cenovus')


class ReadFailure(BaseModel):
    name: str
    value: int
    is_system: bool

class RawTagTimeseries(BaseModel):
    timestamps: list[datetime]
    values: list[float | bool | int | ReadFailure]

class TagDatafile(BaseModel):
    data: RawTagTimeseries


def read_json(raw: str | bytes):
    try:
        return TagDatafile.model_validate_json(raw)

    except ValidationError:
        logger.exception('Validation error')
        return None

def get_file_paths():
    cache = Path('projects/cenovus/.cache')
    cache.mkdir(parents=True, exist_ok=True)
    client = S3Client(
        file_cache_mode=FileCacheMode.persistent,
        local_cache_dir=cache,
    )
    root = S3Path('s3://rlcore-shared/', client)
    return root.glob('Cenovus/**/*.json')

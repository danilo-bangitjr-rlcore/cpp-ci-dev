import random
from typing import Any
from sqlalchemy import Engine, Connection
import time
import logging

logger = logging.getLogger(__name__)


def try_connect(engine: Engine, backoff_seconds: int = 5, max_tries: int = 5) -> Connection:
    connection = None
    tries = 0
    while not connection is not None:
        if tries >= max_tries:
            raise Exception("sql engine connection failed")
        try:
            connection = engine.connect()
        except:
            logger.warning(f"failed to connect sql engine, retrying in {backoff_seconds} seconds...")
            time.sleep(backoff_seconds)
        tries += 1

    return connection


def train_test_split(*lsts, train_split: float = 0.9, shuffle: bool = True) -> list[tuple[Any, Any]]:
    num_samples = len(lsts[0])
    for a in lsts:
        assert len(a) == num_samples

    if shuffle:
        lsts = parallel_shuffle(*lsts)

    num_train_samples = int(train_split * num_samples)
    train_samples = [lsts[:num_train_samples] for lsts in lsts]
    test_samples = [lsts[num_train_samples:] for lsts in lsts]

    return list(zip(train_samples, test_samples, strict=True))


def parallel_shuffle(*args):
    zipped_list = list(zip(*args, strict=True))
    random.shuffle(zipped_list)
    unzipped = zip(*zipped_list, strict=True)
    return list(unzipped)

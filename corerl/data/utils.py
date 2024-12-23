import random
import logging

logger = logging.getLogger(__name__)


def train_test_split[T](
        *lsts: list[T],
        train_split: float = 0.9,
        shuffle: bool = True,
) -> list[tuple[list[T], list[T]]]:
    num_samples = len(lsts[0])
    for a in lsts:
        assert len(a) == num_samples

    if shuffle:
        lsts = parallel_shuffle(*lsts)

    num_train_samples = int(train_split * num_samples)
    train_samples = [lsts[:num_train_samples] for lsts in lsts]
    test_samples = [lsts[num_train_samples:] for lsts in lsts]

    return list(zip(train_samples, test_samples, strict=True))


def parallel_shuffle[T](*args: T) -> tuple[T, ...]:
    zipped_list = list(zip(*args, strict=True))
    random.shuffle(zipped_list)
    unzipped = zip(*zipped_list, strict=True)
    return tuple(unzipped)

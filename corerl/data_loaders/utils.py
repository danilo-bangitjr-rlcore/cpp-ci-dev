import random


def train_test_split(*lsts, train_split: float = 0.9, shuffle: bool = True) -> tuple[list[tuple], list[tuple]]:
    num_samples = len(lsts[0])
    for a in lsts:
        assert len(a) == num_samples

    if shuffle:
        lsts = parallel_shuffle(*lsts)

    num_train_samples = int(train_split * num_samples)
    train_samples = [lsts[:num_train_samples] for lsts in lsts]
    test_samples = [lsts[num_train_samples:] for lsts in lsts]

    return list(zip(train_samples, test_samples))


def parallel_shuffle(*args):
    zipped_list = list(zip(*args))
    random.shuffle(zipped_list)
    unzipped = zip(*zipped_list)
    return list(unzipped)

from datetime import datetime, timedelta

import corerl.utils.time as time_util


def test_split_into_chunks():
    start = datetime(2024, 8, 1, 1)
    end = datetime(2024, 8, 1, 5, 30)
    Δ = timedelta(hours=1)

    chunks = time_util.split_into_chunks(start, end, Δ)

    assert list(chunks) == [
        (datetime(2024, 8, 1, 1), datetime(2024, 8, 1, 2)),
        (datetime(2024, 8, 1, 2), datetime(2024, 8, 1, 3)),
        (datetime(2024, 8, 1, 3), datetime(2024, 8, 1, 4)),
        (datetime(2024, 8, 1, 4), datetime(2024, 8, 1, 5)),
        (datetime(2024, 8, 1, 5), datetime(2024, 8, 1, 5, 30)),
    ]

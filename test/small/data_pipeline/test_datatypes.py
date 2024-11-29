import pandas as pd
from datetime import datetime, timedelta, UTC
from corerl.data_pipeline.datatypes import PipelineFrame, CallerCode

def test_get_last_timestamp():
    now = datetime.now(UTC)
    delta = timedelta(minutes=5)
    df = pd.DataFrame({
        'tag-1': [0., 0.1, 0.2],
    })
    df = df.set_index(pd.Series([now, now + delta, now + 2*delta]))
    pf = PipelineFrame(
        data=df,
        caller_code=CallerCode.ONLINE
    )

    last_ts = pf.get_last_timestamp()
    assert last_ts is not None
    assert last_ts == (now + 2*delta)


def test_get_first_timestamp():
    now = datetime.now(UTC)
    delta = timedelta(minutes=5)
    df = pd.DataFrame({
        'tag-1': [0., 0.1, 0.2],
    })
    df = df.set_index(pd.Series([now, now + delta, now + 2*delta]))
    pf = PipelineFrame(
        data=df,
        caller_code=CallerCode.ONLINE
    )

    first_ts = pf.get_first_timestamp()
    assert first_ts is not None
    assert first_ts == now

import numpy as np
import pandas as pd
from corerl.data_pipeline.datatypes import CallerCode, PipelineFrame
from corerl.data_pipeline.state_constructors.countdown import DecisionPointDetector, CountdownConfig
from corerl.data_pipeline.tag_config import TagConfig
from test.infrastructure.utils.pandas import dfs_close

def test_int_countdown1():
    actions = pd.DataFrame({
        'tag-1': [1, 1, 1, 1, 2, 2, 2],
        'tag-2': [0, 0, 0, 0, 0, 0, 0],
    })

    pf = PipelineFrame(
        data=actions,
        caller_code=CallerCode.ONLINE,
    )

    tag_cfgs = [
        TagConfig('tag-1', is_action=True),
        TagConfig('tag-2', is_action=True),
    ]

    cd_cfg = CountdownConfig(
        action_period=4,
        kind='int',
    )

    cd_adder = DecisionPointDetector(tag_cfgs, cd_cfg)
    pf = cd_adder(pf)

    expected = pd.DataFrame({
        'tag-1':         [1, 1, 1, 1, 2, 2, 2],
        'tag-2':         [0, 0, 0, 0, 0, 0, 0],
        'countdown.[0]': [4, 3, 2, 1, 4, 3, 2],
    })
    assert dfs_close(pf.data, expected)
    assert np.all(pf.decision_points == [1, 0, 0, 0, 1, 0, 0])


    # -----------------
    # -- Second pass --
    # -----------------
    # Ensure that the carry-over temporal state
    # keeps the correct clock from the last df
    actions = pd.DataFrame({
        'tag-1': [2, 3, 3, 3, 3],
        'tag-2': [0, 0, 0, 0, 0],
    })

    pf = PipelineFrame(
        data=actions,
        caller_code=CallerCode.ONLINE,
        temporal_state=pf.temporal_state,
    )

    pf = cd_adder(pf)
    expected = pd.DataFrame({
        'tag-1':         [2, 3, 3, 3, 3],
        'tag-2':         [0, 0, 0, 0, 0],
        'countdown.[0]': [1, 4, 3, 2, 1],
    })
    assert dfs_close(pf.data, expected)
    assert np.all(pf.decision_points == [0, 1, 0, 0, 0])


def test_int_countdown2():
    """
    When an action changes unexpectedly, the count
    just starts over.
    """

    actions = pd.DataFrame({
        'tag-1': [1, 1, 1, 1, 2, 2, 2, 2],
        'tag-2': [0, 0, 1, 1, 1, 1, 0, 0],
    })

    pf = PipelineFrame(
        data=actions,
        caller_code=CallerCode.ONLINE,
    )

    tag_cfgs = [
        TagConfig('tag-1', is_action=True),
        TagConfig('tag-2', is_action=True),
    ]

    cd_cfg = CountdownConfig(
        action_period=4,
        kind='int',
    )

    cd_adder = DecisionPointDetector(tag_cfgs, cd_cfg)
    pf = cd_adder(pf)

    expected = pd.DataFrame({
        'tag-1':         [1, 1, 1, 1, 2, 2, 2, 2],
        'tag-2':         [0, 0, 1, 1, 1, 1, 0, 0],
        'countdown.[0]': [2, 1, 4, 3, 4, 3, 4, 3],
    })
    assert dfs_close(pf.data, expected)
    assert np.all(pf.decision_points == [0, 0, 1, 0, 1, 0, 1, 0])


def test_int_countdown3():
    """
    When an action period ends, but the same
    action is still selected, the countdown ends.
    """

    actions = pd.DataFrame({
        'tag-1': [1, 1, 1, 1, 1, 1, 1, 1],
    })

    pf = PipelineFrame(
        data=actions,
        caller_code=CallerCode.ONLINE,
    )

    tag_cfgs = [
        TagConfig('tag-1', is_action=True),
    ]

    cd_cfg = CountdownConfig(
        action_period=4,
        kind='int',
    )

    cd_adder = DecisionPointDetector(tag_cfgs, cd_cfg)
    pf = cd_adder(pf)

    expected = pd.DataFrame({
        'tag-1':         [1, 1, 1, 1, 1, 1, 1, 1],
        'countdown.[0]': [4, 3, 2, 1, 4, 3, 2, 1],
    })
    assert dfs_close(pf.data, expected)
    assert np.all(pf.decision_points == [1, 0, 0, 0, 1, 0, 0, 0])


def test_two_clock1():
    actions = pd.DataFrame({
        'tag-1': [1, 1, 2, 2, 2, 2, 3, 3],
    })

    pf = PipelineFrame(
        data=actions,
        caller_code=CallerCode.ONLINE,
    )

    tag_cfgs = [
        TagConfig('tag-1', is_action=True),
    ]

    cd_cfg = CountdownConfig(
        action_period=4,
        kind='two_clock',
    )

    cd_adder = DecisionPointDetector(tag_cfgs, cd_cfg)
    pf = cd_adder(pf)

    expected = pd.DataFrame({
        'tag-1':         [1, 1, 2, 2, 2, 2, 3, 3],
        'countdown.[0]': [2, 1, 4, 3, 2, 1, 4, 3],
        'countdown.[1]': [2, 1, 4, 3, 2, 1, 4, 3],
    })
    assert dfs_close(pf.data, expected)
    assert np.all(pf.decision_points == [0, 0, 1, 0, 0, 0, 1, 0])

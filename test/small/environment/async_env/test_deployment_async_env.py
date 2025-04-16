from pandas import DataFrame

from corerl.data_pipeline.tag_config import TagConfig, TagType
from corerl.data_pipeline.transforms import NullConfig
from corerl.environment.async_env.deployment_async_env import clip_action, sanitize_actions
from test.infrastructure.utils.pandas import dfs_close


def test_action_clipping():
    tag_cfgs = {
        "action-1": TagConfig(
            name="action-1",
            type=TagType.ai_setpoint,
            operating_range=(5.0, 15.0),
            red_bounds=(None, 12),
            preprocess=[],
            state_constructor=[NullConfig()],
        ),
        "action-2": TagConfig(
            name="action-2",
            type=TagType.ai_setpoint,
            operating_range=(0.0, 60.0),
            preprocess=[],
            state_constructor=[NullConfig()],
        ),
    }

    actions = DataFrame({"action-1": [2.5], "action-2": [30]})
    expected = DataFrame({"action-1": [5], "action-2": [30]})
    clip_action(actions, tag_cfgs)
    assert dfs_close(actions, expected)

    actions = DataFrame({"action-1": [30], "action-2": [30]})
    expected = DataFrame({"action-1": [12], "action-2": [30]}) # clip to red zone
    clip_action(actions, tag_cfgs)
    assert dfs_close(actions, expected)

    actions = DataFrame({"action-1": [10], "action-2": [-3]})
    expected = DataFrame({"action-1": [10], "action-2": [0]})
    clip_action(actions, tag_cfgs)
    assert dfs_close(actions, expected)

    actions = DataFrame({"action-1": [10], "action-2": [90]})
    expected = DataFrame({"action-1": [10], "action-2": [60]})
    clip_action(actions, tag_cfgs)
    assert dfs_close(actions, expected)

    actions = DataFrame({"action-1": [10], "action-2": [-3]})
    expected = DataFrame({"action-1": [10], "action-2": [0]})
    clip_action(actions, tag_cfgs)
    assert dfs_close(actions, expected)

    actions = DataFrame({"action-1": [30], "action-2": [90]})
    expected = DataFrame({"action-1": [12], "action-2": [60]})
    clip_action(actions, tag_cfgs)
    assert dfs_close(actions, expected)

    actions = DataFrame({"action-1": [3], "action-2": [-13]})
    expected = DataFrame({"action-1": [5], "action-2": [0]})
    clip_action(actions, tag_cfgs)
    assert dfs_close(actions, expected)

def test_action_sanitizer():
    tag_cfgs = {
        "action-1": TagConfig(
            name="action-1",
            type=TagType.ai_setpoint,
            operating_range=(5.0, 15.0),
            red_bounds=(None, 12),
            preprocess=[],
            state_constructor=[NullConfig()],
        ),
        "action-2": TagConfig(
            name="action-2",
            type=TagType.ai_setpoint,
            operating_range=(0.0, 60.0),
            preprocess=[],
            state_constructor=[NullConfig()],
        ),
    }

    actions = DataFrame({"action-1": [30], "action-2": [30]})
    expected = DataFrame({"action-1": [12], "action-2": [30]}) # clip to red zone
    sanitize_actions(actions, tag_cfgs)
    assert dfs_close(actions, expected)

    # If there are too many actions, it's not clear what to do. Nuke the action
    actions = DataFrame({"action-1": [30, 12], "action-2": [30, 2]})
    expected = DataFrame({})
    sanitize_actions(actions, tag_cfgs)
    assert dfs_close(actions, expected)

    actions = DataFrame({})
    expected = DataFrame({})
    sanitize_actions(actions, tag_cfgs)
    assert dfs_close(actions, expected)

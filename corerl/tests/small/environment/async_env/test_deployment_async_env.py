import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from coreio.utils.io_events import IOEvent, IOEventType, OPCUANodeWriteValue
from pandas import DataFrame
from test.infrastructure.utils.pandas import dfs_close

from corerl.config import MainConfig
from corerl.environment.async_env.deployment_async_env import DeploymentAsyncEnv, clip_action, sanitize_actions
from corerl.tags.setpoint import SetpointTagConfig
from tests.infrastructure.config import create_config_with_overrides

# -------------------------
# -- Single action tests --
# -------------------------

@pytest.fixture
def mock_socket(monkeypatch: pytest.MonkeyPatch):
    socket = MagicMock()
    ctx = MagicMock()
    ctx.socket.return_value = socket
    monkeypatch.setattr("corerl.utils.coreio.zmq.Context", lambda: ctx)
    return socket

@pytest.fixture
def single_tag_minimal_cfg():
    config_path = os.path.join(
        os.path.dirname(__file__),
        'test_deployment_async_env_single_tag.yaml',
    )
    return create_config_with_overrides(base_config_path=config_path)

@pytest.fixture
def multi_tag_minimal_cfg():
    config_path = os.path.join(
        os.path.dirname(__file__),
        'test_deployment_async_env_multi_tag.yaml',
    )
    return create_config_with_overrides(base_config_path=config_path)

@pytest.fixture
def deployment_env(
    single_tag_minimal_cfg: MainConfig,
):
    with patch("corerl.environment.async_env.deployment_async_env.DataReader", autospec=True):
        return DeploymentAsyncEnv(single_tag_minimal_cfg.env, single_tag_minimal_cfg.pipeline.tags)


def test_emit_action_emits_correct_event(mock_socket: MagicMock, deployment_env: DeploymentAsyncEnv):
    """
    Emits correct event for a single action via zmq.
    """
    df = pd.DataFrame({"action-1": [5.0]})
    deployment_env.emit_action(df, log_action=True)

    sent: str = mock_socket.send_string.call_args[0][0]
    topic, json_str = sent.split(' ', 1)
    actual_event = IOEvent.model_validate_json(json_str)

    expected_event = IOEvent(
        type=IOEventType.write_to_opc,
        data={
            "conn-1": [
                OPCUANodeWriteValue(node_id="ns=2;i=1", value=5.0),
            ],
        },
        id=actual_event.id,
        time=actual_event.time,
    )

    assert topic == "coreio"
    assert actual_event == expected_event


# ---------------------------
# -- Multiple action tests --
# ---------------------------
@pytest.fixture
def multi_deployment_env(multi_tag_minimal_cfg: MainConfig):
    with patch("corerl.environment.async_env.deployment_async_env.DataReader", autospec=True):
        return DeploymentAsyncEnv(multi_tag_minimal_cfg.env, multi_tag_minimal_cfg.pipeline.tags)


def test_emit_action_multiple_actions(mock_socket: MagicMock, multi_deployment_env: DeploymentAsyncEnv):
    """
    Emits correct event for multiple actions.
    """
    df = pd.DataFrame({"action-1": [3.0], "action-2": [7.0]})
    multi_deployment_env.emit_action(df, log_action=True)
    sent: str = mock_socket.send_string.call_args[0][0]
    topic, json_str = sent.split(' ', 1)
    actual_event = IOEvent.model_validate_json(json_str)
    expected_event = IOEvent(
        type=IOEventType.write_to_opc,
        data={
            "conn-1": [OPCUANodeWriteValue(node_id="ns=2;i=1", value=3.0)],
            "conn-2": [OPCUANodeWriteValue(node_id="ns=2;i=2", value=7.0)],
        },
        id=actual_event.id,
        time=actual_event.time,
    )
    assert topic == "coreio"
    assert actual_event == expected_event

def test_emit_action_clips_out_of_range(mock_socket: MagicMock, multi_deployment_env: DeploymentAsyncEnv):
    """
    Clips out-of-range actions to within allowed bounds.
    """
    df = pd.DataFrame({"action-1": [-5.0], "action-2": [100.0]})
    multi_deployment_env.emit_action(df, log_action=True)
    sent: str = mock_socket.send_string.call_args[0][0]
    topic, json_str = sent.split(' ', 1)
    actual_event = IOEvent.model_validate_json(json_str)

    assert topic == "coreio"
    assert actual_event.data["conn-1"][0].value == pytest.approx(0.008, abs=1e-6)
    assert actual_event.data["conn-2"][0].value == pytest.approx(14.99, abs=1e-6)

def test_emit_action_unknown_action_raises(deployment_env: DeploymentAsyncEnv):
    """
    Raises KeyError if an unknown action is provided.
    """
    df = pd.DataFrame({"unknown-action": [1.0]})
    with pytest.raises(KeyError):
        deployment_env.emit_action(df, log_action=True)


# -------------------
# -- Utility tests --
# -------------------

def test_action_clipping(multi_tag_minimal_cfg: MainConfig):
    """
    Clips actions to operating or red zone bounds as expected.
    """
    tag_cfg_1 = multi_tag_minimal_cfg.pipeline.tags[0]
    tag_cfg_2 = multi_tag_minimal_cfg.pipeline.tags[1]

    assert isinstance(tag_cfg_1, SetpointTagConfig)
    assert isinstance(tag_cfg_2, SetpointTagConfig)

    tag_cfgs = {
        "action-1": tag_cfg_1,
        "action-2": tag_cfg_2,
    }

    actions = DataFrame({"action-1": [-2.5], "action-2": [30]})
    expected = DataFrame({"action-1": [0], "action-2": [15]})
    clip_action(actions, tag_cfgs, rtol=0)
    assert dfs_close(actions, expected)

    actions = DataFrame({"action-1": [30], "action-2": [30]})
    expected = DataFrame({"action-1": [8], "action-2": [15]}) # clip to red zone
    clip_action(actions, tag_cfgs, rtol=0)
    assert dfs_close(actions, expected)

    actions = DataFrame({"action-1": [5], "action-2": [-3]})
    expected = DataFrame({"action-1": [5], "action-2": [5]})
    clip_action(actions, tag_cfgs, rtol=0)
    assert dfs_close(actions, expected)

    actions = DataFrame({"action-1": [5], "action-2": [90]})
    expected = DataFrame({"action-1": [5], "action-2": [15]})
    clip_action(actions, tag_cfgs, rtol=0)
    assert dfs_close(actions, expected)

    actions = DataFrame({"action-1": [5], "action-2": [-3]})
    expected = DataFrame({"action-1": [5], "action-2": [5]})
    clip_action(actions, tag_cfgs, rtol=0)
    assert dfs_close(actions, expected)

    actions = DataFrame({"action-1": [30], "action-2": [90]})
    expected = DataFrame({"action-1": [8], "action-2": [15]})
    clip_action(actions, tag_cfgs, rtol=0)
    assert dfs_close(actions, expected)

    actions = DataFrame({"action-1": [-3], "action-2": [-13]})
    expected = DataFrame({"action-1": [0], "action-2": [5]})
    clip_action(actions, tag_cfgs, rtol=0)
    assert dfs_close(actions, expected)

def test_action_sanitizer(multi_tag_minimal_cfg: MainConfig):
    tag_cfg_1 = multi_tag_minimal_cfg.pipeline.tags[0]
    tag_cfg_2 = multi_tag_minimal_cfg.pipeline.tags[1]

    assert isinstance(tag_cfg_1, SetpointTagConfig)
    assert isinstance(tag_cfg_2, SetpointTagConfig)

    tag_cfgs = {
        "action-1": tag_cfg_1,
        "action-2": tag_cfg_2,
    }

    actions = DataFrame({"action-1": [30], "action-2": [30]})
    expected = DataFrame({"action-1": [8], "action-2": [15]}) # clip to red zone
    sanitize_actions(actions, tag_cfgs, rtol=0)
    assert dfs_close(actions, expected)

    # If there are too many actions, it's not clear what to do. Nuke the action
    actions = DataFrame({"action-1": [30, 12], "action-2": [30, 2]})
    expected = DataFrame({})
    sanitize_actions(actions, tag_cfgs, rtol=0)
    assert dfs_close(actions, expected)

    actions = DataFrame({})
    expected = DataFrame({})
    sanitize_actions(actions, tag_cfgs, rtol=0)
    assert dfs_close(actions, expected)

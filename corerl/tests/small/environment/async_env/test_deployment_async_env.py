from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from coreio.utils.io_events import IOEvent, IOEventType, OPCUANodeWriteValue
from pandas import DataFrame
from test.infrastructure.utils.pandas import dfs_close

from corerl.data_pipeline.db.data_reader import TagDBConfig
from corerl.data_pipeline.transforms import NukeConfig
from corerl.environment.async_env.async_env import AsyncEnvConfig
from corerl.environment.async_env.deployment_async_env import DeploymentAsyncEnv, clip_action, sanitize_actions
from corerl.tags.setpoint import SetpointTagConfig

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
def minimal_cfg():
    return AsyncEnvConfig(
        coreio_origin="tcp://localhost:5555",
        db=TagDBConfig(),
        obs_period=pd.Timedelta("1min"),
    )

@pytest.fixture
def minimal_tag_cfgs():
    return [
        SetpointTagConfig(
            name="action-1",
            operating_range=(0.0, 10.0),
            node_identifier="ns=2;i=1",
            connection_id="conn-1",
        ),
    ]

@pytest.fixture
def deployment_env(
    minimal_cfg: AsyncEnvConfig,
    minimal_tag_cfgs: list[SetpointTagConfig],
):
    with patch("corerl.environment.async_env.deployment_async_env.DataReader", autospec=True):
        return DeploymentAsyncEnv(minimal_cfg, minimal_tag_cfgs)


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
        type=IOEventType.write_opcua_nodes,
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

def make_multi_tag_cfgs():
    return [
        SetpointTagConfig(
            name="action-1",
            operating_range=(0.0, 10.0),
            node_identifier="ns=2;i=1",
            connection_id="conn-1",
        ),
        SetpointTagConfig(
            name="action-2",
            operating_range=(5.0, 15.0),
            node_identifier="ns=2;i=2",
            connection_id="conn-2",
        ),
    ]

@pytest.fixture
def multi_deployment_env(minimal_cfg: AsyncEnvConfig):
    tag_cfgs = make_multi_tag_cfgs()
    with patch("corerl.environment.async_env.deployment_async_env.DataReader", autospec=True):
        return DeploymentAsyncEnv(minimal_cfg, tag_cfgs)


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
        type=IOEventType.write_opcua_nodes,
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
    assert actual_event.data["conn-1"][0].value == pytest.approx(0.01, abs=1e-6)
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

def test_action_clipping():
    """
    Clips actions to operating or red zone bounds as expected.
    """
    tag_cfgs = {
        "action-1": SetpointTagConfig(
            name="action-1",
            operating_range=(5.0, 15.0),
            red_bounds=(None, 12),
            preprocess=[],
            state_constructor=[NukeConfig()],
        ),
        "action-2": SetpointTagConfig(
            name="action-2",
            operating_range=(0.0, 60.0),
            preprocess=[],
            state_constructor=[NukeConfig()],
        ),
    }

    actions = DataFrame({"action-1": [2.5], "action-2": [30]})
    expected = DataFrame({"action-1": [5], "action-2": [30]})
    clip_action(actions, tag_cfgs, rtol=0)
    assert dfs_close(actions, expected)

    actions = DataFrame({"action-1": [30], "action-2": [30]})
    expected = DataFrame({"action-1": [12], "action-2": [30]}) # clip to red zone
    clip_action(actions, tag_cfgs, rtol=0)
    assert dfs_close(actions, expected)

    actions = DataFrame({"action-1": [10], "action-2": [-3]})
    expected = DataFrame({"action-1": [10], "action-2": [0]})
    clip_action(actions, tag_cfgs, rtol=0)
    assert dfs_close(actions, expected)

    actions = DataFrame({"action-1": [10], "action-2": [90]})
    expected = DataFrame({"action-1": [10], "action-2": [60]})
    clip_action(actions, tag_cfgs, rtol=0)
    assert dfs_close(actions, expected)

    actions = DataFrame({"action-1": [10], "action-2": [-3]})
    expected = DataFrame({"action-1": [10], "action-2": [0]})
    clip_action(actions, tag_cfgs, rtol=0)
    assert dfs_close(actions, expected)

    actions = DataFrame({"action-1": [30], "action-2": [90]})
    expected = DataFrame({"action-1": [12], "action-2": [60]})
    clip_action(actions, tag_cfgs, rtol=0)
    assert dfs_close(actions, expected)

    actions = DataFrame({"action-1": [3], "action-2": [-13]})
    expected = DataFrame({"action-1": [5], "action-2": [0]})
    clip_action(actions, tag_cfgs, rtol=0)
    assert dfs_close(actions, expected)

def test_action_sanitizer():
    tag_cfgs = {
        "action-1": SetpointTagConfig(
            name="action-1",
            operating_range=(5.0, 15.0),
            red_bounds=(None, 12),
            preprocess=[],
            state_constructor=[NukeConfig()],
        ),
        "action-2": SetpointTagConfig(
            name="action-2",
            operating_range=(0.0, 60.0),
            preprocess=[],
            state_constructor=[NukeConfig()],
        ),
    }

    actions = DataFrame({"action-1": [30], "action-2": [30]})
    expected = DataFrame({"action-1": [12], "action-2": [30]}) # clip to red zone
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

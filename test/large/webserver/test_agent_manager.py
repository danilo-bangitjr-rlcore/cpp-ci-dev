import asyncio
import sqlite3
from json import dumps
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from websockets import ConnectionClosedOK

from corerl.web.app import app


@pytest.fixture
def tmp_sqlite_file(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory):
    """Fixture to create a temporary SQLite file."""
    db_file = tmp_path_factory.mktemp("db") / "test_coreio.db"
    monkeypatch.setenv("COREIO_SQLITE_DB_PATH", db_file.as_posix())
    yield db_file
    # Cleanup is handled by pytest's tmp_path_factory


@pytest.mark.asyncio
async def test_start_agent_emits_ws_text(tmp_sqlite_file: Path):
    """
    Test that starting an agent emits text to the websocket.
    This test simulates starting an agent with an invalid configuration and checks that the websocket emits messages.
    It uses a temporary SQLite file to store the agent configuration. This test is not intended to verify that our
    agent itself can run, but rather that the subprocess call and websocket communication are working as expected.
    """

    num_lines_to_check = 10
    test_client = TestClient(app)

    # create a temporary SQLite file with stub agent config
    with sqlite3.connect(tmp_sqlite_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS CoreRLConfig (
                id TEXT NOT NULL PRIMARY KEY,
                coreRLVersion TEXT,
                config JSONB
            )
            """
        )
        min_config = dumps(
            {
                "interaction": {"name": "dep_interaction", "action_period": "PT4M", "obs_period": "PT2M"},
                "env": {"name": "dep_async_env"},
            }
        )
        cursor.execute(
            """
            INSERT INTO CoreRLConfig (id, config) VALUES (?, ?)
            """,
            (
                "test_agent_config_id",
                min_config,
            ),
        )
        conn.commit()

    # start the agent using the fastapi agent manager
    response = test_client.post(
        "/api/corerl/agents/test_agent_config_id/start",
        json={},
    )
    assert response.status_code == 200
    assert response.json().get("status") == "success"

    # verify that the agent is running before continuing
    while True:
        response = test_client.get(
            "/api/corerl/agents/test_agent_config_id/status",
        )
        assert response.status_code == 200
        if response.json().get("status") == "running":
            break

    received_messages = []
    # start the agent using a POST request, expecting agent start to fail but observe outputs in WS
    async def listen_to_websocket():
        with test_client.websocket_connect("/api/corerl/agents/test_agent_config_id/ws") as websocket:
            try:
                while len(received_messages) < num_lines_to_check:
                    try:
                        async with asyncio.timeout(0.1):
                            message = await asyncio.to_thread(websocket.receive_text)
                        received_messages.append(message)
                    except asyncio.TimeoutError:
                        continue
                    finally:
                        await asyncio.sleep(0.01)
            except ConnectionClosedOK:
                pass

    # Start the websocket listener in the background
    websocket_task = asyncio.create_task(listen_to_websocket())

    timeout_counter = 0
    while len(received_messages) < num_lines_to_check and timeout_counter < 100:
        await asyncio.sleep(0.5)
        timeout_counter += 1
    assert len(received_messages) >= num_lines_to_check
    websocket_task.cancel()
    try:
        await websocket_task
    except asyncio.CancelledError:
        pass

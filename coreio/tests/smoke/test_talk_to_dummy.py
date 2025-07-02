# test_zmq_communication.py
import asyncio
from typing import Any

import pytest
from test.infrastructure.networking import get_free_port

from tests.infrastructure.mock_opc_server import FakeOpcServer


@pytest.fixture
def opc_port():
    """
    Gets a free port from localhost that the server can listen on
    instead of assuming any particular one will be free
    """
    return get_free_port('localhost')

async def run_subprocess_with_streaming(name: str, *args: str) -> dict[str, Any]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    stdout_lines = []
    assert proc.stdout is not None

    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        line_str = line.decode().rstrip()
        print(f"[{name}] {line_str}")
        stdout_lines.append(line_str)

    await proc.wait()
    return {
        "returncode": proc.returncode,
        "stdout": "\n".join(stdout_lines),
    }

@pytest.mark.timeout(60)
async def test_communication(opc_port: int):
    server = FakeOpcServer(opc_port)
    await server.start()
    await asyncio.sleep(0.1)

    try:
        await asyncio.sleep(0.1)

        task_coreio = asyncio.create_task(run_subprocess_with_streaming(
            "coreio",
            "python",
            "coreio/main.py",
            "--config-name",
            "../config/coreio_test_config.yaml",
            f"coreio.opc_connections[0].opc_conn_url=opc.tcp://admin@localhost:{opc_port}",
        ))

        await asyncio.sleep(1)  # Wait for coreio startup

        task_dummy_agent = asyncio.create_task(run_subprocess_with_streaming(
            "dummy_agent",
            "python",
            "coreio/dummy_agent.py",
            "--config-name",
            "../config/coreio_test_config.yaml",
            "interaction.action_period=00:00:00.02",
        ))

        try:
            dummy_result = await asyncio.wait_for(task_dummy_agent, timeout=30)
            coreio_result = await asyncio.wait_for(task_coreio, timeout=30)
        except TimeoutError as e:
            task_dummy_agent.cancel()
            task_coreio.cancel()
            raise AssertionError("Test timed out waiting for processes to complete") from e

        assert dummy_result["returncode"] == 0, (
            f"dummy_agent failed (exit code: {dummy_result['returncode']})\n"
            f"Output: {dummy_result['stdout']}"
        )
        assert coreio_result["returncode"] == 0, (
            f"CoreIO failed (exit code: {coreio_result['returncode']})\n"
            f"Output: {coreio_result['stdout']}"
        )

    finally:
        await server.close()

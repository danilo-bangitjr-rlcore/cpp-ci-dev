from pathlib import Path

import pytest
from test.infrastructure.networking import get_free_port

from coreio.communication.opc_communication import OPC_Connection_IO
from tests.infrastructure.load_config import load_config
from tests.infrastructure.mock_opc_server import FakeOpcServer


@pytest.fixture
def opc_port():
    """
    Gets a free port from localhost that the server can listen on
    instead of assuming any particular one will be free
    """
    return get_free_port('localhost')

@pytest.fixture
async def client():
    client = OPC_Connection_IO()
    yield client
    await client.cleanup()


async def test_read(server: FakeOpcServer, client: OPC_Connection_IO, opc_port: int):
    """
    Client should be able to connect to a running server.
    """
    config = load_config(Path("assets", "basic.yaml"), opc_port)
    await client.init(config)

    async with client:
        await client.register_node("ns=2;i=2", "temp")
        await client.register_node("ns=2;i=3", "pressure")
        node_names_val = await client.read_nodes_named(client.registered_nodes)

    # Assert nodes have expected values
    assert "temp" in node_names_val
    assert "pressure" in node_names_val
    assert node_names_val["temp"] == 0.12
    assert node_names_val["pressure"] == 0.34

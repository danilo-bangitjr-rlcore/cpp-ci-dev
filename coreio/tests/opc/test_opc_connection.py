import asyncio
from pathlib import Path

import pytest
from test.infrastructure.networking import get_free_port

from coreio.communication.opc_communication import OPC_Connection
from tests.infrastructure.load_config import load_config
from tests.infrastructure.mock_opc_certs import ServerClientKeyCerts
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
    client = OPC_Connection()
    yield client
    await client.cleanup()


@pytest.fixture
async def server_key_cert(opc_port: int, client_server_key_certs: ServerClientKeyCerts):
    s = FakeOpcServer(opc_port)
    await s.start_encrypt(client_server_key_certs)
    yield s
    await s.close()

async def test_connect1(server: FakeOpcServer, client: OPC_Connection, opc_port: int):
    """
    Client should be able to connect to a running server.
    """
    config = load_config(Path("assets", "basic.yaml"), opc_port)
    await client.init(config)
    await client.start()

async def test_connect2(client: OPC_Connection, opc_port: int):
    """
    Client should connect to a server that is started after the client.
    """
    client_config = load_config(Path("assets", "basic.yaml"), opc_port)
    connect_task = asyncio.create_task(client.init(client_config)) # Client tries to connect during init
    await asyncio.sleep(1)

    server = FakeOpcServer(opc_port)
    await server.start()

    await asyncio.sleep(1)
    await asyncio.wait_for(connect_task, 30)
    await client.ensure_connected()

async def test_disconnect1(server: FakeOpcServer, client: OPC_Connection, opc_port: int):
    """
    Client survives when a server goes offline after connection.
    Check this sequence:
      1. Client and server connect
      2. Server closes
      3. Server starts
      4. Client implicitly reconnects in the background
    """
    client_config = load_config(Path("assets", "basic.yaml"), opc_port)
    await client.init(client_config)
    await server.close()
    await asyncio.sleep(0.1)
    await server.start()
    await client.ensure_connected()

async def test_connect_encrypt(
        server_key_cert: FakeOpcServer, client: OPC_Connection,
        opc_port: int, client_server_key_certs: ServerClientKeyCerts,
    ):
    """
    Basic256_SHA256, sign and encrypt, username password client can connect
    to a server endpoint with same config
    """
    client_config = load_config(Path("assets", "sha256_se_up.yaml"), opc_port, client_server_key_certs)
    await client.init(client_config)
    await client.start()

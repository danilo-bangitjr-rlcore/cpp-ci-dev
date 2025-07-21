import asyncio
from pathlib import Path

import pytest
from lib_config.loader import direct_load_config
from test.infrastructure.networking import get_free_port

# from corerl.config import MainConfig
from coreio.communication.opc_communication import OPC_Connection, OPCConnectionConfig
from coreio.config import OPCSecurityPolicyBasic256SHA256Config
from coreio.utils.config_schemas import MainConfigAdapter
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


async def test_connect1(server: FakeOpcServer, client: OPC_Connection, opc_port: int):
    """
    Client should be able to connect to a running server.
    """
    config = load_config(Path("assets", "basic.yaml"), opc_port)
    await client.init(config)
    await client.start()

    # Cleanup
    await client.cleanup()

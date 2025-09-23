from pathlib import Path

import pytest
from test.infrastructure.networking import get_free_port

from coreio.communication.opc_communication import OPC_Connection_IO
from tests.infrastructure.load_config import load_config
from tests.infrastructure.mock_opc_certs import ServerClientKeyCerts
from tests.infrastructure.mock_opc_server import FakeOpcServer


@pytest.fixture
def opc_port():
    return get_free_port('localhost')

@pytest.fixture
async def client():
    c = OPC_Connection_IO()
    yield c
    await c.cleanup()

@pytest.fixture
async def server_key_cert(opc_port: int, client_server_key_certs: ServerClientKeyCerts):
    s = FakeOpcServer(opc_port)
    await s.start_encrypt(client_server_key_certs)
    yield s
    await s.close()


async def test_application_uri_from_cert(
    server_key_cert: FakeOpcServer,
    client: OPC_Connection_IO,
    opc_port: int,
    client_server_key_certs: ServerClientKeyCerts,
):
    """
    When application_uri is not provided in config and security policy is Basic256_SHA256,
    the client should extract application URI from the client certificate SAN (UniformResourceIdentifier)
    and set it on the underlying asyncua Client instance. The mock client certificate has URI 'urn:client'.
    """
    cfg = load_config(Path("assets", "sha256_se_up_no_uri.yaml"), opc_port, client_server_key_certs)
    # Force removal of application_uri to mimic absence (load_config copies directly, but file omits it)
    assert cfg.application_uri is None
    await client.init(cfg)

    # init should have set application_uri from cert (see mock_opc_certs: client uri is urn:client)
    assert client.opc_client is not None
    assert client.opc_client.application_uri == "urn:client"

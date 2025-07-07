import asyncio
from pathlib import Path

import pytest
from lib_config.loader import direct_load_config
from test.infrastructure.networking import get_free_port

from coreio.config import OPCSecurityPolicyBasic256SHA256Config
from coreio.utils.config_schemas import MainConfigAdapter

# from corerl.config import MainConfig
from coreio.utils.opc_communication import OPC_Connection, OPCConnectionConfig
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

def load_config(
        cfg_name: Path, opc_port: int, server_client_key_certs: ServerClientKeyCerts | None = None,
    ) -> OPCConnectionConfig:
    """
    Loads an agent config file, this should have CoreIO config in it
    for testing behavior. Replaces the opc_conn_url with a generated one
    for testing, and also replaces any key/cert info with generated ones
    """
    cfg = direct_load_config(
        MainConfigAdapter,
        config_name=str('tests/opc/' / cfg_name),
    )
    assert isinstance(cfg, MainConfigAdapter)
    config = cfg.coreio.opc_connections[0]
    if isinstance(config.security_policy, OPCSecurityPolicyBasic256SHA256Config):
        assert server_client_key_certs is not None, "Key certs must be provided if security policy is not None"

        return OPCConnectionConfig(
            connection_id=config.connection_id,
            application_uri=config.application_uri,
            authentication_mode=config.authentication_mode,
            opc_conn_url=f'opc.tcp://localhost:{opc_port}',
            security_policy=OPCSecurityPolicyBasic256SHA256Config(
                mode=config.security_policy.mode,
                client_cert_path=Path(server_client_key_certs.client.cert),
                client_key_path=Path(server_client_key_certs.client.key),
                server_cert_path=Path(server_client_key_certs.server.cert),
            ),
        )

    return OPCConnectionConfig(
        connection_id=config.connection_id,
        application_uri=config.application_uri,
        authentication_mode=config.authentication_mode,
        opc_conn_url=f'opc.tcp://localhost:{opc_port}',
        security_policy=config.security_policy,
    )

async def test_connect1(server: FakeOpcServer, client: OPC_Connection, opc_port: int):
    """
    Client should be able to connect to a running server.
    """
    config = load_config(Path("assets", "basic.yaml"), opc_port)
    await client.init(config)
    await client.start()

    # Cleanup
    await client.cleanup()

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

    # Cleanup
    await client.cleanup()
    await server.close()

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

    # Cleanup
    await server.close()
    await client.cleanup()

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

import asyncio

import pytest
from asyncua import Node, Server
from test.infrastructure.networking import get_free_port

from coreio.utils.opc_communication import OPC_Connection, OPCConnectionConfig


class FakeOpcServer:
    def __init__(self, port: int):
        self._port = port
        self._s: Server | None = None
        self._sensors: dict[str, Node] = {}

    async def start(self):
        self._s = Server()
        await self._s.init()

        self._s.set_endpoint(f'opc.tcp://localhost:{self._port}/opcua/')
        self._s.set_server_name('RLCore Test Server')
        idx = await self._s.register_namespace('http://rlcore.test.ai/opcua/')

        virtual_plc = await self._s.nodes.objects.add_object(idx, 'vPLC1')
        self._sensors = {
            'temp': await virtual_plc.add_variable(idx, 'temp', 0.),
            'pressure': await virtual_plc.add_variable(idx, 'pressure', 0.),
        }

        await self._s.start()

    async def step(self, v: float):
        for j, sensor in enumerate(self._sensors.values()):
            await sensor.write_value(float(v + j))

    async def close(self):
        assert self._s is not None
        await self._s.stop()


@pytest.fixture
def opc_port():
    return get_free_port('localhost')


@pytest.fixture
async def server(opc_port: int):
    s = FakeOpcServer(opc_port)
    await s.start()
    yield s
    await s.close()


@pytest.fixture
def client_config(opc_port: int):
    return OPCConnectionConfig(
        connection_id='test',
        opc_conn_url=f'opc.tcp://localhost:{opc_port}',
        client_cert_path=None,
        client_private_key_path=None,
        server_cert_path=None,
    )


@pytest.fixture
async def client():
    client = OPC_Connection()
    yield client
    await client.cleanup()


async def test_connect1(server: FakeOpcServer, client: OPC_Connection, client_config: OPCConnectionConfig):
    """
    Client should be able to connect to a running server.
    """
    await client.init(client_config, [])
    await client.start()


async def test_connect2(client: OPC_Connection, client_config: OPCConnectionConfig):
    """
    Client should fail if a server is not running.
    """
    with pytest.raises(OSError):
        await client.init(client_config, [])
        await client.start()


async def test_disconnect1(server: FakeOpcServer, client: OPC_Connection, client_config: OPCConnectionConfig):
    """
    Client survives when a server goes offline after connection.
    Check this sequence:
      1. Client and server connect
      2. Server closes
      3. Server starts
      4. Client implicitly reconnects in the background
    """
    await client.init(client_config, [])
    await server.close()
    await asyncio.sleep(0.1)
    await server.start()
    await client.ensure_connected()

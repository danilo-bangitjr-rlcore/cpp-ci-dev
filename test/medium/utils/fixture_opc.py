import pytest
from asyncua import Node, Server

from corerl.utils.opc_connection import OpcConfig, OpcConnection


class FakeOpcServer:
    def __init__(self, port: int = 0):
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
async def server(free_localhost_port: int):
    s = FakeOpcServer(free_localhost_port)
    await s.start()
    yield s
    await s.close()


@pytest.fixture
async def client(free_localhost_port: int):
    config = OpcConfig(
        ip_address='localhost',
        port=free_localhost_port,
        timeout=1.,
    )

    client = OpcConnection(config)
    yield client
    await client.disconnect()


@pytest.fixture
async def server_and_client(free_localhost_port: int):
    # building a server should find us an open port
    server = FakeOpcServer(free_localhost_port)
    await server.start()

    # let the client's config know what port we are using
    config = OpcConfig(
        ip_address='localhost',
        port=free_localhost_port,
        timeout=1.,
    )

    client = OpcConnection(config)
    yield (server, client)

    await client.disconnect()
    await server.close()

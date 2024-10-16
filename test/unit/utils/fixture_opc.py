import socket
import pytest_asyncio
from asyncua import Server, Node
from corerl.utils.opc_connection import OpcConnection


class FakeOpcServer:
    def __init__(self, port: int = 0):
        self._port = port or get_free_port('localhost')
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


@pytest_asyncio.fixture(loop_scope='function')
async def server():
    s = FakeOpcServer()
    await s.start()
    yield s
    await s.close()



class OpcConfigStub:
    def __init__(self):
        self.ip_address = 'localhost'
        self.port = 4840
        self.conn_stats = None
        self.vendor = 'ignition'
        self.timeout = 1


@pytest_asyncio.fixture(loop_scope='function')
async def client():
    config = OpcConfigStub()

    client = OpcConnection(config)
    yield client
    await client.disconnect()


@pytest_asyncio.fixture(loop_scope='function')
async def server_and_client():
    # building a server should find us an open port
    server = FakeOpcServer()
    await server.start()

    # let the client's config know what port we are using
    config = OpcConfigStub()
    config.port = server._port

    client = OpcConnection(config)
    yield (server, client)

    await client.disconnect()
    await server.close()



def get_free_port(host: str):
    # binding to port 0 will ask the OS to give us an arbitrary free port
    # since we've just bound that free port, it is by definition no longer free,
    # so we set that port as reuseable to allow another socket to bind to it
    # then we immediately close the socket and release our connection.
    sock = socket.socket()
    sock.bind((host, 0))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    port: int = sock.getsockname()[1]
    sock.close()

    return port

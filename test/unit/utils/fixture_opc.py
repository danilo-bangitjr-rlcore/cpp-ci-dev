import pytest_asyncio
from asyncua import Server, Node
from corerl.utils.opc_connection import OpcConnection


class TestOpcServer:
    def __init__(self):
        self._s: Server | None = None
        self._sensors: dict[str, Node] = {}

    async def start(self):
        self._s = Server()
        await self._s.init()

        self._s.set_endpoint('opc.tcp://127.0.0.1:4840/opcua/')
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
    s = TestOpcServer()
    await s.start()
    yield s
    await s.close()



class OpcConfigStub:
    def __init__(self):
        self.ip_address = '127.0.0.1'
        self.port = 4840
        self.conn_stats = None
        self.vendor = 'ignition'


@pytest_asyncio.fixture(loop_scope='function')
async def client():
    config = OpcConfigStub()

    client = OpcConnection(config)
    yield client
    await client.disconnect()

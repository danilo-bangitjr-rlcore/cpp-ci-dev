import pytest

from corerl.messages.client import WebsocketClient
from corerl.messages.server import WebsocketServer, WebsocketServerConfig
from test.infrastructure.networking import get_free_port


@pytest.fixture
async def server_and_client():
    # Both server and client need to be pointing to the same
    # arbitrary port. Easiest to just build both at the same
    # time to manage this shared state.
    p = get_free_port('localhost')

    config = WebsocketServerConfig(host='localhost', port=p)
    s = WebsocketServer(config)
    c = WebsocketClient('localhost', p)

    yield s, c

    await s.close()
    await c.close()


@pytest.fixture
async def client():
    p = get_free_port('localhost')
    c = WebsocketClient('localhost', p)

    yield c
    await c.close()


@pytest.fixture
async def server():
    p = get_free_port('localhost')
    config = WebsocketServerConfig(host='localhost', port=p)
    s = WebsocketServer(config)

    yield s
    await s.close()

import pytest_asyncio
from test.infrastructure.networking import get_free_port

from corerl.messages.client import WebsocketClient
from corerl.messages.server import WebsocketServer

@pytest_asyncio.fixture(loop_scope='function')
async def server_and_client():
    # Both server and client need to be pointing to the same
    # arbitrary port. Easiest to just build both at the same
    # time to manage this shared state.
    p = get_free_port('localhost')
    s = WebsocketServer('localhost', p)
    c = WebsocketClient('localhost', p)

    yield s, c

    await s.close()
    await c.close()


@pytest_asyncio.fixture(loop_scope='function')
async def client():
    p = get_free_port('localhost')
    c = WebsocketClient('localhost', p)

    yield c
    await c.close()


@pytest_asyncio.fixture(loop_scope='function')
async def server():
    p = get_free_port('localhost')
    s = WebsocketServer('localhost', p)

    yield s
    await s.close()

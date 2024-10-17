import asyncio
from test.unit.messages.fixtures import WebsocketClient, WebsocketServer
from corerl.messages.events import EventType


# ----------------
# -- Connection --
# ----------------
async def test_connect1(server_and_client: tuple[WebsocketServer, WebsocketClient]):
    """
    Client can connect to a running server
    """
    server, client = server_and_client

    await server.start()
    await client.start()

    await client.send_message('hi')

async def test_connect2(server_and_client: tuple[WebsocketServer, WebsocketClient]):
    """
    Client can connect to a server, even when
    the server starts running after the client.
    """
    server, client = server_and_client

    await client.start()
    await asyncio.sleep(0.1)
    await server.start()
    await client.send_message('hi')



# ---------------
# -- Messaging --
# ---------------

async def test_message1(client: WebsocketClient):
    """
    If an unconnected client sends a message,
    it times out but does not raise an exception.
    """
    await client.start()
    await client.send_message('hi')


async def test_message2(server_and_client: tuple[WebsocketServer, WebsocketClient]):
    """
    Client can send a message to a listening server.
    """
    server, client = server_and_client
    await client.start()
    await server.start()

    await client.send_message('hi')


async def test_message3(server_and_client: tuple[WebsocketServer, WebsocketClient]):
    """
    Client can send an event to a listening server.
    """
    server, client = server_and_client
    await client.start()
    await server.start()

    await client.emit_event(EventType.agent_heartbeat)

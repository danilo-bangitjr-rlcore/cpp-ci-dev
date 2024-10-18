import pytest

from websockets.asyncio.client import connect

from test.unit.messages.fixtures import * # noqa: F403
from corerl.messages.events import Event, EventType, SubscribeEvent
from corerl.messages.client import WebsocketClient

# ----------------
# -- Connection --
# ----------------
async def test_connect1(server):
    """
    Multiple clients can connect to a single
    server.
    """
    await server.start()

    c1 = WebsocketClient(server.host, server.port)
    await c1.start()
    await c1.ensure_connected()

    c2 = WebsocketClient(server.host, server.port)
    await c2.start()
    await c2.ensure_connected()

    assert len(server.connected_clients()) == 2

    await c1.close()
    await c2.close()



# ------------------
# -- Subscription --
# ------------------
@pytest.mark.timeout(1)
async def test_subscribe1(server_and_client):
    """
    Given a server and two clients, one client can
    subscribe to events from the server, while the
    other client emits events of that type.

    The subscriber will receive events in json format.
    """
    server, sender = server_and_client

    await server.start()
    await sender.start()

    # manually build a second client so that we can
    # intercept response messages
    receiver = await connect(f'ws://{server.host}:{server.port}')

    # now c2 should receive messages from the server for this event
    # type
    sub_event = SubscribeEvent(subscribe_to=EventType.agent_heartbeat)
    await receiver.send(sub_event.model_dump_json())

    # client 1 sends a heartbeat
    # the server, then, forwards that on to c2
    await sender.emit_event(EventType.agent_heartbeat)
    msg = await receiver.recv()

    event = Event.model_validate_json(msg)
    assert event.type == EventType.agent_heartbeat

    await receiver.close()


@pytest.mark.timeout(10)
async def test_subscribe2(server_and_client):
    """
    Given a server and two clients, one client can
    subscribe to events from the server, while the
    other client emits events of that type.

    The subscriber will receive events in json format.

    The sending client can start and send events
    synchronously, without using the asyncio api.
    """
    server, sender = server_and_client

    await server.start()
    sender.start_sync()

    # manually build a second client so that we can
    # intercept response messages
    receiver = await connect(f'ws://{server.host}:{server.port}')

    # now c2 should receive messages from the server for this event
    # type
    sub_event = SubscribeEvent(subscribe_to=EventType.agent_heartbeat)
    await receiver.send(sub_event.model_dump_json())

    # client 1 sends a heartbeat
    # the server, then, forwards that on to c2
    sender.emit_event_sync(EventType.agent_heartbeat)
    msg = await receiver.recv()

    event = Event.model_validate_json(msg)
    assert event.type == EventType.agent_heartbeat

    await receiver.close()
    sender.close_sync()

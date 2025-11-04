# Event Bus Client

Generic ZeroMQ-based event bus client for microservice communication.

## Overview

The `EventBusClient` provides a type-safe, generic interface for publishing and subscribing to events via a ZeroMQ XPUB/XSUB proxy. It supports:

- **Generic event types**: Use any Pydantic model that extends `BaseEvent`
- **Topic-based filtering**: Subscribe to specific event topics
- **Callback system**: Attach callbacks to specific event types
- **Lifecycle management**: Clean connect/close with resource cleanup
- **Thread-safe consumption**: Background thread for message processing

## Architecture

```
┌─────────────┐          ┌──────────────┐          ┌─────────────┐
│  Publisher  │          │  ZMQ Proxy   │          │ Subscriber  │
│   Client    │──────────│              │──────────│   Client    │
│             │  PUB     │  XSUB/XPUB   │   SUB    │             │
└─────────────┘  :5559   └──────────────┘   :5560  └─────────────┘
```

The event bus uses ZeroMQ's pub/sub pattern with a central proxy (typically run by coredinator):
- Publishers connect to port 5559 (XSUB socket)
- Subscribers connect to port 5560 (XPUB socket)
- The proxy forwards messages between publishers and subscribers

## Usage

### Basic Example

```python
from enum import auto
from pydantic import Field
from lib_defs.type_defs.base_events import BaseEvent, BaseEventTopic, BaseEventType
from lib_utils.messages import EventBusClient
from lib_utils.time import now_iso

# Define your event types
class MyEventType(BaseEventType):
    started = auto()
    updated = auto()
    stopped = auto()

class MyEventTopic(BaseEventTopic):
    my_service = auto()

class MyEvent(BaseEvent[MyEventType]):
    time: str = Field(default_factory=now_iso)
    type: MyEventType

# Create a client
client = EventBusClient[MyEvent, MyEventType, MyEventTopic](
    event_class=MyEvent,
    host="localhost",
    pub_port=5559,
    sub_port=5560,
)

# Connect and use
client.connect()
client.emit_event(MyEventType.started, topic=MyEventTopic.my_service)
client.close()
```

### Creating a Service-Specific Client

For convenience, create a service-specific subclass:

```python
class MyServiceEventBusClient(EventBusClient[MyEvent, MyEventType, MyEventTopic]):
    def __init__(
        self,
        host: str = "localhost",
        pub_port: int = 5559,
        sub_port: int = 5560,
    ):
        super().__init__(
            event_class=MyEvent,
            host=host,
            pub_port=pub_port,
            sub_port=sub_port,
        )
```

### Publishing Events

```python
# Emit an event instance
event = MyEvent(type=MyEventType.started)
client.emit_event(event, topic=MyEventTopic.my_service)

# Emit just the event type (auto-wraps in event)
client.emit_event(MyEventType.updated, topic=MyEventTopic.my_service)
```

### Subscribing to Events

```python
# Subscribe to a topic
client.subscribe(MyEventTopic.my_service)

# Start the consumer thread
client.start_consumer()

# Receive events manually
event = client.recv_event(timeout=1.0)
if event:
    print(f"Received: {event.type}")

# Or use listen_forever generator
for event in client.listen_forever():
    print(f"Received: {event.type}")
```

### Callbacks

```python
def handle_started(event: MyEvent):
    print(f"Service started at {event.time}")

def handle_updated(event: MyEvent):
    print(f"Service updated at {event.time}")

# Attach callbacks
client.attach_callback(MyEventType.started, handle_started)
client.attach_callback(MyEventType.updated, handle_updated)

# Or attach multiple at once
client.attach_callbacks({
    MyEventType.started: handle_started,
    MyEventType.updated: handle_updated,
})

# Start consumer to trigger callbacks
client.start_consumer()
```

## API Reference

### Constructor

```python
EventBusClient[EventClass, EventTypeClass, EventTopicClass](
    event_class: type[EventClass],
    host: str = "localhost",
    pub_port: int = 5559,
    sub_port: int = 5560,
)
```

**Parameters:**
- `event_class`: The Pydantic model class for events
- `host`: ZMQ proxy hostname
- `pub_port`: Port for publishers to connect to (proxy's XSUB)
- `sub_port`: Port for subscribers to connect to (proxy's XPUB)

### Methods

#### `connect()`
Establish ZMQ connections. Must be called before publishing or subscribing.

#### `close()`
Clean shutdown: stops consumer thread, drains queue, closes sockets.

#### `is_connected() -> bool`
Check if client is currently connected.

#### `emit_event(event: EventClass | EventTypeClass, topic: EventTopicClass)`
Publish an event to a topic.

#### `subscribe(topic: EventTopicClass)`
Subscribe to receive events from a specific topic.

#### `start_consumer()`
Start background thread for receiving messages and invoking callbacks.

#### `attach_callback(event_type: EventTypeClass, cb: Callback)`
Register a callback for a specific event type.

#### `attach_callbacks(cbs: dict[EventTypeClass, Callback])`
Register multiple callbacks at once.

#### `recv_event(timeout: float = 0.5) -> EventClass | None`
Manually receive a single event from the queue.

#### `listen_forever() -> Generator[EventClass]`
Generator that yields events as they arrive.

## Integration with CoreRL

CoreRL provides `RLEventBusClient` as a pre-configured subclass:

```python
from corerl.event_bus import EventBusClient  # or RLEventBusClient
from corerl.messages.events import RLEventType, RLEventTopic

client = EventBusClient()  # Uses RLEvent by default
client.connect()
client.emit_event(RLEventType.service_started, topic=RLEventTopic.corerl)
```

## Integration with CoreIO

To use in CoreIO or other services, follow the same pattern:

1. Define your event types (or reuse RLEvent if appropriate)
2. Create a service-specific subclass
3. Use the same API for publishing and subscribing

## Testing

The library includes comprehensive tests. To run:

```bash
cd libs/lib_utils
uv run pytest tests/test_event_bus_client.py
```

## Dependencies

- `pyzmq` - ZeroMQ Python bindings
- `pydantic` - Event validation and serialization
- `lib_defs` - Base event type definitions

## Design Decisions

1. **Generics with PEP 695 syntax**: Uses modern Python 3.13 type parameter syntax for clean generic definitions
2. **Pydantic for serialization**: Events are JSON-serialized via Pydantic's `model_dump_json()`
3. **Multipart messages**: `[topic, payload]` format for efficient topic filtering
4. **Background consumer thread**: Enables async event processing without blocking
5. **Queue-based buffering**: Events are queued for reliable delivery to callbacks
6. **Component-based configuration**: Separate host/port parameters instead of full URLs
7. **Backward compatibility**: CoreRL maintains `EventBusClient` alias for existing code

## Future Enhancements

- Automatic reconnection on connection loss
- Metrics/observability hooks
- Message compression for large payloads
- Authentication/encryption layer

# lib_events

Event bus infrastructure for RLCore microservices using ZeroMQ DEALER/ROUTER pattern.

## Overview

`lib_events` provides a centralized event bus implementation for inter-service communication in the RLCore monorepo. It includes both client and server components for reliable message passing using ZeroMQ's DEALER/ROUTER pattern, supporting both publish-subscribe and request-reply messaging.

## Components

### Client (`lib_events.client`)

- **`EventBusClient`**: Generic, type-safe event bus client with automatic reconnection, topic-based subscriptions, and request-reply support

### Server (`lib_events.server`)

- **`EventBusProxy`**: ZMQ ROUTER broker handling service registration, topic subscriptions, event publishing, and request-reply routing
- **`EventBusManager`**: Lifecycle manager for EventBusProxy with health monitoring

### Protocol (`lib_events.protocol`)

- **`MessageType`**: Message type enum (REGISTER, SUBSCRIBE, PUBLISH, REQUEST, REPLY)
- **`build_message`**, **`parse_message`**: Message serialization/deserialization helpers

### Legacy (`lib_events.legacy`)

⚠️ **Deprecated components** - Use `EventBusClient` instead for new code

- **`BaseEventBus`**: Legacy pub/sub event bus (deprecated)
- **`consumer_task`**, **`Clock`**: Supporting utilities for `BaseEventBus`

## Architecture

The event bus uses a **unified DEALER/ROUTER broker pattern** to support both publish-subscribe and request-reply messaging:

```
Services (DEALER) → [ROUTER] → EventBusProxy → [ROUTER] → Services (DEALER)
                         :5580 (single port)
```

### Message Flow

1. **Service Registration**: Client sends REGISTER message with service_id to enable request routing
2. **Topic Subscription**: Client sends SUBSCRIBE message with topic name to receive published events
3. **Event Publishing**: Client sends PUBLISH message with topic; broker forwards to all subscribers
4. **Request-Reply**: Client sends REQUEST with correlation_id; broker routes to registered service; service sends REPLY

### Key Benefits

- **Unified communication**: Single port and socket type for all messaging patterns
- **Dynamic routing**: Services register on-demand; no static topology
- **Correlation tracking**: Request-reply flows tracked via correlation_id
- **Topic filtering**: Subscribers receive only events matching their subscribed topics

## Message Protocol

All messages follow a 4-frame structure for consistent routing and handling.

### Frame Structure

```
[destination] [message_type] [correlation_id] [payload]
```

1. **destination** (string): Topic name for PUBLISH/SUBSCRIBE, service_id for REQUEST/REPLY
2. **message_type** (string): One of the MessageType enum values
3. **correlation_id** (string): Unique ID for matching requests with replies (empty for pub-sub)
4. **payload** (bytes): JSON-serialized event data or raw bytes for request-reply

### MessageType Enum

```python
class MessageType(StrEnum):
    REGISTER = "REGISTER"      # Register service_id with broker
    SUBSCRIBE = "SUBSCRIBE"    # Subscribe to topic (destination = topic name)
    PUBLISH = "PUBLISH"        # Publish event to topic
    REQUEST = "REQUEST"        # Send request to service
    REPLY = "REPLY"            # Send reply back to requester
```

### Message Examples

**Service Registration:**
```
["my-service", "REGISTER", "", b""]
```

**Topic Subscription:**
```
["corerl", "SUBSCRIBE", "", b""]
```

**Event Publishing:**
```
["corerl", "PUBLISH", "", b'{"type":"step_agent_update","data":{"step":100}}']
```

**Request-Reply:**
```
# Request
["responder-service", "REQUEST", "uuid-1234", b'{"action":"get_state"}']

# Reply
["requester-service", "REPLY", "uuid-1234", b'{"state":"running"}']
```

### Helper Functions

```python
from lib_events.protocol import build_message, parse_message, MessageType, ParsedMessage

# Build a message
frames = build_message(
    destination="corerl",
    msg_type=MessageType.PUBLISH,
    correlation_id="",
    payload=b'{"type":"update"}'
)

# Parse received message
parsed = parse_message(frames)
if parsed.is_some():
    msg = parsed.expect()
    # Access message fields via dataclass
    destination = msg.destination
    msg_type = msg.msg_type
    correlation_id = msg.correlation_id
    payload = msg.payload
```

## Quick Start

### Publish-Subscribe Pattern

```python
from lib_events.client.event_bus_client import EventBusClient
from lib_defs.type_defs.base_events import Event, EventType, EventTopic

# Initialize client
client = EventBusClient(
    host="localhost",
    port=5580,
    service_id="my-service",
)

# Connect and subscribe to topics
client.connect()
client.subscribe(EventTopic.corerl)

# Attach callbacks for specific event types
def handle_update(event: Event):
    print(f"Received: {event}")

client.attach_callback(EventType.step_agent_update, cb=handle_update)

# Start consumer thread
client.start_consumer()

# Publish events
event = Event(type=EventType.step_agent_update, data={"step": 100})
client.emit_event(event, topic=EventTopic.corerl)

# Cleanup
client.stop_consumer()
client.close()
```

### Request-Reply Pattern

```python
from lib_events.client.event_bus_client import EventBusClient
from lib_defs.type_defs.base_events import Event, EventType, EventTopic

# Responder service
responder = EventBusClient(
    host="localhost",
    port=5580,
    service_id="responder-service",
)
responder.connect()
responder.register_request_handler(lambda payload: b"response-data")
responder.start_consumer()

# Requester service
requester = EventBusClient(
    host="localhost",
    port=5580,
    service_id="requester-service",
)
requester.connect()
requester.start_consumer()

# Send request
response = requester.send_request(
    service_id="responder-service",
    payload=b"request-data",
    timeout=5.0
)
print(f"Response: {response}")

# Cleanup
requester.close()
responder.close()
```

### Running Event Bus Broker

```python
from lib_events.server.manager import EventBusManager

manager = EventBusManager(
    router_addr="tcp://*:5580",
)

manager.start()
# ... broker runs ...
manager.stop()
```

## Advanced Usage

### Reconnection and Resilience

EventBusClient supports automatic reconnection with exponential backoff:

```python
client = EventBusClient(
    host="localhost",
    port=5580,
    service_id="resilient-service",
    max_reconnect_attempts=-1,  # Infinite retries
    reconnect_interval=1.0,
    reconnect_backoff_multiplier=2.0,
    reconnect_max_interval=60.0,
)
```

### Multiple Callbacks per Event Type

```python
def log_event(event: Event):
    logger.info(f"Event: {event}")

def process_event(event: Event):
    # Process the event
    pass

client.attach_callback(EventType.step_agent_update, cb=log_event)
requester.close()
```

### Service-Specific Wrappers

Create typed wrappers for specific services:

```python
from lib_events.client.event_bus_client import EventBusClient
from lib_defs.type_defs.io_events import IOEvent, IOEventType, IOEventTopic

class IOEventBusClient(EventBusClient):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5580,
    ):
        super().__init__(
            host=host,
            port=port,
            service_id="coreio",
        )
```

## Testing

Testing utilities and fixtures are provided for both library tests and consumer service tests.

### Example Test Fixture

```python
import pytest
from lib_events.server.proxy import EventBusProxy
from lib_events.client.event_bus_client import EventBusClient
from lib_defs.type_defs.base_events import Event, EventType, EventTopic

def get_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

@pytest.fixture
def event_bus_broker():
    port = get_free_port()
    broker = EventBusProxy(router_addr=f"tcp://*:{port}")
    broker.start()
    yield port
    broker.stop()

@pytest.fixture
def event_bus_client(event_bus_broker):
    port = event_bus_broker
    client = EventBusClient(
        host="localhost",
        port=port,
        service_id="test-client",
    )
    client.connect()
    client.start_consumer()
    yield client
    client.stop_consumer()
    client.close()
```

## Event Bus Proxy Internals

The `EventBusProxy` maintains internal state for routing:

- **Service Registry**: Maps `service_id` → ZMQ identity for request routing
- **Topic Subscriptions**: Maps `topic` → set of ZMQ identities for publish routing
- **Pending Requests**: Maps `correlation_id` → requester ZMQ identity for reply routing

### Proxy Operations

1. **REGISTER**: Store service_id → zmq_identity mapping
2. **SUBSCRIBE**: Add zmq_identity to topic's subscriber set
3. **PUBLISH**: Forward to all subscribers of the topic
4. **REQUEST**: Look up service_id, forward message, track correlation_id → requester
5. **REPLY**: Look up requester by correlation_id, forward response, clean up tracking

### Error Handling

- **Unregistered service**: Returns error reply `{"error": "Service not available"}`
- **Unknown correlation_id**: Logs warning, drops reply
- **Queue full**: Logs warning, drops message (non-blocking sends)

## Dependencies

- **lib-defs**: Event type definitions (`Event`, `EventType`, `EventTopic`)
- **pydantic**: Event serialization and validation
- **pyzmq**: ZeroMQ bindings for DEALER/ROUTER sockets

## Migration Notes

### From Legacy BaseEventBus

If you're migrating from `lib_events.legacy.BaseEventBus`:

```python
# Old (deprecated)
from lib_events.legacy.base_event_bus import BaseEventBus
bus = BaseEventBus(
    pub_endpoint="tcp://localhost:5570",
    sub_endpoint="tcp://localhost:5571",
)

# New
from lib_events.client.event_bus_client import EventBusClient
client = EventBusClient(
    host="localhost",
    port=5580,
)
client.connect()
client.subscribe(EventTopic.corerl)
client.start_consumer()
```

### Key Differences

| Legacy BaseEventBus | New EventBusClient |
|---------------------|-------------------|
| Dual ports (pub/sub) | Single port |
| XPUB/XSUB sockets | DEALER/ROUTER |
| Bind pattern | Connect pattern |
| Pub-sub only | Pub-sub + request-reply |
| No service registration | Service ID registration |
| No correlation tracking | Built-in correlation IDs |

Update `pyproject.toml`:

```toml
dependencies = [
    "lib-events",
    # ...
]

[tool.uv.sources]
lib-events = { path = "../libs/lib_events", editable = true }
```

# lib_events

Event bus infrastructure for RLCore microservices using ZeroMQ pub-sub pattern.

## Overview

`lib_events` provides a centralized event bus implementation for inter-service communication in the RLCore monorepo. It includes both client and server components for reliable message passing using ZeroMQ's publish-subscribe pattern.

## Components

### Client (`lib_events.client`)

- **`EventBusClient`**: Generic, type-safe event bus client with automatic reconnection and topic-based subscription callbacks

### Server (`lib_events.server`)

- **`EventBusProxy`**: ZMQ XPUB/XSUB message forwarder (broker)
- **`EventBusManager`**: Lifecycle manager for EventBusProxy with health monitoring

### Legacy (`lib_events.legacy`)

⚠️ **Deprecated components** - Use `EventBusClient` instead for new code

- **`BaseEventBus`**: Legacy event bus with bound sockets (deprecated)
- **`consumer_task`**, **`Clock`**: Supporting utilities for `BaseEventBus`

## Quick Start

### Using EventBusClient

```python
from lib_events.client.event_bus_client import EventBusClient
from lib_defs.type_defs.base_events import BaseEvent, BaseEventType, BaseEventTopic

# Initialize client
client = EventBusClient[BaseEvent, BaseEventType, BaseEventTopic](
    publisher_endpoint="tcp://localhost:5570",
    subscriber_endpoint="tcp://localhost:5571",
    event_class=BaseEvent,
    event_type_class=BaseEventType,
    event_topic_class=BaseEventTopic,
)

# Register callbacks for specific topics
def handle_message(event: BaseEvent):
    print(f"Received: {event}")

client.register_callback(BaseEventTopic.GLOBAL, handle_message)

# Connect and start
client.connect()
client.start()

# Publish events
event = BaseEvent(event_type=BaseEventType.INFO, message="Hello")
client.publish(event, BaseEventTopic.GLOBAL)

# Cleanup
client.stop()
client.disconnect()
```

### Running Event Bus Server

```python
from lib_events.server.manager import EventBusManager

manager = EventBusManager(
    host="*",
    pub_port=5570,  # Port where publishers connect
    sub_port=5571,  # Port where subscribers connect
)

manager.start()
# ... server runs ...
manager.stop()
```

## Architecture

The event bus uses a **proxy pattern** (XPUB/XSUB) to decouple publishers from subscribers:

```
Publishers → [XSUB] → EventBusProxy → [XPUB] → Subscribers
             :5570                      :5571
```

- **XSUB socket**: Receives published messages from all publishers
- **XPUB socket**: Forwards messages to subscribed clients based on topic filters
- **Benefits**: Dynamic topology, no single point of failure, automatic topic filtering

## Testing

Testing utilities and fixtures are provided for both library tests and consumer service tests.

### Example Test Fixture

```python
import pytest
from lib_events.client.event_bus_client import EventBusClient

@pytest.fixture
def event_bus_client():
    client = EventBusClient[BaseEvent, BaseEventType, BaseEventTopic](
        publisher_endpoint="tcp://localhost:5570",
        subscriber_endpoint="tcp://localhost:5571",
        event_class=BaseEvent,
        event_type_class=BaseEventType,
        event_topic_class=BaseEventTopic,
    )
    client.connect()
    client.start()
    yield client
    client.stop()
    client.disconnect()
```

## Dependencies

- **lib-defs**: Event type definitions (`BaseEvent`, `BaseEventType`, `BaseEventTopic`)
- **pydantic**: Event serialization and validation
- **pyzmq**: ZeroMQ bindings for message transport

## Migration from lib_utils

If you're migrating from the old `lib_utils.messages` location:

```python
# Old import
from lib_utils.messages.event_bus_client import EventBusClient

# New import
from lib_events.client.event_bus_client import EventBusClient
```

Update `pyproject.toml`:

```toml
dependencies = [
    "lib-events",
    # ...
]

[tool.uv.sources]
lib-events = { path = "../libs/lib_events", editable = true }
```

# Coredinator

Coredinator is a FastAPI-based orchestration layer for all RLTune microservices.
One-off microservices --- such as coretelemetry or coregateway --- are managed directly by coredinator.
Per-agent services --- such as corerl and coreio --- are managed as a bundle called an `Agent/` through the `AgentManager` interface within coredinator.

## Getting Started

### Prerequisites

- Python 3.13.0
- uv package manager
- Microservice executables in the base path

### Installation

1. Install dependencies:
```bash
uv sync
```

### Running with Structured Logging

Coredinator includes comprehensive structured logging with file rotation. Basic usage:

```bash
# Run with default console logging
uv run python -m coredinator.app --base-path /path/to/executables

# Run with file logging and rotation
uv run python -m coredinator.app --base-path /path/to/executables --log-file /path/to/logs/coredinator.log

# Run with debug level logging
uv run python -m coredinator.app --base-path /path/to/executables --log-file /path/to/logs/coredinator.log --log-level DEBUG

# Run with file logging only (no console output)
uv run python -m coredinator.app --base-path /path/to/executables --log-file /path/to/logs/coredinator.log --no-console

# Run with auto-reload enabled (useful for development)
uv run python -m coredinator.app --base-path /path/to/executables --reload
```

Log files will automatically rotate when they reach 10MB (keeping 5 backup files by default) and output structured JSON logs for easy parsing and analysis.

2. Run the service:
```bash
uv run python coredinator/app.py --base-path /path/to/executables
```

The service will start on `http://localhost:7000` with interactive API documentation available at `/docs`.

Optionally, you can specify a custom port:
```bash
uv run python coredinator/app.py --base-path /path/to/executables --port 9000
```

For development with auto-reload:
```bash
uv run python coredinator/app.py --base-path /path/to/executables --reload
```

### Configuration

The service requires a `--base-path` argument pointing to the directory containing CoreIO and CoreRL executables. Additionally, you can specify the port using the `--port` argument (default: 7000). Agent configurations are provided as YAML files when starting agents.

Optional configuration:
- `--port`: Specify the port number (default: 7000)
- `--event-bus-host`: Event bus host address (default: *)
- `--event-bus-pub-port`: Event bus publisher port (default: 5559)
- `--event-bus-sub-port`: Event bus subscriber port (default: 5560)

## Architecture

### Design Principles

1. **Service Orchestration**: Centralized management of distributed RL services
2. **State Persistence**: SQLite database ensures agent state survives restarts
3. **Process Management**: Robust lifecycle management with graceful shutdown
4. **API-First**: RESTful interface for programmatic control
5. **Fault Tolerance**: Recovery mechanisms and error handling
6. **Event-Driven Communication**: ZeroMQ-based event bus for decoupled inter-service messaging

### System Components

```
┌─────────────────┐
│   REST API      │
│  (FastAPI)      │
├─────────────────┤
│  AgentManager   │
│  (Orchestrator) │
├─────────────────┤
│  Event Bus      │
│  (ZMQ XPUB/XSUB)│
├─────────────────┤
│   Database      │
│   (SQLite)      │
└─────────────────┘
        │
        ├── CoreIO Service
        └── CoreRL Service
```

**Key Components:**

- **AgentManager**: Core orchestration logic for agent lifecycle management
- **REST API**: HTTP endpoints for internal communication across microservices
- **Event Bus**: ZeroMQ proxy enabling publish-subscribe messaging between services
- **Service Protocols**: Standardized interfaces for service communication
- **Database Layer**: Persistent storage for agent state and configuration

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/agents/start` | Start a new agent with configuration |
| `POST` | `/api/agents/{agent_id}/stop` | Stop a running agent (stops CoreRL only) |
| `GET` | `/api/agents/{agent_id}/status` | Get agent status information |
| `GET` | `/api/agents/` | List all managed agents |
| `POST` | `/api/coreio/start` | Start a CoreIO service instance |
| `POST` | `/api/coreio/{coreio_id}/stop` | Stop a CoreIO service instance |
| `GET` | `/api/coreio/{coreio_id}/status` | Get CoreIO service status |
| `GET` | `/api/coreio/` | List all CoreIO services |
| `GET` | `/api/healthcheck` | Service health check |
| `POST` | `/api/agents/demo/tep/start` | Start a TEP demo agent with a demo configuration |
| `POST` | `/api/agents/demo/tep/{agent_id}/stop` | Stop a running TEP demo agent |
| `GET` | `/api/agents/demo/tep/{agent_id}/status` | Get status for a TEP demo agent |

### Example Usage

**Start an Agent:**
```bash
curl -X POST "http://localhost:7000/api/agents/start" \
  -H "Content-Type: application/json" \
  -d '{"config_path": "/path/to/config.yaml"}'
```

**Start an Agent with Shared CoreIO:**
```bash
curl -X POST "http://localhost:7000/api/agents/start" \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "/path/to/config.yaml",
    "coreio_id": "shared-service-id"
  }'
```

**Check Agent Status:**
```bash
curl -X GET "http://localhost:7000/api/agents/{agent_id}/status"
```

**Stop an Agent:**
```bash
curl -X POST "http://localhost:7000/api/agents/{agent_id}/stop"
```

**Stop a CoreIO Service:**
```bash
curl -X POST "http://localhost:7000/api/coreio/{coreio_id}/stop"
```

**Check CoreIO Service Status:**
```bash
curl -X GET "http://localhost:7000/api/coreio/{coreio_id}/status"
```

**List All CoreIO Services:**
```bash
curl -X GET "http://localhost:7000/api/coreio/"
```

## Service Sharing

Coredinator supports sharing CoreIO services between multiple agents to optimize resource usage and enable coordinated scenarios. When starting an agent, you can optionally specify a `coreio_id` to share a CoreIO instance with other agents.

**Important:** CoreIO services have independent lifecycles from agents. Stopping an agent only stops its CoreRL service, not the CoreIO service. You must explicitly stop CoreIO services using the CoreIO API endpoints.

### Usage

**Start Multiple Agents with Shared CoreIO:**

```bash
# Start first agent with shared CoreIO
curl -X POST "http://localhost:7000/api/agents/start" \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "/path/to/config1.yaml",
    "coreio_id": "shared-coreio-instance"
  }'

# Start second agent sharing the same CoreIO
curl -X POST "http://localhost:7000/api/agents/start" \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "/path/to/config2.yaml",
    "coreio_id": "shared-coreio-instance"
  }'
```

**Start Agent with Independent Services (Default):**

```bash
# Without coreio_id, agent gets its own CoreIO instance
curl -X POST "http://localhost:7000/api/agents/start" \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "/path/to/config.yaml"
  }'
```

### Request Schema

The `/api/agents/start` endpoint accepts the following parameters:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `config_path` | `string` | Yes | Path to the agent's YAML configuration file |
| `coreio_id` | `string` | No | Optional ID for sharing CoreIO service with other agents |

### Service Lifecycle

- **Agent Stopping**: Stopping an agent only stops its CoreRL service. CoreIO services must be stopped independently via the CoreIO API.
- **Shared Services**: When multiple agents share a CoreIO service (by specifying the same `coreio_id`), they all use the same service instance.
- **Service Starting**: Starting an agent will start all dependent services (CoreRL → CoreIO) if they are not already running. Service starts are idempotent.
- **Independent CoreRL**: Each agent always gets its own CoreRL service regardless of CoreIO sharing.

## Event Bus

Coredinator runs a ZeroMQ-based event bus using XPUB/XSUB proxy architecture. This enables decoupled publish-subscribe communication between services without requiring direct point-to-point connections.

### Architecture

The event bus consists of a central proxy that runs in a background thread:

- **XSUB socket** (default: `tcp://*:5559`): Publishers connect here to send events
- **XPUB socket** (default: `tcp://*:5560`): Subscribers connect here to receive events

The proxy automatically forwards messages from publishers to subscribers based on topic subscriptions. All communication happens asynchronously in a dedicated thread, ensuring the main coredinator service remains non-blocking.

The event bus host and ports can be configured via CLI arguments (`--event-bus-host`, `--event-bus-pub-port`, `--event-bus-sub-port`) to support different deployment scenarios or avoid port conflicts.

### Publisher Usage

Services that want to publish events should connect a ZMQ `PUB` socket to the proxy's XSUB endpoint:

```python
import zmq

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.connect("tcp://localhost:5559")

# Publish message on topic "agent.started"
socket.send_multipart([b"agent.started", b'{"agent_id": "abc123"}'])
```

### Subscriber Usage

Services that want to receive events should connect a ZMQ `SUB` socket to the proxy's XPUB endpoint and subscribe to desired topics:

```python
import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5560")

# Subscribe to all agent events
socket.setsockopt_string(zmq.SUBSCRIBE, "agent.")

while True:
    topic, message = socket.recv_multipart()
    print(f"Received: {topic.decode()} - {message.decode()}")
```

### Event Bus Health

The event bus status is included in the `/api/healthcheck` endpoint response:

```json
{
  "status": "healthy",
  "process_id": 12345,
  "service": "coredinator",
  "version": "0.0.1",
  "event_bus": {
    "status": "running",
    "config": {
      "xsub_addr": "tcp://*:5559",
      "xpub_addr": "tcp://*:5560",
      "publisher_endpoint": "tcp://localhost:5559",
      "subscriber_endpoint": "tcp://localhost:5560"
    }
  }
}
```

### Benefits

- **Decoupling**: Services only need to know the proxy address, not each other
- **Scalability**: Multiple publishers and subscribers can connect without coordination
- **Flexibility**: New services can join or leave without affecting existing services
- **Non-blocking**: Proxy runs in background thread, no impact on main service

## TEP Demo: AgentManager demo routes

The TEP demo endpoints provide a quick way to start, inspect, and stop a demo agent used for the TEP showcase. These endpoints are implemented in the `AgentManager` as a convenience wrapper around the regular agent lifecycle APIs. They do not embed special/demo configs; under the hood they follow the same configuration loading and validation as the standard agent start flow and therefore may require a valid config path or standard parameters depending on the runtime implementation.

Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/agents/demo/tep/start` | Start a TEP demo agent. Returns the new `agent_id` and metadata. |
| `POST` | `/api/agents/demo/tep/{agent_id}/stop` | Stop a running TEP demo agent. |
| `GET` | `/api/agents/demo/tep/{agent_id}/status` | Get status for a TEP demo agent. |

Start request example (may require a config or parameters depending on implementation):

```bash
curl -X POST "http://localhost:7000/api/agents/demo/tep/start" \
  -H "Content-Type: application/json" \
  -d '{"config_path": "/path/to/demo_config.yaml"}'
```

Successful start response (JSON, illustrative):

```json
{
  "agent_id": "tep-demo-2025-09-04-001",
  "status": "starting",
  "ports": {
    "coreio": 5001,
    "corerl": 5002
  },
  "config": "/path/to/demo_config.yaml"
}
```

Stop request example:

```bash
curl -X POST "http://localhost:7000/api/agents/demo/tep/tep-demo-2025-09-04-001/stop"
```

Status response example:

```json
{
  "agent_id": "tep-demo-2025-09-04-001",
  "status": "running",
  "uptime_seconds": 42,
  "ports": {
    "coreio": 5001,
    "corerl": 5002
  }
}
```

Notes

- These demo endpoints are intended for local demos and CI smoke tests only. They are convenience wrappers and should be used in trusted environments.
- The demo routes use the same config-loading/validation pathway as the standard `/api/agents/start` flow. If your runtime requires a config path or parameters, pass them in the request.
- The returned `agent_id` follows a predictable pattern but should be treated as opaque by callers.
- If you need fine-grained control (different seeds, port offsets, or non-demo configs), use `/api/agents/start` with an explicit YAML config.

## Development

### Project Structure

```
coredinator/
├── coredinator/
│   ├── app.py             # FastAPI application entry point
│   ├── agent/             # Agent orchestration
│   ├── web/               # REST API routes
│   ├── service/           # Generic service execution
│   ├── services/          # Service wrappers for specific internal services
│   └── utils/             # Generic code utilities
└── tests/
    ├── small/             # Fast functionality tests
    ├── medium/            # Workflow tests involving fake external services
    └── large/             # E2E workflow tests mocking real external interactions
```

### Running Tests

Execute the test suite with different levels of integration:

```bash
# Run all tests
uv run pytest

# Run only unit tests
uv run pytest tests/small/

# Run integration tests
uv run pytest tests/medium/

# Run end-to-end tests
uv run pytest tests/large/

# Run with coverage
uv run pytest --cov=coredinator
```

### Development Workflow

1. **Code Style**: Follow project standards defined in `../standards/python/`
2. **Type Checking**: Use `pyright` for static type analysis
3. **Linting**: Use `ruff` for code formatting and linting
4. **Testing**: Write tests for new functionality following the three-tier structure

### Debugging

The service includes comprehensive logging via the `uvicorn` logger. Set log levels in the application configuration:

```python
import logging
logging.getLogger("uvicorn.error").setLevel(logging.DEBUG)
```

## Contributing

1. Follow the coding standards defined in the project guidelines
2. Write tests for new functionality using the three-tier approach
3. Ensure all tests pass before submitting changes
4. Use meaningful commit messages following conventional commits

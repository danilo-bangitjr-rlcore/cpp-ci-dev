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

2. Run the service:
```bash
uv run python coredinator/app.py --base-path /path/to/executables
```

The service will start on `http://localhost:7000` with interactive API documentation available at `/docs`.

Optionally, you can specify a custom port:
```bash
uv run python coredinator/app.py --base-path /path/to/executables --port 9000
```

### Configuration

The service requires a `--base-path` argument pointing to the directory containing CoreIO and CoreRL executables. Additionally, you can specify the port using the `--port` argument (default: 7000). Agent configurations are provided as YAML files when starting agents.

Optional configuration:
- `--port`: Specify the port number (default: 7000)

## Architecture

### Design Principles

1. **Service Orchestration**: Centralized management of distributed RL services
2. **State Persistence**: SQLite database ensures agent state survives restarts
3. **Process Management**: Robust lifecycle management with graceful shutdown
4. **API-First**: RESTful interface for programmatic control
5. **Fault Tolerance**: Recovery mechanisms and error handling

### System Components

```
┌─────────────────┐
│   REST API      │
│  (FastAPI)      │
├─────────────────┤
│  AgentManager   │
│  (Orchestrator) │
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
- **Service Protocols**: Standardized interfaces for service communication
- **Database Layer**: Persistent storage for agent state and configuration

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/agents/start` | Start a new agent with configuration |
| `POST` | `/api/agents/{agent_id}/stop` | Stop a running agent |
| `GET` | `/api/agents/{agent_id}/status` | Get agent status information |
| `GET` | `/api/agents/` | List all managed agents |
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

**Check Agent Status:**
```bash
curl -X GET "http://localhost:7000/api/agents/{agent_id}/status"
```

**Stop an Agent:**
```bash
curl -X POST "http://localhost:7000/api/agents/{agent_id}/stop"
```

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

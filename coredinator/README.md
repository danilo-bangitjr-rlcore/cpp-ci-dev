
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

The service will start on `http://localhost:8000` with interactive API documentation available at `/docs`.

### Configuration

The service requires a `--base-path` argument pointing to the directory containing CoreIO and CoreRL executables. Agent configurations are provided as YAML files when starting agents.

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

### Example Usage

**Start an Agent:**
```bash
curl -X POST "http://localhost:8000/api/agents/start" \
  -H "Content-Type: application/json" \
  -d '{"config_path": "/path/to/config.yaml"}'
```

**Check Agent Status:**
```bash
curl -X GET "http://localhost:8000/api/agents/{agent_id}/status"
```

**Stop an Agent:**
```bash
curl -X POST "http://localhost:8000/api/agents/{agent_id}/stop"
```

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

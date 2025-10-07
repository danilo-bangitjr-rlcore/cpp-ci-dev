# CoreTelemetry

CoreTelemetry is a FastAPI-based telemetry service that provides a REST API for querying time-series telemetry data from PostgreSQL/TimescaleDB databases. The service reads agent configurations from YAML files and exposes endpoints for retrieving metrics and managing configurations.

## Features

- **Time-series data queries** - Query telemetry data with optional time ranges
- **Agent-based configuration** - YAML-based configuration per agent
- **Automatic NULL filtering** - Returns only non-NULL values by default
- **Cache management** - Clear configuration cache without restart
- **UTC timezone support** - Automatically assumes UTC for timestamps without timezone info
- **CORS enabled** - Ready for cross-origin requests

## Requirements

### Dependencies

- **Python 3.13**
- **PostgreSQL/TimescaleDB** - Running instance with telemetry data
- **Python packages** - Installed via `uv` (see `pyproject.toml`)

### Database Setup

The service requires a running PostgreSQL or TimescaleDB instance with:
- Tables containing time-series telemetry data
- Each table must have a `time` column (timestamp)
- Metric columns (e.g., `temperature`, `pressure`, etc.)

## Installation

1. Clone the repository
2. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

## Configuration

### Agent Configuration Files

Create YAML configuration files in a directory (default: `clean/`) with the format `{agent_id}.yaml`:

```yaml
metrics:
  table_name: test_metrics
```

### Database Configuration

Configure the database connection via the API endpoints (see `POST /api/v1/telemetry/config/db` below). Default connection settings are:
- Host: `localhost:5432`
- Database: `postgres`
- User: `postgres`
- Password: `password`
- Schema: `public`

## Running the Service

### Basic Usage

```bash
uv run python coretelemetry/app.py
```

### With Custom Config Path

```bash
uv run python coretelemetry/app.py --config-path /path/to/config/dir
```


### Command Line Arguments

- `--config-path` - Path to the configuration directory containing agent YAML files (default: `clean/`)
- `--port` - Port to run the server on (default: `8001`)

## API Endpoints

### Data Endpoints

#### `GET /api/v1/telemetry/data/{agent_id}`
Get telemetry data for a specific agent and metric.

**Query Parameters:**
- `metric` (required) - Name of the metric column
- `start_time` (optional) - Start timestamp (UTC assumed if no timezone)
- `end_time` (optional) - End timestamp (UTC assumed if no timezone)

**Response:** List of `{"timestamp": ..., "value": ...}` objects

**Notes:**
- Returns only non-NULL values
- If no time parameters: returns latest value only (LIMIT 1)
- Ordered by time DESC

#### `GET /api/v1/telemetry/data/{agent_id}/metrics`
Get all available metrics for a specific agent.

**Response:** `{"agent_id": "...", "data": ["metric1", "metric2", ...]}`

**Notes:**
- Excludes the `time` column from results

### Configuration Endpoints

#### `GET /api/v1/telemetry/config/db`
Get current database configuration.

**Response:** DBConfig object with connection details

#### `POST /api/v1/telemetry/config/db`
Update database configuration.

**Request Body:** DBConfig JSON object

#### `GET /api/v1/telemetry/config/path`
Get current configuration path.

**Response:** `{"config_path": "..."}`

#### `POST /api/v1/telemetry/config/path`
Update configuration path.

**Query Parameters:**
- `path` (required) - New configuration directory path

#### `POST /api/v1/telemetry/config/clear_cache`
Clear all cached data (YAML configs, SQL reader).

**Response:** `{"message": "Cache cleared successfully"}`

### Utility Endpoints

#### `GET /`
Redirects to `/docs` (Swagger UI)

#### `GET /health`
Health check endpoint.

**Response:** `{"status": "healthy"}`

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test Suites

```bash
# Small tests (unit tests with mocks)
pytest tests/small/ -v

# Medium tests (integration tests with real database)
pytest tests/medium/ -v

# Large tests (end-to-end API tests)
pytest tests/large/ -v
```

### Run Specific Test File

```bash
pytest tests/small/test_telemetry_manager.py -v
```

### Test Coverage

```bash
pytest --cov=coretelemetry --cov-report=html
```

### Test Requirements

Medium and large tests require:
- Docker (for PostgreSQL/TimescaleDB test containers)
- `test.infrastructure` plugins for database fixtures

## Architecture

### Project Structure

```
coretelemetry/
├── coretelemetry/
│   ├── app.py              # FastAPI application
│   ├── services.py         # TelemetryManager service
│   └── utils/
│       └── sql.py          # SqlReader and DBConfig
├── tests/
│   ├── small/              # Unit tests (mocked)
│   ├── medium/             # Integration tests (real DB)
│   └── large/              # E2E API tests
└── README.md
```

### Key Components

- **TelemetryManager** - Orchestrates configuration loading and data retrieval
- **SqlReader** - Handles database queries and connection management
- **DBConfig** - Database connection configuration dataclass

## Development

### Code Style

The project uses:
- `ruff` for linting and formatting
- Type hints throughout
- Docstrings for all public methods

### Testing Philosophy

- **Small tests** - Fast unit tests with mocked dependencies
- **Medium tests** - Integration tests with real PostgreSQL
- **Large tests** - Full E2E tests with FastAPI TestClient

## Error Handling

The API returns appropriate HTTP status codes:
- `200` - Success
- `400` - Bad request (e.g., reserved column name)
- `404` - Not found (table, column, or no data)
- `413` - Payload too large (>5000 rows)
- `500` - Server error (config file issues)
- `503` - Service unavailable (database connection failed)

## License

[Add license information]

## Contributing

[Add contribution guidelines]

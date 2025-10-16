# CoreGateway

CoreGateway is a FastAPI-based API gateway that serves as a proxy layer that forwards REST API requests to internal services.

## Overview

- **Proxying**: Forwards all requests to internal services (Coredinator and CoreTelemetry), which handle internal routing and service resolution.
- **Error Handling**: Normalizes upstream errors and provides clear status codes to clients.
- **Observability**: Adds version headers and structured logging for traceability.
- **Health Monitoring**: Checks downstream service availability.

## Requirements

- **Python 3.13.0**
- **uv** package manager
- Coredinator service running and accessible at `localhost:7000`
- CoreTelemetry service running and accessible at `localhost:7001`

## Installation

1. Install dependencies:
   ```bash
   uv sync
   ```

## Running the Service

Start the gateway on the default port (8001):

```bash
uv run python coregateway/app.py
```

The service will start on `http://localhost:8001` with interactive API documentation available at `/docs`.

## API Endpoints

| Method | Endpoint                          | Description                                    |
|--------|-----------------------------------|------------------------------------------------|
| `GET`  | `/`                               | Redirects to `/docs` (Swagger UI)             |
| `GET`  | `/health`                         | Health check (checks all services)            |
| `GET`  | `/api/v1/coredinator/{path}`      | Proxy GET requests to Coredinator             |
| `POST` | `/api/v1/coredinator/{path}`      | Proxy POST requests to Coredinator            |
| `GET`  | `/api/v1/coretelemetry/{path}`    | Proxy GET requests to CoreTelemetry           |
| `POST` | `/api/v1/coretelemetry/{path}`    | Proxy POST requests to CoreTelemetry          |

NOTE: The request forwarding currently only handles GETs & POSTs because the backend services only expose these methods.

### Health Check Response

```json
{
  "status": "healthy",
  "services": {
    "coredinator": "healthy",
    "coretelemetry": "healthy"
  }
}
```

Returns HTTP 200 if all services are reachable, HTTP 503 if any service is degraded.

## Configuration

The service uses the following default settings:

- **Port**: 8001
- **Coredinator URL**: `http://localhost:7000`
- **CoreTelemetry URL**: `http://localhost:7001`
- **HTTP Timeouts**:
  - Connect: 5.0s
  - Read: 30.0s
  - Write: 10.0s
  - Pool: 5.0s
- **Connection Limits**:
  - Max keep-alive connections: 20
  - Max total connections: 50
  - Keep-alive expiry: 30.0s
- **Retries**: 3 attempts for transient failures

### Command Line Arguments

```bash
python coregateway/app.py --port 8001 --coredinator-port 7000 --coretelemetry-port 7001
```

## Development

### Project Structure

```
coregateway/
├── coregateway/
│   ├── app.py                     # FastAPI application entry point
│   ├── coredinator_proxy.py       # Proxy router for Coredinator
│   ├── coretelemetry_proxy.py     # Proxy router for CoreTelemetry
│   └── proxy_utils.py             # Shared proxy utilities and error handling
├── tests/
│   └── medium/
│       ├── test_coregateway_app.py      # Gateway and health check tests
│       ├── test_coredinator_proxy.py    # Coredinator proxy tests
│       └── test_coretelemetry_proxy.py  # CoreTelemetry proxy tests
└── README.md
```

### Key Components

- **Proxy Routers**: Handle request forwarding for each backend service
- **Proxy Utils**: Shared utilities for header management, error handling, and request proxying
- **HTTP Client**: Configured with timeouts, retries, and connection pooling
- **Error Handling**: Maps HTTP exceptions to appropriate status codes
- **Header Management**: Strips hop-by-hop headers for proper proxying

### Testing

```bash
uv run pytest
```

### Code Style

```bash
uv run ruff check .
uv run pyright
```

## Error Handling

The gateway returns appropriate HTTP status codes:

- `200` - Success
- `502` - Bad gateway (upstream error or connection failure)
- `503` - Service unavailable (one or more services unreachable in health check)
- `504` - Gateway timeout (upstream service timeout)
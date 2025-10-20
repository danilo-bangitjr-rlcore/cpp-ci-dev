# CoreGateway

CoreGateway is a FastAPI-based API gateway that serves as a proxy layer that forwards REST API requests to internal services 

## Overview

- **Proxying**: Forwards all requests to internal services (currently only Coredinator), which handles internal routing and service resolution.
- **Error Handling**: Normalizes upstream errors and provides clear status codes to clients.
- **Observability**: Adds version headers and structured logging for traceability.
- **Health Monitoring**: Checks downstream service availability.

## Requirements

- **Python 3.13.0**
- **uv** package manager
- Coredinator service running and accessible at `localhost:7000`

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

| Method | Endpoint                     | Description                                    |
|--------|------------------------------|------------------------------------------------|
| `GET`  | `/`                          | Redirects to `/docs` (Swagger UI)             |
| `GET`  | `/health`                    | Health check (checks Coredinator status)      |
| `GET`  | `/api/v1/coredinator/{path}` | Proxy GET requests to Coredinator             |
| `POST` | `/api/v1/coredinator/{path}` | Proxy POST requests to Coredinator            |


NOTE: The request forwarding to Coredinator currently only handles GETs & POSTs because Coredinator only contains these types of requests

### Health Check Response

```json
{
  "status": "healthy",
  "services": {
    "coredinator": "healthy"
  }
}
```

Returns HTTP 200 if Coredinator is reachable, HTTP 503 if degraded.

## Configuration

The service uses the following default settings:

- **Port**: 8001
- **Coredinator URL**: `http://localhost:7000`
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

## Development

### Project Structure

```
coregateway/
├── coregateway/
│   ├── app.py                   # FastAPI application entry point
│   └── coredinator_proxy.py     # Proxy logic for forwarding requests
├── tests/
└── README.md
```

### Key Components

- **Proxy Router**: Handles request forwarding with proper header management
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
- `503` - Service unavailable (Coredinator unreachable in health check)
- `504` - Gateway timeout (upstream service timeout)

---
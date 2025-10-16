import argparse
import json
from collections.abc import Callable
from contextlib import asynccontextmanager

import httpx
import uvicorn
from coregateway.coredinator_proxy import coredinator_router
from coregateway.coretelemetry_proxy import coretelemetry_router
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from lib_instrumentation.logging import get_structured_logger

version = "0.0.1"

def parse_args():
    parser = argparse.ArgumentParser(description="CoreGateway Service")
    parser.add_argument("--port", type=int, default=8001, help="Port to run CoreGateway")
    parser.add_argument("--coredinator-port", type=int, default=7000, help="Port for coredinator service")
    parser.add_argument("--coretelemetry-port", type=int, default=7001, help="Port for coredinator service")
    args = parser.parse_args()
    return args.port, args.coredinator_port, args.coretelemetry_port

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Define timeouts
    # Connect timeout: time to establish a connection to the server
    # Read timeout: time to wait for a response from the server
    # Write timeout: time to send the request to the server
    # Pool timeout: time to wait for a connection from the pool
    timeout = httpx.Timeout(
        connect=5.0,
        read=30.0,
        write=10.0,
        pool=5.0,
    )

    # Define connection limits
    # Max keep-alive connections: maximum number of connections to keep alive
    # Max connections: maximum number of connections in total
    # Keep-alive expiry: time to keep a connection alive in the pool
    limits = httpx.Limits(
        max_keepalive_connections=20,
        max_connections=50,
        keepalive_expiry=30.0,
    )

    # Retry transient failures (502, 503, 504, connection errors)
    transport = httpx.AsyncHTTPTransport(
        retries=3,
        limits=limits,
    )

    app.state.httpx_client = httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        transport=transport,
        follow_redirects=False,
    )

    # Use logger from app state
    logger = app.state.logger
    logger.info(f"CoreGateway starting up - port={app.state.port}, version={version}")

    yield

    # Clean up resources
    await app.state.httpx_client.aclose()
    logger.info("CoreGateway shutting down")


def create_app(port: int = 8001, coredinator_port: int = 7000, coretelemetry_port: int = 7001) -> FastAPI:
    """Factory function to create FastAPI app."""
    app = FastAPI(lifespan=lifespan, title="CoreGateway API")
    app.state.port = port
    app.state.coredinator_port = coredinator_port
    app.state.coredinator_base = f"http://localhost:{coredinator_port}"
    # Create logger and store in app state
    app.state.logger = get_structured_logger("coregateway")
    app.state.coretelemetry_port = coretelemetry_port
    app.state.coretelemetry_base = f"http://localhost:{coretelemetry_port}"

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_core_rl_version(request: Request, call_next: Callable):
        client_host = request.client.host if request.client else "unknown"
        app.state.logger.debug(
            f"Processing request - method={request.method}, "
            f"url={request.url!s}, client={client_host}",
        )

        response = await call_next(request)
        response.headers["X-CoreRL-Version"] = version

        app.state.logger.debug(
            f"Request completed - method={request.method}, "
            f"url={request.url!s}, status={response.status_code}",
        )

        return response

    @app.get("/")
    async def redirect():
        return RedirectResponse(url="/docs")

    @app.get("/health")
    async def health_check():
        # Check if we can reach coredinator
        try:
            client: httpx.AsyncClient = app.state.httpx_client
            resp = await client.get(
                f"{app.state.coredinator_base}/api/healthcheck",
                timeout=2.0,
            )
            coredinator_healthy = resp.status_code == 200
        except Exception:
            coredinator_healthy = False

        status = "healthy" if coredinator_healthy else "degraded"
        status_code = 200 if coredinator_healthy else 503

        return Response(
            content=json.dumps({
                "status": status,
                "services": {
                    "coredinator": "healthy" if coredinator_healthy else "unhealthy",
                },
            }),
            status_code=status_code,
            media_type="application/json",
        )

    app.include_router(coredinator_router, prefix="/api/v1/coredinator")
    app.include_router(coretelemetry_router, prefix="/api/v1/coretelemetry")

    return app

if __name__ == "__main__":
    port, coredinator_port, coretelemetry_port = parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.setLevel(logging.DEBUG)

    app = create_app(port, coredinator_port, coretelemetry_port)
    uvicorn.run(app, host="0.0.0.0", port=port)

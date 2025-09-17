import argparse
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from coredinator.agent.agent_manager import AgentManager
from coredinator.service.service_manager import ServiceManager
from coredinator.web.agent_manager import router as agent_manager

# For debugging while running the server
_log = logging.getLogger("uvicorn.error")
_log.setLevel(logging.INFO)

logger = logging.getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CoreRL server is starting up.")
    yield
    logger.info("CoreRL server is shutting down.")


def parse_args():
    parser = argparse.ArgumentParser(description="Coredinator Service")
    parser.add_argument("--base-path", type=Path, required=True, help="Path to microservice executables")
    parser.add_argument("--port", type=int, default=7000, help="Port to run the service on (default: 7000)")
    args = parser.parse_args()

    return args.base_path, args.port


base_path, main_port = parse_args()
app = FastAPI(lifespan=lifespan)
service_manager = ServiceManager()
app.state.service_manager = service_manager
app.state.base_path = base_path
app.state.agent_manager = AgentManager(base_path=base_path, service_manager=service_manager)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

version = "0.0.0"


@app.middleware("http")
async def add_core_rl_version(request: Request, call_next: Callable):
    response = await call_next(request)
    response.headers["X-CoreRL-Version"] = version
    return response


@app.get("/")
async def redirect():
    return RedirectResponse(url="/docs")


@app.get("/api/healthcheck")
async def health():
    ...


app.include_router(agent_manager, prefix="/api/agents", tags=["Agent"])


if __name__ == "__main__":
    # Re-parse args for __main__ execution
    _, main_port = parse_args()
    uvicorn.run("coredinator.app:app", host="0.0.0.0", port=main_port, reload=True)

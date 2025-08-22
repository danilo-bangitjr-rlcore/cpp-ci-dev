import argparse
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from coredinator.agent.agent_manager import AgentManager
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


def parse_base_path():
    parser = argparse.ArgumentParser(description="Coredinator Service")
    parser.add_argument("--base-path", type=Path, required=True, help="Path to microservice executables")
    args, _ = parser.parse_known_args()
    return args.base_path


base_path = parse_base_path()
app = FastAPI(lifespan=lifespan)
app.state.agent_manager = AgentManager(base_path=base_path)

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

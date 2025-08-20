import importlib.metadata
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

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

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    version = importlib.metadata.version("corerl")
except Exception:
    logger.exception("Failed to determine corerl version")
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

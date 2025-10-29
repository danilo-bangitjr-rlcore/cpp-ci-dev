import argparse
import os
import sys
from contextlib import asynccontextmanager
from typing import NamedTuple

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from server.config_api.config_routes import config_router
from server.opc_api.opc_routes import opc_router


class CoreUIConfig(NamedTuple):
    port: int
    dist_path: str | None
    reload: bool

def parse_args():
    parser = argparse.ArgumentParser(description="CoreUI Service")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000)")
    parser.add_argument("--dist-path", type=str, help="Path to the frontend distribution directory")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    return CoreUIConfig(
        port=args.port,
        dist_path=args.dist_path,
        reload=args.reload,
    )

def create_app(dist_path: str | None = None) -> FastAPI:
    """Factory function to create FastAPI app with optional dist path."""
    core_ui = CoreUI(dist_path=dist_path)
    return core_ui.get_app()


class CoreUI:
    def __init__(self, dist_path: str | None = None):
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            app.state.httpx_client = httpx.AsyncClient()
            yield # marks the point between startup and shutdown
            await app.state.httpx_client.aclose()

        self.app = FastAPI(lifespan=lifespan)

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:3000",
                "http://localhost:5173",
                "http://localhost:4173",
                "http://localhost:8000",
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.app.include_router(opc_router, prefix="/api/v1/opc")
        self.app.include_router(config_router, prefix="/api/v1/config")

        if dist_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            dist_path = os.path.join(base_dir, "client", "dist")

        meipass_path = getattr(sys, '_MEIPASS', None)
        if meipass_path:
            dist_path = os.path.join(meipass_path, "dist")

        dist_path = os.path.abspath(dist_path)
        assets_path = os.path.join(dist_path, "assets")
        if os.path.exists(assets_path):
            self.app.mount("/app/assets", StaticFiles(directory=assets_path), name="assets")

        @self.app.get("/api/health")
        async def health_check():
            return {"status": "ok"}

        index_html = os.path.join(dist_path, "index.html")

        @self.app.get("/app")
        async def spa_root():
            return FileResponse(index_html)

        @self.app.get("/app/{full_path:path}")
        async def spa_catch_all(full_path: str):
            return FileResponse(index_html)

    def get_app(self):
        return self.app


def get_app() -> FastAPI:
    coreui_config = parse_args()
    return create_app(coreui_config.dist_path)


if __name__ == "__main__":
    coreui_config = parse_args()

    if coreui_config.reload:
        uvicorn.run(
            "server.core_ui:get_app",
            host="0.0.0.0",
            port=coreui_config.port,
            reload=True,
            factory=True,
        )
    else:
        app = get_app()
        uvicorn.run(app, host="0.0.0.0", port=coreui_config.port)

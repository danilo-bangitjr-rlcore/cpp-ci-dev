import os
import sys
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from server.config_api.config_routes import config_router
from server.coredinator_api.coredinator_routes import coredinator_router
from server.opc_api.opc_routes import opc_router


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
        self.app.include_router(coredinator_router, prefix="/api/v1/coredinator")

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

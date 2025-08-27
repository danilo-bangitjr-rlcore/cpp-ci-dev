import os
import sys

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from server.config_api.config import get_config, get_tag, get_tags
from server.opc_api.opc_routes import print_hello


class CoreUI:
    def __init__(self, dist_path: str | None=None):
        self.app = FastAPI()

        # Default dist path (dev mode)
        if dist_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            dist_path = os.path.join(base_dir, "client", "dist")

        # PyInstaller temp directory (frozen exe)
        meipass_path = getattr(sys, '_MEIPASS', None)
        if meipass_path:
            dist_path = os.path.join(meipass_path, "dist")

        dist_path = os.path.abspath(dist_path)

        # Serve assets if folder exists
        assets_path = os.path.join(dist_path, "assets")
        if os.path.exists(assets_path):
            self.app.mount("/app/assets", StaticFiles(directory=assets_path), name="assets")

        # API routes
        @self.app.get("/api/health")
        async def health_check():
            return {"status": "ok"}

        # Config operations
        @self.app.get("/api/configs/{config_name}")
        async def get_config_endpoint(config_name: str):
            return await get_config(config_name)

        # Tag operations
        @self.app.get("/api/configs/{config_name}/tags")
        async def get_tags_endpoint(config_name: str):
            return await get_tags(config_name)

        @self.app.get("/api/configs/{config_name}/tags/{tag_name}")
        async def get_tag_endpoint(config_name: str, tag_name: str):
            return await get_tag(config_name, tag_name)

        # Serve SPA
        index_html = os.path.join(dist_path, "index.html")

        @self.app.get("/app")
        async def spa_root():
            return FileResponse(index_html)

        @self.app.get("/app/{full_path:path}")
        async def spa_catch_all(full_path: str):
            return FileResponse(index_html)

    def get_app(self):
        print_hello()
        return self.app

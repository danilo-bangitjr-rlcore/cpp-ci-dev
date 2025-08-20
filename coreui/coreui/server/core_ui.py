import os
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

class CoreUI:
    def __init__(self, dist_path=None):
        self.app = FastAPI()

        # Default dist path (dev mode)
        if dist_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dist_path = os.path.join(base_dir, "client", "dist")

        # PyInstaller temp directory (frozen exe)
        if hasattr(sys, "_MEIPASS"):
            dist_path = os.path.join(sys._MEIPASS, "dist")

        dist_path = os.path.abspath(dist_path)

        # Serve assets if folder exists
        assets_path = os.path.join(dist_path, "assets")
        if os.path.exists(assets_path):
            self.app.mount("/app/assets", StaticFiles(directory=assets_path), name="assets")

        # API routes
        @self.app.get("/api/health")
        async def health_check():
            return {"status": "ok"}

        # Serve SPA
        index_html = os.path.join(dist_path, "index.html")

        @self.app.get("/app")
        async def spa_root():
            return FileResponse(index_html)

        @self.app.get("/app/{full_path:path}")
        async def spa_catch_all(full_path: str):
            return FileResponse(index_html)

    def get_app(self):
        return self.app

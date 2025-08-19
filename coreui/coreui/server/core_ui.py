import os
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

class CoreUI:
    def __init__(self, dist_path=None):
        self.app = FastAPI()

        # Default dist path (dev mode)
        if dist_path is None:
            dist_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dist"))

        # PyInstaller temp directory (frozen exe)
        if hasattr(sys, "_MEIPASS"):
            dist_path = os.path.join(sys._MEIPASS, "dist")

        dist_path = os.path.abspath(dist_path)  # ensure absolute path

        # Serve assets if folder exists
        assets_path = os.path.join(dist_path, "assets")
        assets_path = os.path.abspath(assets_path)
        if os.path.exists(assets_path):
            self.app.mount("/assets", StaticFiles(directory=assets_path), name="assets")
        else:
            print(f"Warning: Assets folder does NOT exist at {assets_path}. Skipping /assets mount.")

        # API routes
        @self.app.get("/api/health")
        async def health_check():
            return {"status": "ok"}

        # Frontend SPA mount
        if os.path.exists(dist_path):
            self.app.mount("/app", StaticFiles(directory=dist_path, html=True), name="frontend")
        else:
            print(f"Warning: Frontend dist folder does NOT exist at {dist_path}. Skipping /app mount.")

    def get_app(self):
        return self.app

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

# Mount the React build directory
app.mount("/static", StaticFiles(directory="../client/dist"), name="static")

# API routes
@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

# Catch-all route for React Router (SPA)
@app.get("/{path:path}")
async def serve_react_app(path: str):
    file_path = f"../client/dist/{path}"
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    # Return index.html for client-side routing
    return FileResponse("../client/dist/index.html")

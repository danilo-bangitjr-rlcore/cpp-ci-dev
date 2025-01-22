from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

app.mount("/", StaticFiles(directory="client/dist", html=True), name="static")


@app.get("/health")
async def health():
    return {"status": "OK", "time": f"{datetime.now(tz=UTC).isoformat()}"}


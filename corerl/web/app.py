from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}


@app.get("/health")
async def health():
    return {"status": "OK", "time": f"{datetime.now(tz=UTC).isoformat()}"}

app.mount("/", StaticFiles(directory="client/dist", html=True), name="static")


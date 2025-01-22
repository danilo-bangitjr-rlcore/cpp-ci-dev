from datetime import UTC, datetime

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    return {"status": "OK", "time": f"{datetime.now(tz=UTC).isoformat()}"}


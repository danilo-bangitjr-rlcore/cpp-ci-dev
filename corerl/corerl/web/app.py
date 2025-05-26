"""
This module contains the FastAPI application for CoreRL.
An optional environment variable COREIO_SQLITE_DB_PATH can be set to the path to the sqlite database from coreio.
The default path is /app/coreio-data/sqlite.db which is the path to the sqlite database in the compose.yml bound volume.

After installing corerl, you can start the web application by running the following command:

```sh
start_web_app

# or
fastapi dev corerl/web/app.py
```
"""

import importlib.metadata
import json
import logging
import sqlite3
import traceback
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Callable

import sqlalchemy
import yaml
from fastapi import FastAPI, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError

from corerl.config import DBConfig, MainConfig
from corerl.configs.errors import ConfigValidationErrors
from corerl.configs.loader import config_from_dict, config_to_json
from corerl.sql_logging.sql_logging import table_exists
from corerl.web import get_coreio_sqlite_path
from corerl.web.agent_manager import router as agent_manager
from corerl.web.agent_manager import shutdown_agents

# For debugging while running the server
_log = logging.getLogger("uvicorn.error")
_log.setLevel(logging.INFO)

logger = logging.getLogger("uvicorn")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CoreRL server is starting up.")
    yield
    logger.info("CoreRL server is shutting down.")
    shutdown_agents()
    logger.info("CoreRL server has shut down.")

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


class HealthResponse(BaseModel):
    time: str = datetime(year=2025, month=1, day=1, tzinfo=UTC).isoformat()
    version: str = version
    status: str
    message: str | None = None


class MessageResponse(BaseModel):
    message: str


@app.get("/")
async def redirect():
    response = RedirectResponse(url="/docs")
    return response


@app.get(
    "/api/corerl/health",
    response_model=HealthResponse,
    tags=["Health"],
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal Server Error", "model": str}},
)
async def health():
    """
    Health endpoint for CoreRL. Returns the current version number, server timestamp,
    and the status of the sqlite database.
    """
    status = "OK"
    message = None
    try:
        with sqlite3.connect(get_coreio_sqlite_path()) as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='CoreRLConfig';")
            rows = cur.fetchall()
            if len(rows) <= 0:
                status = "ERROR"
                message = "CoreIO SqliteDB connected but CoreRLConfig table not found"

    except Exception as e:
        logger.exception(f"Could not connect to CoreIO sqlite db '{get_coreio_sqlite_path()}'")
        status = "ERROR"
        message = str(e)


    return {"status": status, "message": message, "time": f"{datetime.now(tz=UTC).isoformat()}", "version": version}


@app.post(
    "/api/corerl/configuration/file",
    response_model=MainConfig,
    tags=["Configuration"],
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "content": {
                "application/json": {"example": {"detail": "<Error description>"}},
            }
        },
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
            "content": {
                "application/json": {"example": {"detail": "Unsupported Media Type"}},
            }
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "content": {
                "application/json": {"example": {"errors": "<Error description>"}},
            }
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal Server Error", "model": str},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/MainConfig"}},
                "application/yaml": {"schema": {"$ref": "#/components/schemas/MainConfig"}},
                "application/x-yaml": {"schema": {"$ref": "#/components/schemas/MainConfig"}},
            },
            "required": True,
        },
        "responses": {
            status.HTTP_200_OK: {
                "content": {
                    "application/json": {"schema": {"$ref": "#/components/schemas/MainConfig"}},
                    "application/yaml": {"schema": {"$ref": "#/components/schemas/MainConfig"}},
                }
            },
        },
    },
)
async def gen_config_file(request: Request, file: UploadFile | None = None):
    """
    Return a fully structured configuration as the response.
    The configuration format should be determined by an http header (application/yaml, application/json).
    """

    content_type = request.headers.get("Content-Type")
    if content_type == "application/json":
        data = await request.json()
    elif content_type in ["application/yaml", "application/x-yaml"]:
        body = await request.body()
        data = yaml.safe_load(body)
        if isinstance(data, str):
            # workaround for string input, yaml.safe_load may return a string if the payload is enclosed in a string
            data = yaml.safe_load(data)
    else:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Unsupported Media Type")

    try:
        res_config = config_from_dict(MainConfig, data)

        if isinstance(res_config, ConfigValidationErrors):
            return JSONResponse(
                content=json.loads(config_to_json(ConfigValidationErrors, res_config)),
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        json_config = json.loads(config_to_json(MainConfig, res_config))
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{str(e)}\n\n{tb}") from e

    accept_header = request.headers.get("accept")
    if accept_header is not None and "application/yaml" in accept_header:
        yaml_response = yaml.safe_dump(
            json_config,
            sort_keys=False,
        )
        yaml_response = (
            f"# This file was auto-generated by corerl@{version}\n# at {datetime.now(tz=UTC).isoformat()}\n"
            + f"# Do not make direct changes to the file.\n{yaml_response}"
        )
        return Response(yaml_response, media_type="application/yaml")
    else:
        return JSONResponse(json_config, media_type="application/json")


class DB_Status_Request(BaseModel):
    db_config: DBConfig
    table_name: str


class DB_Status_Response(BaseModel):
    db_status: bool
    table_status: bool
    has_connected: bool


@app.post("/api/corerl/verify-connection/db")
async def verify_connection_db(db_req: DB_Status_Request) -> DB_Status_Response:
    db_status = False
    table_status = False

    db_config = db_req.db_config

    url_object = sqlalchemy.URL.create(
        drivername=db_config.drivername,
        username=db_config.username,
        password=db_config.password,
        host=db_config.ip,
        port=db_config.port,
        database=db_config.db_name,
    )
    engine = sqlalchemy.create_engine(url_object)

    try:
        connection = engine.connect()
        db_status = True
        table_status = table_exists(engine, table_name=db_req.table_name)

        connection.close()
    except SQLAlchemyError as err:
        logger.info("Database connection unsuccessful", err)

    db_status = DB_Status_Response(db_status=db_status, table_status=table_status, has_connected=True)
    return db_status

app.include_router(agent_manager, prefix="/api/corerl/agents", tags=["Agent"])

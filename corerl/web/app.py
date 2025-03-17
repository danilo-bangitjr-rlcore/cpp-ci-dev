import importlib.metadata
import json
import logging
import traceback
from datetime import UTC, datetime
from typing import Any, Callable, List

import sqlalchemy
import yaml
from asyncua import Client
from asyncua.sync import Client as SyncClient
from fastapi import FastAPI, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError

from corerl.config import DBConfig, MainConfig
from corerl.configs.loader import config_from_dict, config_to_json
from corerl.sql_logging.sql_logging import table_exists
from corerl.utils.opc_connection import sync_browse_opc_nodes

# For debugging while running the server
_log = logging.getLogger("uvicorn.error")
_log.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

app = FastAPI()

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
    status: str = "OK"
    time: str = datetime(year=2025, month=1, day=1, tzinfo=UTC).isoformat()
    version: str = version


class MessageResponse(BaseModel):
    message: str


class OpcNodeDetail(BaseModel):
    val: Any
    DataType: str
    Identifier: str | int
    nodeid: str
    path: str
    key: str


class OpcNodeResponse(BaseModel):
    nodes: List[OpcNodeDetail]


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
    return {"status": "OK", "time": f"{datetime.now(tz=UTC).isoformat()}", "version": version}


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
            f"# This file was auto-generated by corerl.\n# Do not make direct changes to the file.\n{yaml_response}"
        )
        return Response(yaml_response, media_type="application/yaml")
    else:
        return JSONResponse(json_config, media_type="application/json")


@app.get(
    "/api/corerl/opc/nodes",
    tags=["Opc"],
)
async def read_search_opc(opc_url: str, query: str = "") -> OpcNodeResponse:
    with SyncClient(opc_url) as client:
        root = client.nodes.root
        opc_structure = sync_browse_opc_nodes(client, root)

    opc_variables = get_variables_from_dict(opc_structure)
    if query != "":
        query_lc = query.lower()
        opc_variables = [
            variable
            for variable in opc_variables
            if query_lc in variable.key.lower()
            or query_lc in variable.path.lower()
            or query_lc in variable.nodeid.lower()
        ]

    return OpcNodeResponse(nodes=opc_variables)


def get_variables_from_dict(opc_structure: dict) -> List[OpcNodeDetail]:
    _variables = []

    def traverse(node: dict, path: str = "", parent_key: str = ""):
        if "val" in node.keys():
            node["path"] = path
            node["key"] = parent_key
            opc_node_detail = OpcNodeDetail(**node)
            _variables.append(opc_node_detail)
        else:
            if path == "":
                path = parent_key
            else:
                path = path + f"/{parent_key}"

            for key, value in node.items():
                # Variables named Opc.Ua are too long.
                # Hardcoding skipping those variables.
                if key == "Opc.Ua":
                    continue
                traverse(value, path=path, parent_key=key)

    traverse(opc_structure)
    return _variables

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
        database=db_config.db_name
    )
    engine = sqlalchemy.create_engine(url_object)

    try:
        connection = engine.connect()
        db_status = True
        table_status = table_exists(engine, table_name = db_req.table_name)

        connection.close()
    except SQLAlchemyError as err:
        _log.info("Database connection unsuccessful", err)

    db_status = DB_Status_Response(
        db_status = db_status,
        table_status = table_status,
        has_connected = True
    )
    return db_status


class OPC_Status_Response(BaseModel):
    opc_status: bool
    has_connected: bool

# Only 1 attribute for now, but using BaseModel for fastAPI conventions
class OPC_Status_Request(BaseModel):
    opc_url: str

@app.post("/api/corerl/verify-connection/opc")
async def verify_connection_opc(opc_req: OPC_Status_Request) -> OPC_Status_Response:
    opc_status = False
    opc_client = Client(opc_req.opc_url)

    try:
        await opc_client.connect()
        await opc_client.disconnect()
        opc_status = True

    # There are many different errors that can ocurr if connection is unsuccessful
    except Exception as err:
        _log.info("OPC Connection unsuccessful", err)

    opc_status = OPC_Status_Response(opc_status=opc_status, has_connected=True)
    return opc_status

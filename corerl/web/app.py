import json
import logging
import traceback
from datetime import UTC, datetime
from typing import Any, List

import yaml
from asyncua.sync import Client
from fastapi import FastAPI, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from corerl.config import MainConfig
from corerl.configs.loader import config_from_dict, config_to_json
from corerl.utils.opc_connection import sync_browse_opc_nodes

# For debugging while running the server
_log = logging.getLogger("uvicorn.error")
_log.setLevel(logging.DEBUG)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str = "OK"
    time: str = datetime.now(tz=UTC).isoformat()


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

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal Server Error", "model": str}},
)
async def health():
    return {"status": "OK", "time": f"{datetime.now(tz=UTC).isoformat()}"}


@app.post("/api/file")
async def test_file(file: UploadFile):
    return {"filename": file.filename}


@app.post(
    "/api/configuration/file",
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
        raise HTTPException(status_code=415, detail="Unsupported Media Type")

    try:
        res_config = config_from_dict(MainConfig, data)
        json_config = json.loads(config_to_json(MainConfig, res_config))
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"{str(e)}\n\n{tb}") from e

    accept_header = request.headers.get("accept")
    if accept_header is not None and "application/yaml" in accept_header:
        yaml_response = yaml.safe_dump(
            json_config,
            sort_keys=False,
        )
        return Response(yaml_response, media_type="application/yaml")
    else:
        return JSONResponse(json_config, media_type="application/json")


@app.get("/api/opc/nodes",
         tags=["Opc"],
         )
async def read_search_opc(opc_url: str, query: str = "") -> OpcNodeResponse:
    with Client(opc_url) as client:
        root = client.nodes.root
        opc_structure = sync_browse_opc_nodes(client, root)

    opc_variables = get_variables_from_dict(opc_structure)
    if query != "":
        opc_variables = [
            variable
            for variable in opc_variables
            if query in variable.key or query in variable.path
        ]

    return OpcNodeResponse(nodes = opc_variables)
    # Also add e2e tests

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


app.mount("/", StaticFiles(directory="client/dist", html=True, check_dir=False), name="static")

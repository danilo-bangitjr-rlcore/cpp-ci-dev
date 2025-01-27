from datetime import UTC, datetime
import time
from pathlib import Path

from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from json import JSONDecodeError
import json
import yaml
import tempfile
import logging

from corerl.config import MainConfig
from corerl.configs.loader import direct_load_config, config_to_dict, config_to_json
from datetime import timedelta

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

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {"status": "OK", "time": f"{datetime.now(tz=UTC).isoformat()}"}

@app.get("/configuration/test")
async def gen_config_test(request: Request) -> Response:
    config = MainConfig()
    raw_yaml = yaml.dump(config_to_dict(MainConfig, config))
    response = Response(content=raw_yaml, media_type="application/yaml")
    return response

@app.post("/configuration/file", response_model=MainConfig)
async def gen_config_file(item: dict, request: Request):
    """
    Return a fully structured configuration as the response.
    The configuration format should be determined by an http header (application/yaml, application/json).
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as fp:
        yaml.safe_dump(item, fp, sort_keys=False, default_flow_style=False)
        fp.flush()
        res_config = direct_load_config(MainConfig, "", fp.name)

    json_config = json.loads(config_to_json(MainConfig, res_config))

    accept_header = request.headers.get("accept")
    if "application/yaml" in accept_header:
        yaml_response = yaml.safe_dump(
            json_config,
            sort_keys=False,
            default_flow_style=False
        )
        return Response(yaml_response, media_type="application/yaml")
    else:
        return JSONResponse(json_config, media_type="application/json")

app.mount("/", StaticFiles(directory="client/dist", html=True, check_dir=False), name="static")


from datetime import UTC, datetime
import time
from pathlib import Path

from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from json import JSONDecodeError
import yaml
import tempfile
import logging

from corerl.config import MainConfig
from corerl.configs.loader import direct_load_config, config_to_dict, _walk_config_and_interpolate

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

@app.post("/items/")
async def create_item(item: Item, request: Request):
    response_data = {
        "message": "Item received",
        "item": item.model_dump()
    }

    accept_header = request.headers.get("accept")
    if "application/yaml" in accept_header:
        yaml_response = yaml.dump(response_data, default_flow_style=False)
        return Response(yaml_response, media_type="application/yaml")
    else:
        return JSONResponse(response_data)


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

    # return str(config_to_dict(MainConfig, res_config))
    return res_config

    # content_type = request.headers.get("Accept")
    # if content_type == "application/json":
    #     try:
    #         data = await request.json()
    #         return {"message": "Got JSON", "data": data}
    #     except JSONDecodeError as json_error:
    #         raise HTTPException(status_code=400, detail=f"Invalid JSON: {json_error}")
    #
    # elif content_type == "application/yaml":
    #     try:
    #         body = await request.body()
    #         data = yaml.safe_load(body)
    #         return {"message": "Got YAML", "data": data}
    #     except yaml.YAMLError as yaml_error:
    #         raise HTTPException(status_code=400, detail=f"Invalid YAML: {yaml_error}")
    # else:
    #     raise HTTPException(status_code=415, detail="Unsupported Content-Type")

    # config_payload = MainConfig()
    # return config_payload

# Test for this. Have different files (yaml, json). Test for error too, give resonable error message


app.mount("/", StaticFiles(directory="client/dist", html=True, check_dir=False), name="static")


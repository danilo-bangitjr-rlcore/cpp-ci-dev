import json
import logging
import tempfile
from datetime import UTC, datetime

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from corerl.config import MainConfig
from corerl.configs.loader import config_to_json, direct_load_config

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

@app.get("/health")
async def health():
    return {"status": "OK", "time": f"{datetime.now(tz=UTC).isoformat()}"}

@app.post("/api/configuration/file", response_model=MainConfig)
async def gen_config_file(request: Request):
    """
    Return a fully structured configuration as the response.
    The configuration format should be determined by an http header (application/yaml, application/json).
    """

    content_type = request.headers.get("Content-Type")
    if content_type == "application/json":
        data = await request.json()
    elif content_type == "application/yaml":
        body = await request.body()
        data = yaml.safe_load(body)
    else:
        raise HTTPException(status_code=415, detail="Unsupported Media Type")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as fp:
        yaml.safe_dump(data, fp, sort_keys=False, default_flow_style=False)
        fp.flush()
        res_config = direct_load_config(MainConfig, "", fp.name)

    json_config = json.loads(config_to_json(MainConfig, res_config))

    accept_header = request.headers.get("accept")
    if accept_header is not None and "application/yaml" in accept_header:
        yaml_response = yaml.safe_dump(
            json_config,
            sort_keys=False,
        )
        return Response(yaml_response, media_type="application/yaml")
    else:
        return JSONResponse(json_config, media_type="application/json")

app.mount("/", StaticFiles(directory="client/dist", html=True, check_dir=False), name="static")


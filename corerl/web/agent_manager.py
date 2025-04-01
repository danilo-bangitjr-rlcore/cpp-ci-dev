"""
Agent Manager endpoints for starting and stopping our `corerl_main` scripts.
Currently, this logic will only function as expected using one web worker!
"""

import asyncio
import json
import logging
import os
import sqlite3
import subprocess
import tempfile
import uuid
from collections import deque
from pathlib import Path
from threading import Thread
from typing import Dict, List, Literal, Optional, TypedDict

import yaml
from anyio import ClosedResourceError
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status, websockets
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from corerl.config import MainConfig
from corerl.configs.errors import ConfigValidationErrors
from corerl.configs.loader import config_from_dict, config_to_json
from corerl.utils.time import now_iso
from corerl.web import get_coreio_sqlite_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

router = APIRouter()


class WSMessage(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    time: str = Field(default_factory=now_iso)
    type: Literal["log"] = "log"
    message: str | None = None


class CoreRLProcess(TypedDict):
    """TypedDict for CoreRL processes managed within the FastAPI server."""

    log_buffer: deque[WSMessage]
    process: subprocess.Popen | None


processes: Dict[str, CoreRLProcess] = {}
clients: Dict[str, set[WebSocket]] = {}


class StartAgentRequestPayload(BaseModel):
    """Payload for starting an agent."""

    extra_options: Optional[List[str]] = None


async def start_proc_read_stream(yaml_config: str, config_id: str, req_payload: StartAgentRequestPayload):
    """Start our process, send all stdout and stderr to connected websocket clients."""
    Path("config").mkdir(exist_ok=True)

    # cannot delete the temporary configuration file within the context manager because
    # there is an unspecified delay between context manager exit and subprocess's python script running
    # instead, remove the file after some output has been emitted from the process
    with tempfile.NamedTemporaryFile(suffix=".yaml", dir="config", delete=False) as temp_file:
        temp_file.write(yaml_config.encode())
        temp_file.seek(0)
        logging.info(f"Temporary config file created at: {temp_file.name}")

        cmd = [
            "corerl_main",
            "--config-name",
            temp_file.name,
        ]
        if req_payload.extra_options:
            cmd.extend(req_payload.extra_options)

    unbuffered_env = os.environ.copy()
    unbuffered_env["PYTHONUNBUFFERED"] = "true"
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # redirect stderr to stdout
        universal_newlines=False,
        start_new_session=True,
        env=unbuffered_env,
    )

    processes[config_id]["process"] = process
    log_buffer = processes[config_id]["log_buffer"]

    stream = process.stdout
    assert stream is not None

    while process.poll() is None:
        line = stream.readline()
        if not line:
            continue

        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

        line = line.decode("utf-8").rstrip()
        ws_msg = WSMessage(message=line)
        log_buffer.append(ws_msg)

        logger.debug(f"[{config_id}] {line}")

        if config_id not in clients:
            continue

        for client in clients[config_id]:
            if client.application_state != websockets.WebSocketState.CONNECTED:
                continue
            try:
                await client.send_json(json.loads(ws_msg.model_dump_json()))
            except ClosedResourceError:
                pass
            except Exception:
                logger.exception("Error sending websocket message")
                continue


@router.post("/{config_id}/start")
def agent_start(req_payload: StartAgentRequestPayload, config_id: str):
    """
    Start the agent. This post request should take the configuration UUID
    and start the core-rl agent, tracking the process id.
    """

    if config_id in processes:
        return {"status": "ERROR", "message": "agent already started, must stop before restart"}
    try:
        processes[config_id] = CoreRLProcess({"log_buffer": deque(maxlen=20000), "process": None})
        # get the raw MainConfig JSON payload from CoreIO's sqlite DB
        with sqlite3.connect(get_coreio_sqlite_path()) as conn:
            cur = conn.cursor()
            cur.execute("SELECT config FROM CoreRLConfig WHERE id=?", (config_id,))
            config = cur.fetchone()
        if not config:
            return {
                "status": "ERROR",
                "message": f"Config with id {config_id} not found.",
                "sqlite_path": get_coreio_sqlite_path(),
            }

        # validate the raw JSON configuration and convert it to yaml
        raw_json_config = json.loads(config[0])
        res_config = config_from_dict(MainConfig, raw_json_config)
        if isinstance(res_config, ConfigValidationErrors):
            return JSONResponse(
                content=json.loads(config_to_json(ConfigValidationErrors, res_config)),
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )
        json_config = json.loads(config_to_json(MainConfig, res_config))
        yaml_config = yaml.safe_dump(json_config, sort_keys=False)

        t = Thread(target=asyncio.run, args=(start_proc_read_stream(yaml_config, config_id, req_payload),), daemon=True)
        t.start()

        return {"status": "SUCCESS"}

    except Exception as e:
        return {"status": "ERROR", "message": str(e), "sqlite_path": get_coreio_sqlite_path()}


@router.post("/{config_id}/stop")
def agent_stop(config_id: str):
    """
    Stop the agent.
    """
    if config_id in processes:
        coreRLProcess = processes[config_id]
        process = coreRLProcess["process"]

        if process and process.poll() is None:
            process.terminate()
        del processes[config_id]

        return {"status": "stopped"}
    return {"status": "not running"}


@router.get("/{config_id}/status")
def agent_status(config_id: str):
    """
    Get the status of the agent.
    """
    if config_id in processes:
        coreRLProcess = processes[config_id]
        process = coreRLProcess["process"]
        if not process:
            return {"status": "starting_up"}
        if process.poll() is None:
            # Process is still running
            return {"status": "running", "pid": process.pid}
        else:
            # Process has terminated
            return {"status": "stopped", "pid": process.pid}
    return {"status": "not_started", "pid": None}


@router.get("/")
def agent_list():
    """
    List all agents.
    """
    return {"managed": list(processes.keys())}


@router.websocket("/{config_id}/ws")
async def websocket_endpoint(websocket: WebSocket, config_id: str, skip_log_buffer: bool = False):
    await websocket.accept()

    if config_id not in clients:
        clients[config_id] = set()
    clients[config_id].add(websocket)

    if not skip_log_buffer and config_id in processes:
        coreRLProcess = processes[config_id]
        log_buffer = coreRLProcess["log_buffer"]
        for ws_msg in log_buffer:
            try:
                await websocket.send_json(json.loads(ws_msg.model_dump_json()))
            except Exception:
                logger.exception("Error sending websocket message")
                continue
    try:
        while websocket.application_state != websockets.WebSocketState.DISCONNECTED:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        if config_id not in clients:
            return
        clients[config_id].discard(websocket)

        if not clients[config_id]:
            # Clean up if no more clients
            del clients[config_id]


def shutdown_agents():
    """
    Shutdown all agents.
    """
    for config_id in list(processes.keys()):
        agent_stop(config_id)
    logger.info("All agents stopped.")

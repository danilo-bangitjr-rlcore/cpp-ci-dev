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
from queue import Empty, Queue
from threading import Thread
from typing import IO, Dict, List, Literal, Optional, TypedDict

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

logger = logging.getLogger("uvicorn")

router = APIRouter()


class WSMessage(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    time: str = Field(default_factory=now_iso)
    type: Literal["log"] | Literal["return_code"] = "log"
    message: str | None = None


class CoreRLProcess(TypedDict):
    """TypedDict for CoreRL processes managed within the FastAPI server."""

    log_buffer: deque[WSMessage]
    process: subprocess.Popen | None
    extra_options: List[str]


processes: Dict[str, CoreRLProcess] = {}
clients: Dict[str, set[WebSocket]] = {}


class StartAgentRequestPayload(BaseModel):
    """Payload for starting an agent."""

    extra_options: Optional[List[str]] = None

def output_reader(stream: IO[bytes], queue: Queue):
    for line in iter(stream.readline, b""):
        queue.put(line.decode("utf-8"))


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
            processes[config_id]["extra_options"] = req_payload.extra_options

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

    outq = Queue()
    t = Thread(target=output_reader, args=(stream, outq), daemon=True)
    t.start()

    while (return_code := process.poll()) is None or outq.qsize() > 0:
        output_lines = []
        while True:
            try:
                line = outq.get(block=False)
                output_lines.append(line)
            except Empty:
                break
        if not output_lines:
            await asyncio.sleep(1.0)
            continue

        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

        msg_payload = "".join(output_lines)

        ws_msg = WSMessage(message=msg_payload)
        log_buffer.append(ws_msg)

        logger.debug(f"[{config_id}] {msg_payload}")

        if config_id not in clients:
            continue

        for client in clients[config_id]:
            if client.application_state != websockets.WebSocketState.CONNECTED:
                continue
            try:
                await client.send_json(json.loads(ws_msg.model_dump_json()))
            except WebSocketDisconnect:
                pass
            except ClosedResourceError:
                pass
            except Exception:
                logger.exception("Error sending websocket message")
                continue

    # poll has returned, emit that the agent has died
    if config_id in clients:
        ws_msg = WSMessage(type="return_code", message=f"{return_code}")
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

    t.join()

class AgentStartErrorResponse(BaseModel):
    status: Literal["error"]
    message: str
    sqlite_path: Optional[str]


class AgentStartSuccessResponse(BaseModel):
    status: Literal["success"]


@router.post(
    "/{config_id}/start",
    tags=["Agent"],
    responses={
        status.HTTP_200_OK: {"model": AgentStartSuccessResponse},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": AgentStartErrorResponse},
        status.HTTP_400_BAD_REQUEST: {"model": AgentStartErrorResponse},
    },
)
def agent_start(req_payload: StartAgentRequestPayload, config_id: str):
    """
    Start the agent. This post request should take the configuration UUID
    and start the core-rl agent, tracking the process id.
    """

    if config_id in processes:
        return JSONResponse(
            {"status": "error", "message": "agent already started, must stop before restart"},
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    try:
        processes[config_id] = CoreRLProcess({"log_buffer": deque(maxlen=20000), "process": None, "extra_options": []})
        # get the raw MainConfig JSON payload from CoreIO's sqlite DB
        with sqlite3.connect(get_coreio_sqlite_path()) as conn:
            cur = conn.cursor()
            cur.execute("SELECT config FROM CoreRLConfig WHERE id=?", (config_id,))
            config = cur.fetchone()
        if not config:
            raise Exception(f"Config with id {config_id} not found.")

        # validate the raw JSON configuration and convert it to yaml
        raw_json_config = json.loads(config[0])
        res_config = config_from_dict(MainConfig, raw_json_config)
        if isinstance(res_config, ConfigValidationErrors):
            del processes[config_id]
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "Configuration failed core-rl validation.",
                    "sqlite_path": get_coreio_sqlite_path(),
                },
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )
        json_config = json.loads(config_to_json(MainConfig, res_config))
        yaml_config = yaml.safe_dump(json_config, sort_keys=False)

        t = Thread(target=asyncio.run, args=(start_proc_read_stream(yaml_config, config_id, req_payload),))
        t.start()

        return JSONResponse({"status": "success"})

    except Exception as e:
        del processes[config_id]
        return JSONResponse(
            {"status": "error", "message": str(e), "sqlite_path": get_coreio_sqlite_path()},
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class AgentStopResponse(BaseModel):
    status: Literal["stopped"] | Literal["not_running"]


@router.post("/{config_id}/stop", response_model=AgentStopResponse)
def agent_stop(config_id: str):
    """
    Stop the agent.
    """
    if config_id in processes:
        logger.info(f"Shutting down agent {config_id}")
        coreRLProcess = processes[config_id]
        process = coreRLProcess["process"]

        if process and process.poll() is None:
            process.terminate()
        del processes[config_id]

        return JSONResponse({"status": "stopped"})
    return JSONResponse({"status": "not_running"})


class AgentStatusResponse(BaseModel):
    status: Literal["starting_up"] | Literal["running"] | Literal["dead"] | Literal["not_started"]
    pid: int | None
    extra_options: Optional[list[str]]


@router.get("/{config_id}/status", response_model=AgentStatusResponse)
def agent_status(config_id: str):
    """
    Get the status of the agent.
    """
    if config_id in processes:
        coreRLProcess = processes[config_id]
        process = coreRLProcess["process"]
        extra_options = coreRLProcess["extra_options"]
        if not process:
            return JSONResponse({"status": "starting_up", "pid": None, "extra_options": extra_options})
        if process.poll() is None:
            # Process is still running
            return JSONResponse({"status": "running", "pid": process.pid, "extra_options": extra_options})
        else:
            # Process has terminated
            return JSONResponse({"status": "dead", "pid": process.pid, "extra_options": extra_options})
    return JSONResponse({"status": "not_started", "pid": None})


class AgentListResponse(BaseModel):
    managed: list[str]


@router.get("/", response_model=AgentListResponse)
def agent_list():
    """
    List all agents.
    """
    return JSONResponse({"managed": list(processes.keys())})


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
    logger.info("Shutting down agents...")
    for config_id in list(processes.keys()):
        agent_stop(config_id)
    logger.info("All agents stopped.")

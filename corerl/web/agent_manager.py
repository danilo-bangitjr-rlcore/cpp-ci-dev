import asyncio
import json
import logging
import os
import sqlite3
import subprocess
import tempfile
import threading
from collections import deque
from pathlib import Path
from typing import IO, Dict, List, Optional, TypedDict

import yaml
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status, websockets
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from corerl.config import MainConfig
from corerl.configs.errors import ConfigValidationErrors
from corerl.configs.loader import config_from_dict, config_to_json
from corerl.web import get_coreio_sqlite_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

router = APIRouter()


class CoreRLProcess(TypedDict):
    """TypedDict for CoreRL processes managed within the FastAPI server."""

    process: subprocess.Popen
    thread: threading.Thread
    log_buffer: deque[str]
    temp_config_file: str


processes: Dict[str, CoreRLProcess] = {}
clients: Dict[str, set[WebSocket]] = {}


async def read_stream(stream: IO[bytes] | None, config_id: str, temp_config_file: str):
    """Reads a subprocess stream and send it to connected websocket clients."""
    coreRLProcess = processes[config_id]
    process = coreRLProcess["process"]
    log_buffer = coreRLProcess["log_buffer"]

    while process.poll() is None:
        line = stream.readline() if stream else None
        if not line:
            continue
        line = line.decode("utf-8").rstrip()
        logger.debug(f"[{config_id}] {line}")
        log_buffer.appendleft(line)
        if config_id in clients:
            for client in clients[config_id]:
                try:
                    await client.send_text(line)
                except Exception:
                    logger.exception("Error sending websocket message")
                    continue

        if os.path.exists(temp_config_file):
            os.remove(temp_config_file)

    else:
        # process has terminated
        if os.path.exists(temp_config_file):
            os.remove(temp_config_file)


class StartAgentRequestPayload(BaseModel):
    """Payload for starting an agent."""

    extra_options: Optional[List[str]] = None


@router.post("/{config_id}/start")
def agent_start(req_payload: StartAgentRequestPayload, config_id: str):
    """
    Start the agent. This post request should take the configuration UUID
    and start the core-rl agent, tracking the process id.
    """

    try:
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

        Path("config").mkdir(exist_ok=True)

        # Cannot delete the file within the context manager because subprocess may take some time
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

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # redirect stderr to stdout
                universal_newlines=False,
                start_new_session=True,
            )

        logger.info(f"Started agent with PID: {process.pid}")

        stdout_thread = threading.Thread(
            target=asyncio.run,
            args=(
                read_stream(
                    process.stdout,
                    config_id,
                    temp_file.name,
                ),
            ),
            daemon=True,
            name=f"corerl_agent_{config_id}_stdout",
        )
        processes[config_id] = {
            "process": process,
            "thread": stdout_thread,
            # used to store the last 100 lines of log output for each agent
            "log_buffer": deque(maxlen=100),
            "temp_config_file": temp_file.name,
        }

        stdout_thread.start()

        return {"status": "SUCCESS", "pid": process.pid}

    except Exception as e:
        return {"status": "ERROR", "message": str(e), "sqlite_path": get_coreio_sqlite_path}


@router.post("/{config_id}/stop")
def agent_stop(config_id: str):
    """
    Stop the agent.
    """
    if config_id in processes:
        coreRLProcess = processes[config_id]
        process = coreRLProcess["process"]
        thread = coreRLProcess["thread"]

        if process.poll() is None:
            process.terminate()
        thread.join(timeout=5)
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
        for line in log_buffer:
            try:
                await websocket.send_text(line)
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

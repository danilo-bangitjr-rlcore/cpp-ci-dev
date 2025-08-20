import logging

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger("uvicorn")

router = APIRouter()

class StartAgentRequestPayload(BaseModel):
    ...


@router.post("/{config_id}/start")
def agent_start(req_payload: StartAgentRequestPayload, config_id: str):
    ...


@router.post("/{config_id}/stop")
def agent_stop(config_id: str):
    ...


@router.get("/{config_id}/status")
def agent_status(config_id: str):
    ...


@router.get("/")
def agent_list():
    ...

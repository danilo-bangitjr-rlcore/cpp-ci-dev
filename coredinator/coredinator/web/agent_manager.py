import logging

from fastapi import APIRouter
from pydantic import BaseModel

from coredinator.agent.agent_manager import AgentManager

logger = logging.getLogger("uvicorn")

router = APIRouter()
agent_manager = AgentManager()

class StartAgentRequestPayload(BaseModel):
    ...


@router.post("/{config_id}/start")
def agent_start(req_payload: StartAgentRequestPayload, config_id: str):
    agent_manager.start_agent(config_id)


@router.post("/{config_id}/stop")
def agent_stop(config_id: str):
    agent_manager.stop_agent(config_id)


@router.get("/{config_id}/status")
def agent_status(config_id: str):
    return agent_manager.get_agent_status(config_id)


@router.get("/")
def agent_list():
    return agent_manager.list_agents()

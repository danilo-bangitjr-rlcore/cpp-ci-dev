import logging
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from coredinator.agent.agent_manager import AgentManager
from coredinator.agent.agent_process import AgentID

logger = logging.getLogger("uvicorn")

router = APIRouter()
agent_manager = AgentManager()

class StartAgentRequestPayload(BaseModel):
    config_path: Path


@router.post("/start")
def agent_start(req_payload: StartAgentRequestPayload):
    return agent_manager.start_agent(req_payload.config_path)


@router.post("/{agent_id}/stop")
def agent_stop(agent_id: AgentID):
    agent_manager.stop_agent(agent_id)


@router.get("/{agent_id}/status")
def agent_status(agent_id: AgentID):
    return agent_manager.get_agent_status(agent_id)


@router.get("/")
def agent_list():
    return agent_manager.list_agents()

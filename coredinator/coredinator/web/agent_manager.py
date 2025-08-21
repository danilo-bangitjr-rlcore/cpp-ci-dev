from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from coredinator.agent.agent_manager import AgentID, AgentManager

router = APIRouter()


# Dependency injection for agent_manager
def get_agent_manager(request: Request) -> AgentManager:
    return request.app.state.agent_manager


class StartAgentRequestPayload(BaseModel):
    config_path: Path


@router.post("/start")
def agent_start(req_payload: StartAgentRequestPayload, request: Request):
    agent_manager = get_agent_manager(request)
    cfg = req_payload.config_path
    # Treat bad input as client error
    if not cfg.exists():
        raise HTTPException(status_code=400, detail=f"Config file not found at {cfg}")

    try:
        return agent_manager.start_agent(cfg)
    except FileNotFoundError as e:
        # Missing executables or other FileNotFoundError -> 400 Bad Request
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/{agent_id}/stop")
def agent_stop(agent_id: AgentID, request: Request):
    agent_manager = get_agent_manager(request)
    agent_manager.stop_agent(agent_id)


@router.get("/{agent_id}/status")
def agent_status(agent_id: AgentID, request: Request):
    agent_manager = get_agent_manager(request)
    return agent_manager.get_agent_status(agent_id)


@router.get("/")
def agent_list(request: Request):
    agent_manager = get_agent_manager(request)
    return agent_manager.list_agents()

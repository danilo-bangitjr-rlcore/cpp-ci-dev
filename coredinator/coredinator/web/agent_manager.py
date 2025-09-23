from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from coredinator.agent.agent import Agent
from coredinator.agent.agent_manager import AgentID, AgentManager
from coredinator.demo.tep_demo_agent import TEPDemoAgent
from coredinator.service.protocols import ServiceID

router = APIRouter()


# Dependency injection for agent_manager
def get_agent_manager(request: Request) -> AgentManager:
    return request.app.state.agent_manager


class StartAgentRequestPayload(BaseModel):
    config_path: Path
    coreio_id: ServiceID | None = None


@router.post("/start")
def agent_start(req_payload: StartAgentRequestPayload, request: Request):
    agent_manager = get_agent_manager(request)
    cfg = req_payload.config_path
    return _start_agent(agent_manager, cfg, coreio_service_id=req_payload.coreio_id)


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


@router.post("/demos/tep/start")
def demo_tep_start(req_payload: StartAgentRequestPayload, request: Request):
    agent_manager = get_agent_manager(request)
    cfg = req_payload.config_path
    return _start_agent(agent_manager, cfg, TEPDemoAgent, coreio_service_id=req_payload.coreio_id)

@router.post("/demos/tep/{agent_id}/stop")
def demo_tep_stop(agent_id: AgentID, request: Request):
    agent_manager = get_agent_manager(request)
    agent_manager.stop_agent(agent_id)

@router.get("/demos/tep/{agent_id}/status")
def demo_tep_status(agent_id: AgentID, request: Request):
    agent_manager = get_agent_manager(request)
    return agent_manager.get_agent_status(agent_id)


def _start_agent(
    agent_manager: AgentManager,
    cfg: Path,
    agent_factory: type[Agent] | None = None,
    coreio_service_id: ServiceID | None = None,
):
    # Treat bad input as client error
    if not cfg.exists():
        raise HTTPException(status_code=400, detail=f"Config file not found at {cfg}")

    if not agent_factory:
        agent_factory = Agent

    try:
        return agent_manager.start_agent(cfg, agent_factory, coreio_service_id=coreio_service_id)
    except FileNotFoundError as e:
        # Missing executables or other FileNotFoundError -> 400 Bad Request
        raise HTTPException(status_code=400, detail=str(e)) from e

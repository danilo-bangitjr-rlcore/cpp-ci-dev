# ruff: noqa: B008
import json
import logging
from typing import Any

import httpx
from fastapi import APIRouter, Body, HTTPException, Request, Response
from pydantic import BaseModel
from server.config_api.config_routes import get_all_clean_configs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)


class Payload(BaseModel):
    config_path: str
    coreio_id: str | None = None

class ValidationErrorDetail(BaseModel):
    loc: list[Any]
    msg: str
    type: str

class ValidationErrorResponse(BaseModel):
    detail: list[ValidationErrorDetail]

class NotFoundResponse(BaseModel):
    detail: str

class AgentWithConfigResponse(BaseModel):
    agents: list[str]

SuccessResponse = str
InternalServerErrorResponse = str

COREDINATOR_BASE = "http://localhost:7000"
PROXY_PREFIX = "/api/v1/coredinator"

coredinator_router = APIRouter(
    tags=["Coredinator Proxy"],
)

# Hop-by-hop headers are only valid for a single transport-level connection
# and must not be forwarded by proxies (RFC 2616 §13.5.1).
# Examples include "Connection", "Keep-Alive", "Transfer-Encoding", etc.
# Forwarding them can break request/response handling, so we strip them out.

# Drop Content-Length: the upstream value may be wrong after proxying.

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
}


def clean_headers(orig: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in orig.items() if k.lower() not in HOP_BY_HOP}

async def proxy_request(request: Request, path: str, body: dict | None = None) -> Response:
    """Core proxy logic handling all HTTP methods, body, headers, and redirect rewriting."""
    client: httpx.AsyncClient = request.app.state.httpx_client
    target_url = f"{COREDINATOR_BASE}/{path}"

    req_headers = clean_headers(dict(request.headers))
    params = dict(request.query_params)

    # Determine content to send
    json_data = body if body else None
    raw_body = await request.body() if json_data is None else None

    resp = await client.request(
        request.method,
        target_url,
        headers=req_headers,
        params=params,
        json=json_data,
        content=raw_body,
        follow_redirects=False,
    )

    try:
        resp_body = resp.json()
    except Exception:
        resp_body = resp.text

    return Response(
        content=resp_body if isinstance(resp_body, (str, bytes)) else json.dumps(resp_body),
        status_code=resp.status_code,
        headers=clean_headers(dict(resp.headers)),
        media_type=resp.headers.get("content-type"),
    )


# ----------------- Routes -----------------
@coredinator_router.get(
    "/agents/missing-config",
    response_model=AgentWithConfigResponse,
    responses={
        200: {
            "description": "List of agents active in coredinator but missing a clean config.",
            "content": {
                "application/json": {
                    "example": {
                        "agents": ["orphan_agent_1", "orphan_agent_2"],
                    },
                },
            },
        },
        500: {"model": InternalServerErrorResponse},
    },
    summary="Get Agents Missing Config",
    description="Retrieve agents that are active in coredinator but do not have a clean configuration available.",
)
async def get_agents_missing_config(request: Request):
    """Get agents active in coredinator but missing a clean config."""
    try:
        # Get clean configs
        clean_configs_response = await get_all_clean_configs()

        # Parse response body properly
        if hasattr(clean_configs_response, "body"):
            body = clean_configs_response.body
            # Convert memoryview to bytes if needed
            # Solves pyright error: Expression of type 'bytes | memoryview' cannot be assigned to declared type 'bytes'
            if isinstance(body, memoryview):
                body = bytes(body)
            clean_configs_data = json.loads(body)
        else:
            clean_configs_data = clean_configs_response

        configs = clean_configs_data.get("configs", []) if isinstance(clean_configs_data, dict) else []
        clean_configs = set(configs)

        # Get active agents from coredinator
        client: httpx.AsyncClient = request.app.state.httpx_client
        resp = await client.get(f"{COREDINATOR_BASE}/api/agents/")
        content_type = resp.headers.get("content-type", "")
        coredinator_data = resp.json() if content_type.startswith("application/json") else resp.text

        # Extract agent names
        if isinstance(coredinator_data, dict) and "agents" in coredinator_data:
            active_agents = set(coredinator_data["agents"])
        elif isinstance(coredinator_data, list):
            active_agents = set(coredinator_data)
        else:
            active_agents = set()

        # Return agents missing config
        return {"agents": sorted(active_agents - clean_configs)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agents missing config: {e!s}") from e


# ----------------- Keep these routes on the bottom since they are a catchall-----------------
@coredinator_router.api_route(
    "/{path:path}",
    methods=["GET"],
    summary="Proxy requests (no body) to Service B",
    response_model=SuccessResponse,
    responses={
        422: {"model": ValidationErrorResponse},
        404: {"model": NotFoundResponse},
        500: {"model": InternalServerErrorResponse},
    },
)
async def proxy_no_body(path: str, request: Request):
    return await proxy_request(request, path)


@coredinator_router.api_route(
    "/{path:path}",
    methods=["POST"],
    summary="Proxy requests (with body) to Service B",
    response_model=SuccessResponse,
    responses={
        422: {"model": ValidationErrorResponse},
        404: {"model": NotFoundResponse},
        500: {"model": InternalServerErrorResponse},
    },
)
async def proxy_with_body(
    path: str,
    request: Request,
    body: Payload = Body(None, description="Optional JSON body to forward"),
):
    return await proxy_request(request, path, body.model_dump() if body else None)

